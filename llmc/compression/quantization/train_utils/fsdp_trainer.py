from typing import Callable, Dict, List, Optional, Tuple, Union

# isort: on
import numpy as np
import torch
from packaging import version
from torch import nn
import torch.nn.functional as F

from .train_utils import SGDG
from ..rotate_utils import ActRotater, RotateModule, SmoothModule
# Integrations must be imported before ML frameworks:
# isort: off
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    TrainerCallback,
)
from transformers.trainer_utils import (
    EvalPrediction,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as PT_FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator

import os
import nni


def pt_fsdp_state_dict(model: torch.nn.Module):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with PT_FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()

class FSDPTrainer(Trainer):
    _optimizer = None
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        ignored_modules=[],
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        if hasattr(self.accelerator.state, 'fsdp_plugin') and self.accelerator.state.fsdp_plugin is not None:
            # Do not wrap rotation matrix
            for ignored_module in ignored_modules:
                ignored_module.to(torch.cuda.current_device())
            self.accelerator.state.fsdp_plugin.ignored_modules = ignored_modules
            # self.accelerator.state.fsdp_plugin.fsdp_version = 2
            # self.accelerator.state.fsdp_plugin.reshard_after_forward = True
            # use_orig_params because part of the model is freezed
            self.accelerator.state.fsdp_plugin.use_orig_params = True
        # handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        # handler_class_to_attr = {
        #     DistributedDataParallelKwargs: "ddp_handler",
        #     # GradScalerKwargs: "scaler_handler",
        #     # InitProcessGroupKwargs: "init_handler",
        #     # FP8RecipeKwargs: "fp8_recipe_handler",
        #     # AutocastKwargs: "autocast_handler",
        #     # ProfileKwargs: "profile_handler",
        #     # AORecipeKwargs: "ao_recipe_handler",
        #     # TERecipeKwargs: "te_recipe_handler",
        #     # MSAMPRecipeKwargs: "msamp_recipe_handler",
        # }
        # handler_attr = handler_class_to_attr[handler.__class__]
        # setattr(self, handler_attr, handler)
        _old_prepare = Accelerator.prepare
        def _new_prepare(self, *args, **kwargs):
            rets = _old_prepare(self, *args, **kwargs)
            # for ret in (rets if isinstance(rets, (tuple, list)) else [rets]):
            #     if isinstance(ret, nn.Module) and hasattr(ret, '_set_static_graph') and not ret.static_graph:
            #         ret._set_static_graph()
            return rets
        Accelerator.prepare = _new_prepare
            

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """

        # Overwrite optimizer creation because optimizer is already created
        self.optimizer = self._optimizer
        self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=self.optimizer,
        )

    def get_trained_params(self):
        """
        Returns a copy of the model on CPU.
        """
        if self.is_fsdp_enabled:
            state_dict = pt_fsdp_state_dict(self.model)
            return state_dict
        else:
            return self.model.state_dict()

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        teacher_model = kwargs.pop("teacher_model", None)
        super().__init__(*args, **kwargs)
        if (
            hasattr(self.accelerator.state, "fsdp_plugin")
            and self.accelerator.state.fsdp_plugin is not None
        ):
            model: nn.Module = self.model
            ignored_modules = list()
            for m in model.modules():
                if isinstance(m, (RotateModule, SmoothModule)):
                    ignored_modules.append(m)
                    m.to(torch.cuda.current_device())
            self.accelerator.state.fsdp_plugin.ignored_modules = ignored_modules
            self.accelerator.state.fsdp_plugin.use_orig_params = True

    def training_step(
        self, model: nn.Module, inputs, num_items_in_batch=None
    ):
        
        loss = super().training_step(model, inputs, num_items_in_batch)
        if int(os.environ['RANK']) == 0:
            nni.report_intermediate_result({"default": 1000.0, "loss": loss.item()})
        return loss

    @torch.compile(fullgraph=False)
    def compute_loss(self, model, inputs, **kwargs):
        args = self.args
        loss_type = args.special.get("loss_type", "origin")
        if loss_type == "origin":
            return super().compute_loss(model, inputs, **kwargs)

        if loss_type == "rkl":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.kl_div(
                F.log_softmax(ori_logits.flatten(0, -2), dim=-1),
                F.softmax(logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return loss
        if loss_type == "kl":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.kl_div(
                F.log_softmax(logits.flatten(0, -2), dim=-1),
                F.softmax(ori_logits, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            return loss

        if (
            "r_kl_top" in loss_type
        ):  
            labels = inputs.pop("labels", None)
            if loss_type == "k_top":
                k = 1000
            else:
                k = int(loss_type.split("_")[-1])
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            top_logits, indices = logits.topk(k, dim=-1, sorted=False)
            top_ori_logits = ori_logits.gather(-1, indices)
            loss = F.kl_div(
                F.log_softmax(top_ori_logits.flatten(0, -2), dim=-1),
                F.softmax(top_logits.flatten(0, -2), dim=-1),
                reduction="batchmean",
            )
            return loss

        if "kl_top" in loss_type:
            labels = inputs.pop("labels", None)
            if loss_type == "kl_top":
                k = 1000 
            else:
                k = int(loss_type.split("_")[-1])
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            top_ori_logits, indices = ori_logits.topk(k, dim=-1, sorted=False)
            if getattr(args, "post_attn", False):
                ref = F.softmax(ori_logits,dim=-1).gather(-1,indices).flatten(0,-2)
                can = F.log_softmax(logits,dim=-1).gather(-1,indices).flatten(0,-2)
                loss = F.kl_div(can,ref,reduction="batchmean")
            else:
                top_logits = logits.gather(-1, indices)
                loss = F.kl_div(
                    F.log_softmax(top_logits, dim=-1).flatten(0, -2),
                    F.softmax(top_ori_logits, dim=-1).flatten(0, -2),
                    reduction="batchmean",
                )
            return loss


        if loss_type == "mse":
            labels = inputs.pop("labels", None)
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.mse_loss(logits, ori_logits)
            return loss
        if loss_type == "kd":
            ori_logits = self.get_ori_outputs(model, inputs).logits
            outputs = model(**inputs)
            logits = outputs.logits
            T, alpha = self.temperature, self.loss_alpha
            ori_loss = outputs["loss"]
            logits = logits.view(-1, logits.size(-1))
            ori_logits = ori_logits.view(-1, ori_logits.size(-1))
            distill_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1).flatten(0, -2),
                F.softmax(ori_logits / T, dim=-1).flatten(0, -2),
                reduction="batchmean",
            )
            loss = ori_loss * (1 - alpha) + distill_loss * (alpha * T * T)
            return loss
        if loss_type == "DFT":
            shift_labels = inputs.pop("labels", None)[:, 1:].contiguous().flatten()
            outputs = model(**inputs)
            shift_logits = outputs.logits[:, :-1].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            loss = (loss * F.softmax(shift_logits, dim=-1).gather(1, shift_labels.unsqueeze(-1)).squeeze(-1).detach()).mean()
            return loss

    @torch.no_grad()
    def get_ori_outputs(self, model, inputs):
        inputs = dict(inputs)
        inputs.pop("labels", None)

        outputs = model.teacher(**inputs, output_hidden_states=True)
        model.teacher._is_root = False
        return outputs

    def create_optimizer_and_scheduler(self, num_training_steps: int):

        args = self.args
        params_rotate = []
        params_smooth = []
        for param in self.model.parameters():
            param: torch.nn.Parameter
            if param.requires_grad:
                if len(param.size()) == 2:
                    params_rotate.append(param)
                else:
                    params_smooth.append(param)
        dict_rotate = {
            "params": params_rotate,
            "lr": args.special.get("rotate_lr", 0.1),
            "momentum": args.special.get("rotate_momentum", 0.0),
            "stiefel": True,
            "grassmann": True,
            "omega": 0.1,
        }
        dict_smooth = {
            "params": params_smooth,
            "lr": args.special.get("smooth_lr", 0.0),
            "momentum": args.special.get("smooth_momentum", 0.0),
            "stiefel": False,
            "nesterov": False,
        }
        if args.special.get("opt_type", "SGDG") == "SGDG":
            optimizer = SGDG(
                [dict_rotate, dict_smooth], weight_decay=0
            )  
        elif args.special.get("opt_type", "SGDG") == "RSGD":
            import geoopt
            optimizer = geoopt.optim.RiemannianSGD(
                [dict_rotate, dict_smooth], weight_decay=0, lr=args.special.get("rotate_lr", 0.1),stabilize=10,
            )
        elif args.special.get("opt_type", "SGDG") == "RAdam":
            import geoopt
            optimizer = geoopt.optim.RiemannianAdam(
                [dict_rotate, dict_smooth], weight_decay=0, lr=args.special.get("rotate_lr", 0.1),stabilize=10
            )
        self.optimizer = optimizer
        
        self.create_scheduler(
            num_training_steps=num_training_steps,
            optimizer=optimizer,
        )

    def get_trained_params(self):
        """
        Returns a copy of the model on CPU.
        """
        if self.is_fsdp_enabled:
            state_dict = pt_fsdp_state_dict(self.model)
            return state_dict
        else:
            return self.model.state_dict()