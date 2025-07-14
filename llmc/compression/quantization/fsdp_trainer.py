from typing import Callable, Dict, List, Optional, Tuple, Union

# isort: on
import numpy as np
import torch
from packaging import version
from torch import nn

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