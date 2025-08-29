import copy
import gc
import os
from functools import partial
import json

import torch
import torch.nn as nn
from loguru import logger
from transformers import (default_data_collator, AutoTokenizer)

from llmc.utils.registry_factory import ALGO_REGISTRY
from llmc.data import MixDataset, BaseTokenizer, TrainJsonDataset
from llmc.utils.registry_factory import MODEL_REGISTRY

from .train_utils.fsdp_trainer import MyTrainer
from .train_utils.train_utils import LLMCTrainingArguments
from .base_blockwise_quantization import BaseBlockwiseQuantization
from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix
from .module_utils import *
from .module_utils import (_LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_,
                           EffcientFakeQuantLinear, FakeQuantLinear, RotateFakeQuantLinear,
                           LlmcRMSNorm, OriginEmbedding, OriginFloatLinear,
                           OriginFloatConv3d,
                           RotateEmbedding, RotateLinear2, _ROTATE_LINEAR_MAP_,
                           _REALQUANT_LINEAR_MAP_, get_module_name)
from .rotate_utils import ActRotater, RotateModule, WeightRotater


@ALGO_REGISTRY
class SpinQuant(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input,  padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.dev = torch.device('cuda')
        self.add_quant_config()
        self._atten_inspect_name = 'self_attn'
        if self.modality == 'vision':
            self.vision_preprocess()
        elif self.modality == 'language':
            self.preprocess()
        else:
            raise ValueError(f'Unsupported modality {self.modality}')

        self.avaliable_train_state = ["train_rotate_quant"]

    def vision_preprocess(self):
        for m in self.model.modality_model.parameters():
            m.requires_grad = False
        Q1 = self.get_orthogonal_matrix(self.hidden_size)
        self.model.modality_model.Q1 = RotateModule(Q1)
        # Rotate the vision projector
        layers_dict = {}
        args = {}
        args['Q1'] = self.model.modality_model.Q1
        args['Q2'] = None
        args['transpose'] = False
        params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
        vision_projector = self.model.vision_projector
        if vision_projector is not None:
            logger.info('Rotating vision projector layer.')
            pre_vison_proj_ln = vision_projector.ln_q
            self.fuse_ln_fcs(pre_vison_proj_ln, [vision_projector.mlp[0]])
            pre_vison_proj_ln_name = get_module_name(vision_projector, pre_vison_proj_ln)
            self.model.replace_module_subset(
                LlmcRMSNorm,
                self.model.vision_projector,
                {'layers': {pre_vison_proj_ln_name: pre_vison_proj_ln}},
                None,
                {},
            )
            vision_up_proj = [self.model.vision_projector.mlp[0]]
            for layer in vision_up_proj:
                rot_layer_name = get_module_name(self.model.model, layer)
                layers_dict[rot_layer_name] = layer
        if layers_dict:
            self.model.replace_module_subset(
                RotateLinear2,
                self.model.model,
                {'layers': layers_dict},
                None,
                params_dict
            )

        # Rotate the vision embed layers

        args = {}
        args['Q1'] = self.model.modality_model.Q1
        args['Q2'] = None
        args['transpose'] = True
        params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
        vision_embed = [self.model.vision_embed.proj]
        if vision_embed is not None:
            logger.info('Rotating vision head layers.')
            for layer in vision_embed:
                rot_layer_name = get_module_name(self.model.model, layer)

                self.model.replace_module_subset(
                    _ROTATE_LINEAR_MAP_[type(layer)],
                    self.model.model,
                    {'layers': {rot_layer_name: layer}},
                    None,
                    params_dict
                )

    def add_quant_config(self):
        self.rotate_mode = self.quant_config['special']['rotate_mode']
        self.weight_rotate = True
        self.w_rotater = WeightRotater(weight_rotate_func=self.rotate_weight, dev=self.dev)
        # self.o_proj_group_quant = self.quant_config['special']['o_proj_group_quant']

    def preprocess(self):
        for m in self.model.modality_model.parameters():
            m.requires_grad = False

        if not self.config['model']['type'].startswith('Qwen'):
            self.remove_mean_from_embed()

        Q1 = self.get_orthogonal_matrix(self.hidden_size)
        self.model.modality_model.Q1 = RotateModule(Q1)

        self.register_embed_spin_parameters()

        pre_head_ln = self.model.get_pre_head_layernorm_layers()[0]
        self.fuse_ln_fcs(pre_head_ln, self.model.get_head_layers())

        pre_head_ln_name = get_module_name(self.model.model, pre_head_ln)
        self.model.replace_module_subset(
            LlmcRMSNorm,
            self.model.model,
            {'layers': {pre_head_ln_name: pre_head_ln}},
            None,
            {},
        )
        self.register_lmhead_spin_parameters()

        gc.collect()
        torch.cuda.empty_cache()

    def get_trainable_params(self, model=None):
        trainable_parameters = []
        if model is None:
            model = self.model
        for n, m in model.model.named_parameters():
            if 'Q1' in n or 'Q2' in n:
                trainable_parameters.append(m)
        return trainable_parameters

    def a_rot(self, act, module, a_rotater):
        return a_rotater.rotate(act)

    def w_rot(self, module, w_rotater, args):
        return w_rotater.rotate(module.weight, module.bias, args['Q1'], args['Q2'], args['transpose'])

    def w_qdq_tmp(self, module, weight, wquantizer):
        args = {'lowbound_factor': None, 'upbound_factor': None}
        if hasattr(module, 'buf_lowbound_factor'):
            args['lowbound_factor'] = module.buf_lowbound_factor
        if hasattr(module, 'buf_upbound_factor'):
            args['upbound_factor'] = module.buf_upbound_factor

        return wquantizer.fake_quant_weight_dynamic(weight, args)

    def register_embed_spin_parameters(self):
        embedding_layer = self.model.get_embed_layers()[0]
        args = {}
        args['Q1'] = self.model.modality_model.Q1
        args['Q2'] = None
        args['transpose'] = False
        params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
        params_dict.pop('a_rot')
        embedding_layer_name = get_module_name(self.model.model, embedding_layer)
        self.model.replace_module_subset(
            RotateEmbedding,
            self.model.model,
            {'layers': {embedding_layer_name: embedding_layer}},
            None,
            params_dict
        )
        self.model.find_embed_layers()
        layers_dict = {}
        args = {}
        args['Q1'] = self.model.modality_model.Q1
        args['Q2'] = None
        args['transpose'] = True
        params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
        for rot_layer in self.model.get_extra_rot_module_besides_embed_layers():
            logger.info('For multimodal model, quarot need rotate last layer in projector.')
            logger.info(f'rot_layer : {rot_layer}')
            # docformatter: off
            """
            txt_input     img_input
                |             |
            Embedding      vision_projector
                |             |
                       |
                  input_embeds
                       |
                       Y
            Therefore:
            X_txt ~ W_embedding * Q = X_txt ~ (W_embedding * Q)
            X_proj * W_proj.t() * Q = X_proj * (Q.t() * W_proj).t()
            """
            module_name = get_module_name(self.model.model, rot_layer)
            if module_name is None:
                raise ValueError(f'Cannot find module name for {rot_layer}. Please check the model structure.')
            logger.info(f'Replacing module {module_name} with RotateLinear2')
            layers_dict[module_name] = rot_layer
        if layers_dict:
            self.model.replace_module_subset(
                RotateLinear2,
                self.model.model,
                {'layers': layers_dict},
                None,
                params_dict
            )
        self._vision_rotate_layers = layers_dict

    def register_lmhead_spin_parameters(self):
        lm_head_layer = self.model.get_head_layers()[0]
        args = {}
        args['Q1'] = self.model.modality_model.Q1
        args['Q2'] = None
        args['transpose'] = False
        params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
        lm_head_layer_name = get_module_name(self.model.model, lm_head_layer)
        self.model.replace_module_subset(
            RotateLinear2,
            self.model.model,
            {'layers': {lm_head_layer_name: lm_head_layer}},
            None,
            params_dict
        )

    def apply_fc_rotate_weight(self):
        for idx, block in enumerate(self.blocks):
            block.cuda()
            logger.info(f'Start apply {idx}-th block rotate weights')
            for name, module in block.named_modules():
                if isinstance(module, (RotateLinear2, FakeQuantLinear, RotateFakeQuantLinear, LlmcScaleRMSNorm)):
                    weight, bias = module._rotate_weight()
                    module.weight.data = weight.data
                    if bias is not None:
                        module.bias.data = bias.data
            block.cpu()
            logger.info(f'End apply {idx}-th block rotate weights')

    def apply_embedding_rotate_weight(self):
        self.model.find_embed_layers()
        embedding_layer = self.model.get_embed_layers()[0]
        if isinstance(embedding_layer, RotateEmbedding):
            embedding_layer.cuda()
            weight = embedding_layer._rotate_weight()
            embedding_layer.weight.data = weight.data
            embedding_layer_name = get_module_name(self.model.model, embedding_layer)
            self.model.replace_module_subset(
                OriginEmbedding,
                self.model.model,
                {'layers': {embedding_layer_name: embedding_layer}},
                None,
                {}
            )
            embedding_layer.cpu()

    def apply_lmhead_rotate_weight(self):
        lm_head_layer = self.model.get_head_layers()[0]
        if isinstance(lm_head_layer, RotateLinear2):
            lm_head_layer.cuda()
            weight, bias = lm_head_layer._rotate_weight()
            lm_head_layer.weight.data = weight.data
            if bias is not None:
                lm_head_layer.bias.data = bias.data
            # lm_head_layer.weight, lm_head_layer.bias = weight, bias
            lm_head_layer_name = get_module_name(self.model.model, lm_head_layer)
            self.model.replace_module_subset(
                OriginFloatLinear,
                self.model.model,
                {'layers': {lm_head_layer_name: lm_head_layer}},
                None,
                {}
            )
            lm_head_layer.cpu()

    def apply_vision_rotate_weight(self):
        vision_up_proj = self.model.vision_projector.mlp[0]
        if isinstance(vision_up_proj, RotateLinear2):
            vision_up_proj.cuda()
            weight, bias = vision_up_proj._rotate_weight()
            vision_up_proj.weight.data = weight.data
            if bias is not None:
                vision_up_proj.bias.data = bias.data
            vision_up_proj_name = get_module_name(self.model.model, vision_up_proj)
            self.model.replace_module_subset(
                OriginFloatLinear,
                self.model.model,
                {'layers': {vision_up_proj_name: vision_up_proj}},
                None,
                {}
            )
            vision_up_proj.cpu()
        vision_embed = self.model.vision_embed.proj
        if isinstance(vision_embed, RotateConv3d):
            vision_embed.cuda()
            weight, bias = vision_embed._rotate_weight()
            vision_embed.weight.data = weight.data
            if bias is not None:
                vision_embed.bias.data = bias.data
            vision_embed_name = get_module_name(self.model.model, vision_embed)
            self.model.replace_module_subset(
                OriginFloatConv3d,
                self.model.model,
                {'layers': {vision_embed_name: vision_embed}},
                None,
                {}
            )
            vision_embed.cpu()

    def get_orthogonal_matrix(self, size):
        if self.rotate_mode == 'random':
            return random_orthogonal_matrix(size, self.dev)
        elif self.rotate_mode == 'hadamard':
            return random_hadamard_matrix(size, self.dev)
        else:
            raise ValueError(f'Unsupported mode {self.mode}')

    def block_transform(self, block):
        logger.info(f'Start transform the {self.block_idx+1}-th block')

        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            self.subset_transform(block, subset)

        self.set_non_linear_mode('fake_quant', block, False)
        self.model.replace_module_block(LlmcRMSNorm, block, self.block_idx, {})

        logger.info(f'block:{block}')
        logger.info(f'End transform the {self.block_idx+1}-th block')

    def subset_transform(self, block, subset):
        prev_op = subset['prev_op']
        layers_dict = subset['layers']
        assert (
            len(prev_op) == 1
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'

        layers = list(layers_dict.values())
        if self.modality == 'vision':
            raise NotImplementedError('SpinQuant does not support vision modality yet. Becasue(qkv concat Linear)')

        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.fuse_ln_fcs(prev_op[0], layers)
            for n in layers_dict.keys():
                m = layers_dict[n]
                self.replace_rotate_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=False)
            if 'is_mlp' not in subset or not subset['is_mlp']:
                Q2 = self.get_orthogonal_matrix(self.hidden_size // self.num_heads)
                subset['inspect'].Q2 = RotateModule(Q2)

        else:
            if self.config['model']['type'] in ['Opt']:
                self.bake_mean_into_linear(layers[0])

            n = list(layers_dict.keys())[0]
            m = layers[0]
            if 'is_mlp' in subset and subset['is_mlp']:
                if self.online_rotate:
                    apply_exact_had_to_linear(m, had_dim=-1, output=False)
                self.replace_rotate_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=True)
            else:
                self.replace_rotate_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=self._get_block_Q2(block), transpose=True)
                self.replace_rotate_fc(block, f'{self._atten_inspect_name}.v_proj', prev_op[0], Q1=self.model.modality_model.Q1, Q2=self._get_block_Q2(block), transpose=False)


    def _get_block_Q2(self, block):
        if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'Q2'):
            self._atten_inspect_name = 'self_attn'
            return block.self_attn.Q2
        elif hasattr(block, 'attn') and hasattr(block.attn, 'Q2'):
            self._atten_inspect_name = 'attn'
            return block.attn.Q2
        return None

    def get_ignored_modules(self, model=None):
        if model is None:
            model = self.model
        ignored_modules = []
        for n, m in model.model.named_modules():
            if n.endswith('Q1') or n.endswith('Q2'):
                ignored_modules.append(m)
        return ignored_modules

    def apply_rotate_weight(self):
        if self.modality == 'vision':
            self.apply_vision_rotate_weight()
        else:
            self.apply_embedding_rotate_weight()
            self.apply_lmhead_rotate_weight()
            self.apply_fc_rotate_weight()
        if hasattr(self, "_vision_rotate_layers"):
            for module_name in self._vision_rotate_layers:
                module = self.model.model.get_submodule(module_name)
                if isinstance(module, (RotateLinear2, RotateFakeQuantLinear)):
                    weight, bias = module._rotate_weight()
                    module.weight.data = weight.data
                    if bias is not None:
                        module.bias.data = bias.data
                    # module.weight, module.bias = weight, bias
                self.model.replace_module_subset(
                    OriginFloatLinear,
                    self.model.model,
                    {'layers': {module_name: module}},
                    None,
                    {}
                )
                del self._vision_rotate_layers


    def deploy(self, quant_format, keep_device=False):
        if quant_format == 'train_rotate_quant':
            logger.info(f'-- deploy_{quant_format}_model start --')
            logger.info(f'quant_config : {self.quant_config}')
            logger.info(self.model.model)

            params_dict = {}
            params_dict['w_qdq'] = partial(self.w_qdq_tmp, wquantizer=self.wquantizer)
            params_dict['a_qdq'] = (
                partial(self.a_qdq, aquantizer=self.aquantizer)
                if not self.w_only
                else None
            )
            if self.modality == 'vision':
                self.model.replace_vision_module_all(
                    RotateFakeQuantLinear, params_dict
                )
            else:
                self.model.replace_module_all(
                    RotateFakeQuantLinear, params_dict
                )

            logger.info(f'-- deploy_{quant_format}_model done --')
            logger.info(f'-- strat train rotation--')
        else:
            with torch.no_grad():
                if quant_format == 'origin_float' and 'save' in self.config and self.config.save.get('save_rotate_weight', False):
                    # save rotate weight state_dict
                    state_dict = {}
                    for name, param in self.model.model.state_dict().items():
                        if 'Q1' in name or 'Q2' in name:
                            state_dict[name] = param
                    torch.save(state_dict, os.path.join(self.config.save.save_path,'rotate_weight.pth'))
                self.apply_rotate_weight()
                super().deploy(quant_format)
        if quant_format == 'origin_float':
            self.set_non_linear_mode('fake_quant', self.model.model, True)

    @torch.no_grad()
    def save_model(self, path):
        super().save_model(path)
        path = os.path.join(path, 'config.json')
        with open(path, 'r') as f:
            config = json.load(f)
        if 'tie_word_embeddings' in config:
            config['tie_word_embeddings'] = False
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    def train(self, tokenizer=None):
        # ignored_modules = []
        llmc_model_to_train = copy.deepcopy(self.model)
        llmc_model_to_train.config.calib = self.config.train.data
        # ignored_modules.extend(self.get_ignored_modules(llmc_model_to_train))
        llmc_model_to_train.model.config.use_cache = False

        dataset = MixDataset(tokenizer.get_tokenizer(), self.config.train.data, llmc_model_to_train.batch_process, llmc_model_to_train.processor)
        model_max_length=self.config.train.data[0].seq_len if isinstance(self.config.train.data, list) else self.config.train.data.seq_len
        train_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model.path,
            cache_dir=getattr(self.config.train.data, "cache_dir", None),
            model_max_length=model_max_length,
            padding_side='right',
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
        )
        llmc_model_to_train.processor.tokenizer = train_tokenizer


        # if 'eval' in config and len(config.eval.eval_pos):
        #     eval_list = []
        #     name_list = (
        #         config.eval.name
        #         if not isinstance(config.eval.name, str)
        #         else [config.eval.name]
        #     )
        #     for name in name_list:
        #         eval_config = copy.deepcopy(config.eval)
        #         eval_config.name = name
        #         if len(name_list) != 1:  # eval multi datasets
        #             eval_config.path = os.path.join(config.eval.path, name)
        #         ppl_eval = PerplexityEval(self.model, eval_config)
        #         eval_list.append(ppl_eval)
        train_data = TrainJsonDataset(
            dataset.get_raw_calib_dataset(),
            train_tokenizer,
            block_size=model_max_length,
        )

        train_args = LLMCTrainingArguments(**self.config.train.train_args)
        # trainable_parameters = self.get_trainable_params(llmc_model_to_train)
        llmc_model_to_train.model.seqlen = model_max_length
        # optimizer = SGDG(trainable_parameters, lr=self.config.train.train_args.learning_rate, stiefel=True)
        # FSDPTrainer._optimizer = optimizer
        need_teacher = train_args.special.get("loss_type", 'origin') not in  ("origin", "DFT")
        if need_teacher:
            _backup = self.config.model.path
            if train_args.special.get("teacher_path"):
                self.config.model.path = train_args.special.get("teacher_path")
            teacher_model = MODEL_REGISTRY[self.config.model.type](self.config).model
            self.config.model.path = _backup
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.config.use_cache = False
            llmc_model_to_train.model.teacher = TeacherModel(teacher_model)
        # from trl.trainer.utils import DataCollatorForCompletionOnlyLM
        # from accelerate.utils import operations
        from llmc.utils import patch
        from types import MethodType
        # _concatenate = operations.concatenate
        # operations.concatenate = patch.concatenate
        train_tokenizer.pad = MethodType(patch.pad, train_tokenizer)
        trainer = MyTrainer(
            model=llmc_model_to_train.model,
            tokenizer=train_tokenizer,
            args=train_args,
            train_dataset=train_data,
            eval_dataset=None,
            # data_collator=default_data_collator,
            data_collator=patch.CustomDataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=train_tokenizer, pad_to_multiple_of=8),
            # optimizers=(optimizer, None),
            # optimizers=(None, None),
            # ignored_modules=ignored_modules,
        )
        
        torch.distributed.barrier()

        res = trainer.train()
        # operations.concatenate = _concatenate
        # if int(os.environ['RANK']) == 0:
        #     import nni
        #     nni.report_final_result(res.training_loss)
        
        torch.distributed.barrier()

        logger.info('End training')
        if need_teacher:
            del llmc_model_to_train.model.teacher, teacher_model
        

        # self.model.model.to('cpu')
        # self.model.model.load_state_dict(trainer.get_trained_params(), device_map='auto')
        state_dict = trainer.get_trained_params()
        model_state = self.model.model.state_dict()
        for name, param in model_state.items():
            if name in state_dict:
                # 保持原 device，只拷贝数据
                param.copy_(state_dict[name].to(param.device, dtype=param.dtype))