import gc
import os
from functools import partial
import json

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix
from .module_utils import *
from .module_utils import (_LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_,
                           EffcientFakeQuantLinear, FakeQuantLinear, RotateFakeQuantLinear,
                           LlmcRMSNorm, OriginEmbedding, OriginFloatLinear,
                           RotateEmbedding, RotateLinear2,_REALQUANT_LINEAR_MAP_, get_module_name)
from .rotate_utils import ActRotater, RotateModule, WeightRotater


@ALGO_REGISTRY
class SpinQuant(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input,  padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.dev = torch.device('cuda')
        self.add_quant_config()
        self.preprocess()

    def add_quant_config(self):
        self.rotate_mode = self.quant_config['special']['rotate_mode']
        self.weight_rotate = True
        self.w_rotater = WeightRotater(weight_rotate_func=self.rotate_weight, dev=self.dev)
        # self.o_proj_group_quant = self.quant_config['special']['o_proj_group_quant']

    def preprocess(self):
        for m in self.model.model.parameters():
            m.requires_grad = False

        if self.config['model']['type'] not in ['Qwen25VL']:
            self.remove_mean_from_embed()

        Q1 = self.get_orthogonal_matrix(self.hidden_size)
        self.model.model.Q1 = RotateModule(Q1)

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

    def w_qdq_tmp(self, module, wquantizer):
        args = {'lowbound_factor': None, 'upbound_factor': None}
        if hasattr(module, 'buf_lowbound_factor'):
            args['lowbound_factor'] = module.buf_lowbound_factor
        if hasattr(module, 'buf_upbound_factor'):
            args['upbound_factor'] = module.buf_upbound_factor

        return wquantizer.fake_quant_weight_dynamic(module.tmp_weight, args)

    def register_embed_spin_parameters(self):
        embedding_layer = self.model.get_embed_layers()[0]
        args = {}
        args['Q1'] = self.model.model.Q1
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
        args['Q1'] = self.model.model.Q1
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
        args['Q1'] = self.model.model.Q1
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
                if isinstance(module, (RotateLinear2, FakeQuantLinear, RotateFakeQuantLinear)):
                    weight, bias = module._rotate_weight()
                    module.weight, module.bias = weight, bias
            block.cpu()
            logger.info(f'End apply {idx}-th block rotate weights')

    def apply_embedding_rotate_weight(self):
        self.model.find_embed_layers()
        embedding_layer = self.model.get_embed_layers()[0]
        if isinstance(embedding_layer, RotateEmbedding):
            embedding_layer.cuda()
            weight = embedding_layer._rotate_weight()
            embedding_layer.weight.data = weight
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
            lm_head_layer.weight, lm_head_layer.bias = weight, bias
            lm_head_layer_name = get_module_name(self.model.model, lm_head_layer)
            self.model.replace_module_subset(
                OriginFloatLinear,
                self.model.model,
                {'layers': {lm_head_layer_name: lm_head_layer}},
                None,
                {}
            )
            lm_head_layer.cpu()


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

        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.fuse_ln_fcs(prev_op[0], layers)
            for n in layers_dict.keys():
                m = layers_dict[n]
                self.replace_rotate_fc(block, n, m, Q1=self.model.model.Q1, Q2=None, transpose=False)
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
                self.replace_rotate_fc(block, n, m, Q1=self.model.model.Q1, Q2=None, transpose=True)
            else:
                self.replace_rotate_fc(block, n, m, Q1=self.model.model.Q1, Q2=block.self_attn.Q2, transpose=True)
                self.replace_rotate_fc(block, 'self_attn.v_proj', prev_op[0], Q1=self.model.model.Q1, Q2=block.self_attn.Q2, transpose=False)

    def get_ignored_modules(self, model=None):
        if model is None:
            model = self.model
        return [model.model.Q1] + [
            block.self_attn.Q2 for block in model.get_blocks()
        ]

    def apply_rotate_weight(self):
        self.apply_embedding_rotate_weight()
        self.apply_lmhead_rotate_weight()
        self.apply_fc_rotate_weight()
        if hasattr(self, "_vision_rotate_layers"):
            for module_name in self._vision_rotate_layers:
                module = self.model.model.get_submodule(module_name)
                if isinstance(module, (RotateLinear2, RotateFakeQuantLinear)):
                    weight, bias = module._rotate_weight()
                    module.weight, module.bias = weight, bias
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
            self.model.replace_module_all(
                RotateFakeQuantLinear, params_dict
            )

            logger.info(f'-- deploy_{quant_format}_model done --')
            logger.info(f'-- strat train rotation--')
        else:
            with torch.no_grad():
                self.apply_rotate_weight()
                super().deploy(quant_format)

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