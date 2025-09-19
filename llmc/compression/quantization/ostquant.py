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
from .spinquant import SpinQuant
from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix
from .module_utils import *
from .module_utils import (_LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_,
                           EffcientFakeQuantLinear, FakeQuantLinear, RotateFakeQuantLinear,
                           LlmcScaleRMSNorm, OriginEmbedding, OriginFloatLinear,
                           OriginFloatConv3d,
                           RotateEmbedding, RotateLinear2, _ROTATE_LINEAR_MAP_,
                           _REALQUANT_LINEAR_MAP_, get_module_name)
from .rotate_utils import ActRotater, RotateModule, WeightRotaterSmooth, SmoothModule


@ALGO_REGISTRY
class OSTQuant(SpinQuant):
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
        self.had_dim = self.hidden_size // self.num_heads
        special_config = self.quant_config.get('special', {})
        self.selected_layers = special_config.get('selected_layers', None)

    def add_quant_config(self):
        self.rotate_mode = self.quant_config['special']['rotate_mode']
        self.weight_rotate = True
        self.w_rotater = WeightRotaterSmooth(weight_rotate_func=self.rotate_weight, dev=self.dev)
        # self.o_proj_group_quant = self.quant_config['special']['o_proj_group_quant']


    # def register_lmhead_spin_parameters(self):
    #     pre_head_ln = self.model.get_pre_head_layernorm_layers()[0]
    #     pre_head_ln_name = get_module_name(self.model.model, pre_head_ln)
    #     S_head = SmoothModule(torch.ones(pre_head_ln.weight.shape[0],dtype=torch.float32,device=self.dev))
    #     self.model.modality_model.S_head = S_head
    #     args = {'Sout': S_head}
    #     params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
    #     self.model.replace_module_subset(
    #         LlmcScaleRMSNorm,
    #         self.model.model,
    #         {'layers': {pre_head_ln_name: pre_head_ln}},
    #         None,
    #         params_dict,
    #     )
    #     args = {'Sin': S_head, 'Q1': self.model.modality_model.Q1}
    #     params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
    #     head_layers = {get_module_name(self.model.model, h): h for h in self.model.get_head_layers()}
    #     self.model.replace_module_subset(
    #         RotateLinear2,
    #         self.model.model,
    #         {'layers': head_layers},
    #         None,
    #         params_dict,
    #     )

    def w_rot(self, module, w_rotater, args):
        return w_rotater.rotate(module.weight, module.bias, args.get('Q1'), args.get('Q2'), args.get('transpose'), args.get('Sin'), args.get('Sout'), args.get('inverse_out'), self.had_dim, args.get('is_qk',False))

    def block_transform(self, block):
        logger.info(f'Start transform the {self.block_idx+1}-th block')
        block.S_up_down = nn.ModuleList()
        subsets = self.model.get_subsets_in_block(block)
        self._s_up_down_cnt = 0
        for index, subset in enumerate(subsets):
            self.subset_transform(block, subset)
        
        self.set_non_linear_mode('fake_quant', block, False)

        # self.model.replace_module_block(LlmcRMSNorm, block, self.block_idx, {})

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
            if 'is_mlp' not in subset or not subset['is_mlp']:
                if self.rotate_mode == 'klt':
                    Q2 = self.get_orthogonal_matrix(self.hidden_size // self.num_heads, block)
                else:
                    Q2 = torch.stack([self.get_orthogonal_matrix(self.hidden_size // self.num_heads) for _ in range(self.num_key_value_heads)], dim=0)
                block.Q2 = RotateModule(Q2)
                S_norm_qkv = SmoothModule(torch.ones(prev_op[0].weight.shape[0],dtype=torch.float32,device=self.dev))
                S_qk = SmoothModule(torch.ones(layers_dict['self_attn.k_proj'].weight.shape[0]//2,dtype=torch.float32,device=self.dev)) # TODO //2 for 等价
                S_ov = SmoothModule(torch.ones(layers_dict['self_attn.v_proj'].weight.shape[0],dtype=torch.float32,device=self.dev))
                block.S_norm_qkv = S_norm_qkv
                block.S_qk = S_qk
                block.S_ov = S_ov
                # for n in layers_dict.keys():
                n = 'self_attn.q_proj'
                m = layers_dict[n]
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=False, Sin=block.S_norm_qkv, Sout=block.S_qk, inverse_out=False, is_qk=True)
                n = 'self_attn.k_proj'
                m = layers_dict[n]
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=False, Sin=block.S_norm_qkv, Sout=block.S_qk, inverse_out=True, is_qk=True)
                n = 'self_attn.v_proj'
                m = layers_dict[n]
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=block.Q2, transpose=False, Sin=block.S_norm_qkv, Sout=block.S_ov, inverse_out=False)

                args = {'Sout': S_norm_qkv}
                params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
                norm_name = get_module_name(block, prev_op[0])
                self.model.replace_module_subset(
                    LlmcScaleRMSNorm,
                    block,
                    {'layers': {norm_name: prev_op[0]}},
                    self.block_idx,
                    params_dict,
                )
                if self.selected_layers:
                    if "S_norm_qkv" not in self.selected_layers:
                        block.S_norm_qkv.weight.requires_grad = False
                    if "S_qk" not in self.selected_layers:
                        block.S_qk.weight.requires_grad = False
                    if "S_ov" not in self.selected_layers:
                        block.S_ov.weight.requires_grad = False

            else:
                S_norm_upgate = SmoothModule(torch.ones(prev_op[0].weight.shape[0],dtype=torch.float32,device=self.dev))
                
                block.S_norm_upgate = S_norm_upgate
                for n in layers_dict.keys():
                # n = 'mlp.up_proj'
                    m = layers_dict[n]
                    if not n.endswith("up_proj"):
                        self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=False, Sin=block.S_norm_upgate, inverse_out=False)
                    else:
                        S_up_down = SmoothModule(torch.ones(m.weight.shape[0],dtype=torch.float32,device=self.dev))
                        self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=False, Sin=block.S_norm_upgate, Sout=S_up_down, inverse_out=False)
                        block.S_up_down.append(S_up_down)
                        if self.selected_layers and "S_up_down" not in self.selected_layers:
                            S_up_down.weight.requires_grad = False
                            
                

                args = {'Sout': S_norm_upgate}
                params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=None, args=args)
                norm_name = get_module_name(block, prev_op[0])
                self.model.replace_module_subset(
                    LlmcScaleRMSNorm,
                    block,
                    {'layers': {norm_name: prev_op[0]}},
                    self.block_idx,
                    params_dict,
                )
                if self.selected_layers:
                    if "S_norm_upgate" not in self.selected_layers:
                        block.S_norm_upgate.weight.requires_grad = False
                    # if "S_up_down" not in self.selected_layers:
                    #     block.S_up_down.weight.requires_grad = False

        else:
            if self.config['model']['type'] in ['Opt']:
                self.bake_mean_into_linear(layers[0])
            
            for i, n in enumerate(layers_dict.keys()):
                m = layers_dict[n]
                if 'is_mlp' in subset and subset['is_mlp']:
                    # up_m = prev_op[i]
                    if self.online_rotate:
                        apply_exact_had_to_linear(m, had_dim=-1, output=False)
                    self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=True, Sin=block.S_up_down[self._s_up_down_cnt], inverse_out=False)
                    self._s_up_down_cnt += 1
                    
                else:
                    self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=block.Q2, transpose=True,Sin=block.S_ov)
                    # self.replace_rotate_sm_fc(block, f'{self._atten_inspect_name}.v_proj', prev_op[0], Q1=self.model.modality_model.Q1, Q2=self._get_block_Q2(block), transpose=False)
    
    def replace_rotate_sm_fc(self, block, n, m, Q1=None, Q2=None, transpose=False, Sin=None, Sout=None, inverse_out=False, is_qk=False):
        args = {}
        if hasattr(self, 'weight_rotate') and self.weight_rotate:
            args['Q1'] = Q1
            args['Q2'] = Q2
            args['transpose'] = transpose
            args['Sin'] = Sin
            args['Sout'] = Sout
            args['inverse_out'] = inverse_out
            args['is_qk'] = is_qk

        params_dict = self.get_replacement_params(mode='rotate', w_only=self.w_only, name=n, args=args)
        if params_dict == {}:
            return

        subset = {'layers': {n: m}}
        self.model.replace_module_subset(
            RotateLinear2,
            block,
            subset,
            self.block_idx,
            params_dict
        )

    def apply_lmhead_rotate_weight(self):
        pre_head_ln = self.model.get_pre_head_layernorm_layers()[0]
        if isinstance(pre_head_ln, LlmcScaleRMSNorm):
            pre_head_ln.cuda()
            weight, bias = pre_head_ln._rotate_weight()
            pre_head_ln.weight.data = weight.data
            if bias is not None:
                pre_head_ln.bias.data = bias.data
            pre_head_ln_name = get_module_name(self.model.model, pre_head_ln)
            pre_head_ln.cpu()
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

            
    def deploy(self, quant_format, keep_device=False):
        super().deploy(quant_format, keep_device=keep_device)
        if quant_format == 'origin_float':
            self.set_non_linear_mode('fake_quant', self.model.model, True)
            self.model.replace_module_all(OriginLlmcRMSNorm, {})
            
            # pre_head_ln = self.model.get_pre_head_layernorm_layers()[0]
            # pre_head_ln_name = get_module_name(self.model.model, pre_head_ln)

            # self.model.replace_module_subset(
            #     OriginLlmcRMSNorm,
            #     self.model.model,
            #     {'layers': {pre_head_ln_name: pre_head_ln}},
            #     None,
            #     {},
            # )