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
                           LlmcRMSNorm, OriginEmbedding, OriginFloatLinear,
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

    def add_quant_config(self):
        self.rotate_mode = self.quant_config['special']['rotate_mode']
        self.weight_rotate = True
        self.w_rotater = WeightRotaterSmooth(weight_rotate_func=self.rotate_weight, dev=self.dev)
        # self.o_proj_group_quant = self.quant_config['special']['o_proj_group_quant']


    def w_rot(self, module, w_rotater, args):
        return w_rotater.rotate(module.weight, module.bias, args['Q1'], args['Q2'], args['transpose'], args.get('Sin'), args.get('Sout'), args.get('inverse_out'), self.had_dim)


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
        if self.modality == 'vision':
            raise NotImplementedError('SpinQuant does not support vision modality yet. Becasue(qkv concat Linear)')

        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.fuse_ln_fcs(prev_op[0], layers)
            if 'is_mlp' not in subset or not subset['is_mlp']:
                Q2 = self.get_orthogonal_matrix(self.hidden_size // self.num_heads)
                subset['inspect'].Q2 = RotateModule(Q2)
                # S_norm_qkv = SmoothModule(torch.ones(prev_op[0].weight.shape[-1],dtype=torch.float32,device=self.dev)).weight
                S_qk = SmoothModule(torch.ones(layers_dict['self_attn.k_proj'].weight.shape[0],dtype=torch.float32,device=self.dev))
                S_ov = SmoothModule(torch.ones(layers_dict['self_attn.v_proj'].weight.shape[0],dtype=torch.float32,device=self.dev))
                block.S_norm_qkv = None # S_norm_qkv
                block.S_qk = S_qk
                block.S_ov = S_ov
                # for n in layers_dict.keys():
                n = 'self_attn.q_proj'
                m = layers_dict[n]
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=False, Sin=block.S_norm_qkv, Sout=block.S_qk, inverse_out=False)
                n = 'self_attn.k_proj'
                m = layers_dict[n]
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=False, Sin=block.S_norm_qkv, Sout=block.S_qk, inverse_out=True)
                n = 'self_attn.v_proj'
                m = layers_dict[n]
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=subset['inspect'].Q2, transpose=False, Sin=block.S_norm_qkv, Sout=block.S_ov, inverse_out=False)
            else:
                # S_norm_upgate = SmoothModule(torch.ones(prev_op[0].weight.shape[-1],dtype=torch.float32,device=self.dev)).weight
                S_up_down = SmoothModule(torch.ones(layers_dict['mlp.up_proj'].weight.shape[0],dtype=torch.float32,device=self.dev))
                block.S_norm_upgate = None # S_norm_upgate
                block.S_up_down = S_up_down
                # for n in layers_dict.keys():
                n = 'mlp.up_proj'
                m = layers_dict[n]
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=False, Sin=block.S_norm_upgate, Sout=block.S_up_down, inverse_out=False)
                n = 'mlp.gate_proj'
                m = layers_dict[n]
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=False, Sin=block.S_norm_upgate, inverse_out=False)

        else:
            if self.config['model']['type'] in ['Opt']:
                self.bake_mean_into_linear(layers[0])

            n = list(layers_dict.keys())[0]
            m = layers[0]
            if 'is_mlp' in subset and subset['is_mlp']:
                if self.online_rotate:
                    apply_exact_had_to_linear(m, had_dim=-1, output=False)
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=None, transpose=True, Sin=block.S_up_down, inverse_out=False)
                
            else:
                self.replace_rotate_sm_fc(block, n, m, Q1=self.model.modality_model.Q1, Q2=self._get_block_Q2(block), transpose=True,Sin=block.S_ov)
                # self.replace_rotate_sm_fc(block, f'{self._atten_inspect_name}.v_proj', prev_op[0], Q1=self.model.modality_model.Q1, Q2=self._get_block_Q2(block), transpose=False)
    
    def replace_rotate_sm_fc(self, block, n, m, Q1=None, Q2=None, transpose=False, Sin=None, Sout=None, inverse_out=False):
        args = {}
        if hasattr(self, 'weight_rotate') and self.weight_rotate:
            args['Q1'] = Q1
            args['Q2'] = Q2
            args['transpose'] = transpose
            args['Sin'] = Sin
            args['Sout'] = Sout
            args['inverse_out'] = inverse_out

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
