import copy
import functools
import gc
import math
import os
import random
from contextlib import nullcontext
from math import inf

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import (_LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_,
                           _MODEL_LN_TYPES_PAIRS_, _TRANSFORMERS_LINEAR_TYPES_,
                           FakeQuantLinear)
from .train_utils import (LossFunction, NativeScalerWithGradNormCount,
                          TruncateFunction)


@ALGO_REGISTRY
class PrefixQuant(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)


        self.dev = torch.device('cuda')
        self.model_dtype = next(self.model.model.parameters()).dtype
        special_config = self.quant_config.get('special', {})
        if 'prefixed_tokens' in special_config:
            prefixed_tokens = special_config['prefixed_tokens']
            if len(prefixed_tokens) == 0:
                self.prefixed_kv = []
            else:
                output = self.model.model(
                    torch.tensor([prefixed_tokens], device=next(self.model.model.parameters()).device, dtype=torch.int64),
                    return_dict=True)
                prefixed_key_values = output.past_key_values
                self.prefixed_kv = [
                    [prefixed_key_values.key_cache[i].detach().to(self.dev),
                        prefixed_key_values.value_cache[i].detach().to(self.dev)]
                    for i in range(len(prefixed_key_values.key_cache))
                ]
            self.prefix_token_num = len(prefixed_tokens)
        elif 'prefixed_kv_path' in special_config:
            self.prefixed_kv = torch.load(
                special_config['prefixed_kv_path'],
                map_location=self.dev
            )
            self.prefix_token_num = self.prefixed_kv[0][0].shape[-2]
        else:
            raise ValueError(
                'Please specify either "prefixed_tokens" or "prefixed_kv_path" in'
                ' the quantization configuration.'
            )
        

        self.model.rotary_emb.register_forward_pre_hook(
            self._rotary_emb_input_hook(), with_kwargs=True
        )

    def _kv_cache_input_hook(self, attn_layer):
        def hook_fn(module, args, kwargs):
            # if self.prefix_token_num == 0:
            #     return args, kwargs
            past_key_value = kwargs['past_key_value']
            past_seen_tokens = past_key_value.get_seq_length(len(self.model.blocks)) if past_key_value is not None else 0
            if not past_key_value:
                past_key_value = DynamicCache()
                kwargs['past_key_value'] = past_key_value
            # assert past_key_value is not None, "past_key_value must not be None"

            old_update_cache = past_key_value.update
            def update_cache(key_states, value_states, layer_idx, cache_kwargs=None):
                inp_seq_len = key_states.shape[-2]
                key_states, value_states = old_update_cache(key_states, value_states, layer_idx, cache_kwargs)
                if key_states.shape[-2] < self.prefix_token_num + past_seen_tokens + inp_seq_len:
                    key_states = torch.cat(
                        [self.prefixed_kv[layer_idx][0].to(dtype=key_states.dtype, device=key_states.device), key_states], dim=-2
                    )
                    value_states = torch.cat(
                        [self.prefixed_kv[layer_idx][1].to(dtype=value_states.dtype, device=value_states.device), value_states], dim=-2
                    )
                return key_states, value_states
            if type(old_update_cache) != type(update_cache):
                setattr(past_key_value, 'update', update_cache)
            if 'cache_position' in kwargs:
                cache_position = kwargs['cache_position']
            else:
                cache_position = torch.arange(
                   past_seen_tokens, past_seen_tokens + seq_len, device=device
                )
                kwargs['cache_position'] = cache_position
            
            if 'hidden_states' not in kwargs:
                assert len(args) >= 1
                hidden_states = args[0]
            else:
                hidden_states = kwargs['hidden_states']
            bs, seq_len, _ = hidden_states.shape
            dtype = hidden_states.dtype
            device = hidden_states.device
            
            attention_mask = kwargs.get('attention_mask', None)
            target_length = past_seen_tokens + seq_len + 1
            if attention_mask is None:
                # attention_mask = torch.zeros(
                #     bs, 1, seq_len, seq_len + self.prefix_token_num,
                #     device=device, dtype=hidden_states.dtype
                # )
                min_dtype = torch.finfo(dtype).min
                causal_mask = torch.full(
                    (seq_len, target_length), fill_value=min_dtype, dtype=dtype, device=device
                )
                if seq_len != 1:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
                attention_mask = causal_mask[None, None, :, :].expand(bs, 1, -1, -1)
            attention_mask = torch.cat(
                    (torch.zeros(bs, 1, seq_len, self.prefix_token_num, device=device, dtype=dtype),
                     attention_mask), dim=-1
                )
            kwargs['attention_mask'] = attention_mask
            if 'position_ids' in kwargs:
                position_ids = kwargs['position_ids']
                position_ids = position_ids + self.prefix_token_num
                kwargs['position_ids'] = position_ids
            else:
                raise NotImplementedError

            return args, kwargs

        return hook_fn

    def _rotary_emb_input_hook(self,):
        def hook_fn(module, args, kwargs):
            args = list(args)
            if 'position_ids' in kwargs:
                position_ids = kwargs['position_ids']
                kwargs['position_ids'] = position_ids + self.prefix_token_num
            else:
                position_ids = args[1]
                args[1] = position_ids + self.prefix_token_num
            return tuple(args), kwargs
        return hook_fn
            

    @torch.no_grad()
    def register_prefix_kwargs(self, block):
        attn_layers_dict = self.model.get_attn_in_block(block)
        attn_layer = attn_layers_dict[list(attn_layers_dict.keys())[0]]
        attn_layer.register_forward_pre_hook(
            self._kv_cache_input_hook(attn_layer), with_kwargs=True
        )


    def block_forward(self, block, input_data=None):
        output = []

        if input_data is None:
            input_data = self.input['data']

        for i in range(len(input_data)):
            input_data[i] = input_data[i].to(device=next(block.parameters()).device)
            if (
                'attention_mask' in self.input['kwargs'][i]
                and self.input['kwargs'][i]['attention_mask'] is not None
            ):
                self.input['kwargs'][i]['attention_mask'] = self.input['kwargs'][i][
                    'attention_mask'
                ].cuda()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = block(input_data[i], **self.input['kwargs'][i])[0]
                    output.append(out)
        return output


    def block_transform(self, block):
        logger.info(f'Start transform the {self.block_idx}-th block')

        self.register_prefix_kwargs(block)

        logger.info(f'End transform the {self.block_idx}-th block')

    def deploy(self, quant_format):
        with torch.no_grad():
            if quant_format == 'origin_float' and 'save' in self.config and self.config.save.get('save_prefixed_kv', False):
                torch.save(self.prefixed_kv, os.path.join(self.config.save.save_path,'prefixed_kv.pth'))
        super().deploy(quant_format)
        self.model.convert_dtype(self.model_dtype)

    def save_model(self, path):
        self.model.convert_dtype(self.model_dtype)
        super().save_model(path)
