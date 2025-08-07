import gc

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .smoothquant import SmoothQuant
from .module_utils import (_LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_,
                           _TRANSFORMERS_LINEAR_TYPES_,
                           _TRANSFORMERS_LN_TYPES_, FakeQuantLinear,
                           LlmcFp8Linear)


@ALGO_REGISTRY
class SmoothQuantCustom(SmoothQuant):
    @torch.no_grad()
    def filter_subset(self, prev_op, input_name):
        if input_name in ("self_attn.o_proj", "mlp.down_proj") or isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            return True
        else:
            return False




    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        layers_dict = subset['layers']
        prev_op = subset['prev_op']
        input_name = subset['input'][0]

        if self.selected_layers and input_name not in self.selected_layers:
            logger.info(f'Skipping layer {input_name} as it is not in selected layers.')
            return
        if not self.filter_subset(prev_op, input_name):
            logger.info('Do not transform this subset.')
            return
        layers = list(layers_dict.values())
        if (
            isinstance(prev_op[0], (nn.Linear, FakeQuantLinear, LlmcFp8Linear))
            and prev_op[0].out_features != layers[0].in_features * 3
            and prev_op[0].out_features != layers[0].in_features * 2
            and prev_op[0].out_features != layers[0].in_features
        ):

            if self.has_gqa and self.do_gqa_trans:
                is_gqa = True
                # input_keys = list(input_feat.keys())
                # input_name = input_keys[input_keys.index(input_name) - 1]
            else:
                logger.info('Cannot apply scale. Do not transform this subset.')
                return
        else:
            is_gqa = False
        scale = self.search_scale_subset(layers, input_feat[input_name], is_gqa=is_gqa)
        self.apply_scale(scale, prev_op, layers)
        if self.act_static:
            self.update_input_feat(scale, input_feat, layers_dict, is_gqa)
