import gc

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .smoothquant import SmoothQuant
from .module_utils import _LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_


@ALGO_REGISTRY
class SmoothQuantCustom(SmoothQuant):
    @torch.no_grad()
    def filter_subset(self, input_name):
        if input_name == "mlp.down_proj":
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
        if not self.filter_subset(input_name):
            logger.info('Do not transform this subset.')
            return
        layers = list(layers_dict.values())
        scale = self.search_scale_subset(layers, input_feat[input_name])
        self.apply_scale(scale, prev_op, layers)
        if self.act_static:
            self.update_input_feat(scale, input_feat, layers_dict, False)
