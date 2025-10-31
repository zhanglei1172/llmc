import gc
import json
import os

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix
from .module_utils import (_LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_,
                           LlmcRMSNorm, RotateLinear, get_module_name)
from .constant import *


@ALGO_REGISTRY
class QuarotOmni(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        self.dev = torch.device('cuda')
        self.add_quant_config()
        if self.modality == 'vision':
            self.vision_preprocess()
        elif self.modality == 'language':
            self.preprocess()
        else:
            raise ValueError(f'Unsupported modality {self.modality}')

    def vision_preprocess(self):
        self.Q = self.get_orthogonal_matrix()
        # self.Q = self.get_test_eye_matrix(self.hidden_size, self.dev)
        self.R2 = random_hadamard_matrix(self.hidden_size // self.num_heads, self.dev)

        # Rotate the vision projector
        vision_projectors = self.model.vision_projector
        # if vision_projector is not None:
        for vision_projector in vision_projectors:
            logger.info('Rotating vision projector layer.')
            pre_vison_proj_ln = vision_projector.ln_q
            self.fuse_ln_fcs(pre_vison_proj_ln, [vision_projector.mlp[0]])
            pre_vison_proj_ln_name = get_module_name(vision_projector, pre_vison_proj_ln)
            self.model.replace_module_subset(
                LlmcRMSNorm,
                vision_projector,
                {'layers': {pre_vison_proj_ln_name: pre_vison_proj_ln}},
                None,
                {},
            )
            vision_up_proj = [vision_projector.mlp[0]]
            for layer in vision_up_proj:
                W_ = layer.weight.data
                dtype = layer.weight.data.dtype
                init_shape = W_.shape
                temp = W_.reshape(-1, init_shape[-1] // self.hidden_size, self.hidden_size)
                temp = temp.to(device=self.dev, dtype=torch.float64) @ self.Q
                W_ = temp.reshape(init_shape)
                layer.weight.data = W_.to(device=layer.weight.device, dtype=dtype)

        embeddings = self.model.get_embed_layers()
        # # Rotate the vision embed layers
        # vision_embed = [self.model.vision_embed.proj]
        # if vision_embed is not None:
        for vision_embed in embeddings:
            logger.info('Rotating vision head layers.')
            for layer in vision_embed:
                W_ = layer.weight.data
                dtype = layer.weight.data.dtype
                init_shape = W_.shape
                temp = W_.reshape(self.hidden_size, -1)
                # bake mean
                temp = temp - temp.mean(dim=-2, keepdim=True)
                temp = self.Q.T @ temp.to(device=self.dev, dtype=torch.float64)
                W_ = temp.reshape(init_shape)
                layer.weight.data = W_.to(device=layer.weight.device, dtype=dtype)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    b_ = layer.bias.data.double()
                    b_ = b_ - b_.mean()
                    layer.bias.data = self.Q.T @ b_.to(device=self.dev, dtype=torch.float64)
                    layer.bias.data = layer.bias.data.to(layer.bias.device, dtype=dtype)

    def preprocess(self):
        if torch.equal(
            self.model.get_head_layers()[0].weight,
            self.model.get_embed_layers()[0].weight,
        ):
            logger.info('Tie weight! Copy embed_layer for head_layer!')
            del self.model.get_head_layers()[0].weight
            w = self.model.get_embed_layers()[0].weight.clone()
            self.model.get_head_layers()[0].weight = nn.Parameter(w, requires_grad=False)

        if not self.config['model']['type'].startswith('Qwen'):
            self.remove_mean_from_embed()

        self.Q = self.get_orthogonal_matrix()
        self.R2 = random_hadamard_matrix(self.hidden_size // self.num_heads, self.dev)
        self.rotate_embeddings(self.Q)

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

        self.rotate_head(self.Q)

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
            # docformatter: on
            dtype = rot_layer.weight.dtype
            device = self.Q.device
            W = rot_layer.weight.data.to(device=device, dtype=torch.float64)
            rot_layer.weight.data = torch.matmul(self.Q.T, W).to(device=ROTATE_DEV, dtype=dtype) # noqa
            if hasattr(rot_layer, 'bias') and rot_layer.bias is not None:
                b = rot_layer.bias.data.to(device=device, dtype=torch.float64)
                rot_layer.bias.data = torch.matmul(self.Q.T, b).to(device=ROTATE_DEV, dtype=dtype)

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def add_quant_config(self):
        self.rotate_mode = self.quant_config['special']['rotate_mode']

    def get_test_eye_matrix(self, size, device):
        return torch.eye(size, dtype=torch.float64).to(device)

    def random_orthogonal_matrix(self, size, device):
        torch.cuda.empty_cache()
        random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
        q, r = torch.linalg.qr(random_matrix)
        q *= torch.sign(torch.diag(r)).unsqueeze(0)
        return q

    def get_orthogonal_matrix(self):
        if self.rotate_mode == 'random':
            return self.random_orthogonal_matrix(self.hidden_size, self.dev)
        elif self.rotate_mode == 'hadamard':
            return random_hadamard_matrix(self.hidden_size, self.dev)
        else:
            raise ValueError(f'Unsupported mode {self.mode}')

    def block_transform(self, block):
        logger.info(f'Start transform the {self.block_idx+1}-th block')

        if self.online_rotate:
            self.replace_rotate_linears(block)
        subsets = self.model.get_subsets_in_block(block)
        for index, subset in enumerate(subsets):
            self.subset_transform(block, subset)

        self.model.replace_module_block(LlmcRMSNorm, block, self.block_idx, {})
        gc.collect()

        logger.info(f'block:{block}')
        logger.info(f'End transform the {self.block_idx+1}-th block')

    @torch.no_grad()
    def subset_transform(self, block, subset):
        prev_op = subset['prev_op']
        layers_dict = subset['layers']
        assert (
            len(prev_op) == 1
        ), 'Only support single prev_op. If multi prev_ops, code need to be updated.'

        layers = list(layers_dict.values())

        if 'skip_rotate' in subset and subset['skip_rotate']:
            return

        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            self.fuse_ln_fcs(prev_op[0], layers)
            self.rotate_pre_layers(layers, self.Q)
        else:
            if self.config['model']['type'] in ['Opt', 'StableLm']:
                self.bake_mean_into_fc(layers[0])

            if self.config['model']['type'] in ['Qwen3Omni'] and self.model.get_modality() == 'vision':
                self.bake_mean_into_fc(layers[0])

            if 'is_mlp' in subset and subset['is_mlp']:
                self.rotate_post_layers(
                    layers, self.Q, exact_had=True if self.online_rotate else False
                )
            else:
                self.rotate_post_layers(layers, self.Q, exact_had=False)
                if prev_op[0] is not None:
                    if 'qkv' in get_module_name(self.model.model, prev_op[0]):
                        qkv_data = list(torch.split(prev_op[0].weight.data, self.hidden_size, dim=0))
                        v_data = qkv_data[-1].clone()
                        prev_op[0].weight.data = v_data
                        if prev_op[0].bias is not None:
                            qkv_bias = list(torch.split(prev_op[0].bias.data, self.hidden_size, dim=0))
                            v_bias = qkv_bias[-1].clone()
                            prev_op[0].bias.data = v_bias
                        apply_exact_had_to_linear(prev_op[0], had_dim=self.head_dim, output=True, R2=self.R2)
                        qkv_data[-1] = prev_op[0].weight.data
                        prev_op[0].weight.data = torch.cat(qkv_data, dim=0)
                        if prev_op[0].bias is not None:
                            qkv_bias[-1] = prev_op[0].bias.data
                            prev_op[0].bias.data = torch.cat(qkv_bias, dim=0)
                    else:
                        apply_exact_had_to_linear(prev_op[0], had_dim=self.head_dim, output=True, R2=self.R2)
                apply_exact_had_to_linear(layers[0], had_dim=self.head_dim, output=False, R2=self.R2)

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
