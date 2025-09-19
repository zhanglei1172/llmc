from importlib.metadata import version

import packaging

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class Qwen25(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def find_blocks(self):
        self.blocks = self.model.language_model.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.language_model.embed_tokens
        if packaging.version.parse(version('transformers')) >= packaging.version.parse('4.45.0'):
            self.rotary_emb = self.model.language_model.rotary_emb

    def find_block_name(self):
        self.block_name_prefix = 'model.layers'
        self.pairs = {'q_proj': 'qkv', 'o_proj': 'out', 'up_proj': 'fc1'}

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_attn_in_block(self, block):
        return {'self_attn': block.self_attn}

    def get_attention_rotary_layers(self):
        if packaging.version.parse(version('transformers')) >= packaging.version.parse('4.45.0'):
            return [self.rotary_emb]
        else:
            return []

    def get_head_layers(self):
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        return [self.model.language_model.norm]

    def get_layers_except_blocks(self):
        if packaging.version.parse(version('transformers')) >= packaging.version.parse('4.45.0'):
            return [self.embed_tokens, self.rotary_emb, self.model.language_model.norm, self.model.lm_head] # noqa
        else:
            return [self.embed_tokens, self.model.language_model.norm, self.model.lm_head]

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.post_attention_layernorm,
        }

    # flake8: noqa
    def apply_chat_template(self, prompt):
        messages = [
            {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def get_matmul_in_block(self, block):
        return {
            'self_attn.matmul_1': block.self_attn.matmul_1,
            'self_attn.matmul_2': block.self_attn.matmul_2,
        }

    def get_softmax_in_block(self, block):
        return {'self_attn.softmax': block.self_attn.softmax}

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'self_attn.q_proj': block.self_attn.q_proj,
                    'self_attn.k_proj': block.self_attn.k_proj,
                    'self_attn.v_proj': block.self_attn.v_proj,
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attn.q_proj'],
                'inspect': block.self_attn,
                'has_kwargs': True,
            },
            {
                'layers': {'self_attn.o_proj': block.self_attn.o_proj},
                'prev_op': [block.self_attn.v_proj],
                'input': ['self_attn.o_proj'],
                'inspect': block.self_attn.o_proj,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'mlp.gate_proj': block.mlp.gate_proj,
                    'mlp.up_proj': block.mlp.up_proj,
                },
                'prev_op': [block.post_attention_layernorm],
                'input': ['mlp.gate_proj'],
                'inspect': block.mlp,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'mlp.down_proj': block.mlp.down_proj},
                'prev_op': [block.mlp.up_proj],
                'input': ['mlp.down_proj'],
                'inspect': block.mlp.down_proj,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]
