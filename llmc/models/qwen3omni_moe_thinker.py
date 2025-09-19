from importlib.metadata import version

import packaging

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel
from llmc.compression.quantization.constant import ATTN_IMPL

@MODEL_REGISTRY
class Qwen3OmniMoeThinker(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.eval_name = 'Qwen25VLEval'
        import sys
        sys.path.insert(0, '/dataset/model_engine/Omni-vllm-transformers/vllm&transformers/transformers-internal/src/')
        from transformers import Qwen2_5_VLForConditionalGeneration
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.vlm_model_config, 'use_cache'):
                self.vlm_model_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        # from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
        # Qwen2_5_VLAttention.forward = torch.compile(Qwen2_5_VLAttention.forward)
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=ATTN_IMPL,
        )
        self.mm_model = self.vlm_model
        logger.info(f'self.vlm_model : {self.vlm_model}')

        self.vision_model = self.vlm_model.visual
        self.vision_projector = self.vision_model.merger
        self.vision_embed = self.vision_model.patch_embed
        self.vision_config = self.vlm_model_config.vision_config
        self.model = self.vlm_model
        self.model_config = self.vlm_model_config

        # self.min_pixels = 256 * 28 * 28
        # self.max_pixels = 1280 * 28 * 28
        # logger.warning(f'min_pixels is set to: {self.min_pixels}')
        # logger.warning(f'max_pixels is set to: {self.max_pixels}')
        logger.warning('You can refer to the link https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct '
                       'to get more info of image resolution for performance boost.')
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
        )
        self.processor.tokenizer.padding_side = 'left'

    def find_blocks(self):
        self.blocks = self.model.model.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.embed_tokens
        self.rotary_emb = self.model.model.rotary_emb

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

    def get_extra_modules(self, block):
        return {
            'mlp': block.mlp
        }

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.norm]

    def get_layers_except_blocks(self):
        if packaging.version.parse(version('transformers')) >= packaging.version.parse('4.45.0'):
            return [self.embed_tokens, self.rotary_emb, self.model.model.norm, self.model.lm_head] # noqa
        else:
            return [self.embed_tokens, self.model.model.norm, self.model.lm_head]

    def skip_layer_name(self):
        return ['lm_head']

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.post_attention_layernorm,
        }

    def get_subsets_in_block(self, block):
        layers = []
        layers.append(
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
            }
        )

        layers.append(
            {
                'layers': {'self_attn.o_proj': block.self_attn.o_proj},
                'prev_op': [block.self_attn.v_proj],
                'input': ['self_attn.o_proj'],
                'inspect': block.self_attn.o_proj,
                'has_kwargs': False,
            }
        )

        if hasattr(block.mlp, 'gate'):
            layers.append(
                {
                    'layers': {
                        **{f'mlp.experts.{i}.gate_proj': block.mlp.experts[i].gate_proj # noqa
                           for i in range(len(block.mlp.experts))},
                        **{f'mlp.experts.{i}.up_proj': block.mlp.experts[i].up_proj # noqa
                           for i in range(len(block.mlp.experts))},
                        # 'mlp.shared_expert.gate_proj': block.mlp.shared_expert.gate_proj, # noqa
                        # 'mlp.shared_expert.up_proj': block.mlp.shared_expert.up_proj, # noqa
                        'mlp.gate': block.mlp.gate,
                        # 'mlp.shared_expert_gate': block.mlp.shared_expert_gate,
                    },
                    'prev_op': [block.post_attention_layernorm],
                    'input': ['mlp'],
                    'inspect': block.mlp,
                    'has_kwargs': False,
                    'is_mlp': True,
                }
            )
            for i in range(len(block.mlp.experts)):
                layers.append(
                    {
                        'layers': {f'mlp.experts.{i}.down_proj': block.mlp.experts[i].down_proj}, # noqa
                        'prev_op': [block.mlp.experts[i].up_proj],
                        'input': [f'mlp.experts.{i}.down_proj'],
                        'inspect': block.mlp.experts[i].down_proj,
                        'has_kwargs': False,
                        'is_mlp': True,
                    }
                )
            # layers.append(
            #     {
            #         'layers': {'mlp.shared_expert.down_proj': block.mlp.shared_expert.down_proj}, # noqa
            #         'prev_op': [block.mlp.shared_expert.up_proj],
            #         'input': ['mlp.shared_expert.down_proj'],
            #         'inspect': block.mlp.shared_expert.down_proj,
            #         'has_kwargs': False,
            #     }
            # )
        else:
            layers.append(
                {
                    'layers': {
                        'mlp.gate_proj': block.mlp.gate_proj,
                        'mlp.up_proj': block.mlp.up_proj,
                    },
                    'prev_op': [block.post_attention_layernorm],
                    'input': ['mlp.gate_proj'],
                    'inspect': block.mlp,
                    'has_kwargs': False,
                }
            )

            layers.append(
                {
                    'layers': {'mlp.down_proj': block.mlp.down_proj},
                    'prev_op': [block.mlp.up_proj],
                    'input': ['mlp.down_proj'],
                    'inspect': block.mlp.down_proj,
                    'has_kwargs': False,
                }
            )

        return layers
