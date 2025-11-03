from importlib.metadata import version

import packaging

from llmc.utils.registry_factory import MODEL_REGISTRY
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
import transformers
from transformers.configuration_utils import PretrainedConfig
transformers_version = packaging.version.parse(packaging.version.parse(transformers.__version__).base_version)
if transformers_version >= packaging.version.parse('4.57.0'):
    latest_transformers = True
else:
    latest_transformers = False
from loguru import logger
from .base_model import BaseModel
from llmc.compression.quantization.constant import ATTN_IMPL
from llmc.compression.quantization.module_utils import _REALQUANT_LINEAR_MAP_

@MODEL_REGISTRY
class Qwen3OmniMoe(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        # self.eval_name = 'Qwen25VLEval'
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.vlm_model_config.enable_audio_output = False
        if not self.use_cache:
            if hasattr(self.vlm_model_config, 'use_cache'):
                self.vlm_model_config.use_cache = False
        logger.info(f'self.vlm_model_config : {self.vlm_model_config}')
        def set_dtype(config, dtype):
            if hasattr(config, 'dtype'):
                config.dtype = dtype
            for k in config:
                sub_config = getattr(config, k)
                if isinstance(sub_config, PretrainedConfig):
                    set_dtype(sub_config, dtype)
        if not isinstance(self.torch_dtype, str):
            set_dtype(self.vlm_model_config, self.torch_dtype)
        if latest_transformers:
            from transformers import Qwen3OmniMoeForConditionalGeneration
            from transformers import Qwen3OmniMoeProcessor
            from accelerate import infer_auto_device_map, init_empty_weights
            import torch
            if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
                with init_empty_weights():
                    self.vlm_model = Qwen3OmniMoeForConditionalGeneration._from_config(
                        self.vlm_model_config,
                        # trust_remote_code=True,
                        dtype=self.torch_dtype,
                        # low_cpu_mem_usage=True,
                        attn_implementation=ATTN_IMPL,
                    )
            else:
                self.vlm_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    self.model_path,
                    config=self.vlm_model_config,
                    trust_remote_code=True,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation=ATTN_IMPL,
                )
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        else:
            self.vlm_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.vlm_model_config,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation=ATTN_IMPL,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
        self.mm_model = self.vlm_model.thinker
        logger.info(f'self.vlm_model : {self.vlm_model}')

        self.vision_model = self.vlm_model.thinker.visual
        self.audio_model = self.vlm_model.thinker.audio_tower
        self.vision_projector = self.vision_model.merger
        self.vision_embed = self.vision_model.patch_embed
        self.vision_config = self.vlm_model_config.thinker_config.vision_config
        self.model = self.vlm_model.thinker
        # self.model.model = self.vlm_model.thinker
        self.model_config = self.vlm_model_config.thinker_config.text_config
        self.audio_config = self.vlm_model_config.thinker_config.audio_config

        # self.min_pixels = 256 * 28 * 28
        # self.max_pixels = 1280 * 28 * 28
        # logger.warning(f'min_pixels is set to: {self.min_pixels}')
        # logger.warning(f'max_pixels is set to: {self.max_pixels}')
        logger.warning('You can refer to the link https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct '
                       'to get more info of image resolution for performance boost.')

        self.processor.tokenizer.padding_side = 'left'

    def set_modality(self, modality='language'):
        assert modality in ['audio', 'vision', 'language', 'video_gen']
        self.modality = modality
        if self.modality == 'audio':
            self.modality_model = self.audio_model
        elif self.modality == 'vision':
            self.modality_model = self.vision_model
        elif self.modality == 'video_gen':
            self.modality_model = self.vision_projector
        else:
            self.modality_model = self.model.model
        self.update_key_info()

    def get_extra_rot_module_besides_embed_layers(self):
        return [self.model.audio_tower.proj2, self.model.visual.merger.mlp[-1]] + [self.model.visual.merger_list[i].mlp[-1] for i in range(len(self.vision_config.deepstack_visual_indexes))]

    def find_blocks(self):
        self.blocks = self.modality_model.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.embed_tokens
        self.rotary_emb = self.model.model.rotary_emb

    def get_embedding_layer(self):
        return [self.modality_model.positional_embedding.positional_embedding]

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
        return [self.modality_model.norm]

    def get_layers_except_blocks(self):
        if packaging.version.parse(version('transformers')) >= packaging.version.parse('4.45.0'):
            return [self.embed_tokens, self.rotary_emb, self.modality_model.norm, self.model.lm_head] # noqa
        else:
            return [self.embed_tokens, self.modality_model.norm, self.model.lm_head]

    def skip_layer_name(self):
        ret = []
        for name, module in self.vlm_model.named_modules():
            try:
                if 'Linear' in module.__class__.__name__ and type(module) not in _REALQUANT_LINEAR_MAP_.values():
                    ret.append(name)
            except:
                pass
        return ret

    def has_bias(self):
        return False

    def get_layernorms_in_block(self, block):
        try:
            return {
                'input_layernorm': block.input_layernorm,
                'post_attention_layernorm': block.post_attention_layernorm,
            }
        except Exception as e:
            return {
                'input_layernorm': block.self_attn_layer_norm,
                'post_attention_layernorm': block.final_layer_norm,
            }

    def get_prev_decoder_layers(self):
        return [self.modality_model.conv_out]

    def get_subsets_in_block(self, block):
        if self.modality == 'audio':
            return [
                {
                    'layers': {
                        'self_attn.q_proj': block.self_attn.q_proj,
                        'self_attn.k_proj': block.self_attn.k_proj,
                        'self_attn.v_proj': block.self_attn.v_proj,
                    },
                    'prev_op': [block.self_attn_layer_norm],
                    'input': ['self_attn.q_proj'],
                    'inspect': block.self_attn,
                    'has_kwargs': True,
                },
                {
                    'layers': {'self_attn.o_proj': block.self_attn.out_proj},
                    'prev_op': [block.self_attn.v_proj],
                    'input': ['self_attn.o_proj'],
                    'inspect': block.self_attn.out_proj,
                    'has_kwargs': False,
                },
                {
                    'layers': {
                        'block.fc1': block.fc1,
                    },
                    'prev_op': [block.final_layer_norm],
                    'input': ['block.fc1'],
                    'inspect': block,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
                {
                    'layers': {'block.fc2': block.fc2},
                    'prev_op': [block.fc1],
                    'input': ['block.fc2'],
                    'inspect': block.fc2,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
            ]

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
