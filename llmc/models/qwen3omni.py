import librosa
from loguru import logger
from transformers import AutoConfig, AutoProcessor

try:
    from transformers import Qwen3OmniMoeForConditionalGeneration
except Exception:
    logger.warning(
        'Can not import Qwen3OmniMoeForConditionalGeneration. '
        'If you need it, please upgrade transformers.'
    )

from llmc.utils.registry_factory import MODEL_REGISTRY

from .qwen3moe import Qwen3Moe


@MODEL_REGISTRY
class Qwen3Omni(Qwen3Moe):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.omni_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.omni_model_config, 'use_cache'):
                self.omni_model_config.use_cache = False
        logger.info(f'self.omni_model_config : {self.omni_model_config}')
        self.omni_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.omni_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.mm_model = self.omni_model.thinker
        logger.info(f'self.omni_model : {self.omni_model}')

        self.vision_model = self.omni_model.thinker.visual
        self.vision_embed = [[self.vision_model.patch_embed.proj], [self.vision_model.pos_embed]]
        self.vision_config = self.omni_model_config.thinker_config.vision_config
        self.vision_projector = [self.vision_model.merger] + [self.vision_model.merger_list[i] for i in range(len(self.vision_config.deepstack_visual_indexes))]

        self.audio_model = self.omni_model.thinker.audio_tower
        self.audio_projector = self.audio_model.proj2
        self.audio_embed = self.audio_model.conv_out
        self.audio_config = self.omni_model_config.thinker_config.audio_config

        self.model = self.omni_model.thinker
        self.model_config = self.omni_model_config.thinker_config.text_config

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.processor.tokenizer.padding_side = 'left'

    def get_extra_rot_module_besides_embed_layers(self):
        return [self.audio_model.proj2, self.vision_model.merger.mlp[-1]] + [self.vision_model.merger_list[i].mlp[-1] for i in range(len(self.vision_config.deepstack_visual_indexes))]
    
    def build_tokenizer(self):
        super().build_tokenizer()
        if self.tokenizer is not None:
            self.tokenizer.padding_side = 'left'

    def get_embed_layers(self):
        if self.get_modality() == 'language':
            return super().get_embed_layers()
        elif self.get_modality() == 'vision':
            return self.vision_embed
        elif self.get_modality() == 'audio':
            return self.audio_embed
        else:
            raise Exception(f'{self.get_modality()} modality not supported!')

    def find_blocks(self):
        if self.get_modality() == 'language':
            super().find_blocks()
        elif self.get_modality() == 'vision':
            self.blocks = self.vision_model.blocks
        elif self.get_modality() == 'audio':
            self.blocks = self.audio_model.blocks
        else:
            raise Exception(f'{self.get_modality()} modality not supported!')
        
    def get_layernorms_in_block(self, block):
        if self.get_modality() == 'language':
            return super().get_layernorms_in_block(block)
        elif self.get_modality() == 'vision':
            return {
                'norm1': block.norm1,
                'norm2': block.norm2,
            }
        elif self.get_modality() == 'audio':
            return {
                'self_attn_layer_norm': block.self_attn_layer_norm,
                'final_layer_norm': block.final_layer_norm,
            }
        else:
            raise Exception(f'{self.get_modality()} modality not supported!')

    def get_subsets_in_block(self, block):
        if self.get_modality() == 'language':
            return super().get_subsets_in_block(block)
        elif self.get_modality() == 'vision':
            return [
                {
                    'layers': {
                        'attn.qkv': block.attn.qkv,
                    },
                    'prev_op': [block.norm1],
                    'input': ['attn.qkv'],
                    'inspect': block.attn,
                    'has_kwargs': True,
                },
                {
                    'layers': {'attn.proj': block.attn.proj},
                    'prev_op': [block.attn.qkv],
                    'input': ['attn.proj'],
                    'inspect': block.attn.proj,
                    'has_kwargs': False,
                },
                {
                    'layers': {
                        'mlp.linear_fc1': block.mlp.linear_fc1,
                    },
                    'prev_op': [block.norm2],
                    'input': ['mlp.linear_fc1'],
                    'inspect': block.mlp,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
                {
                    'layers': {'mlp.linear_fc2': block.mlp.linear_fc2},
                    'prev_op': [block.mlp.linear_fc1],
                    'input': ['mlp.linear_fc2'],
                    'inspect': block.mlp.linear_fc2,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
            ]
        elif self.get_modality() == 'audio':
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
                    'layers': {'self_attn.out_proj': block.self_attn.out_proj},
                    'prev_op': [block.self_attn.v_proj],
                    'input': ['self_attn.out_proj'],
                    'inspect': block.self_attn.out_proj,
                    'has_kwargs': False,
                },
                {
                    'layers': {
                        'mlp.fc1': block.mlp.fc1,
                    },
                    'prev_op': [block.final_layer_norm],
                    'input': ['mlp.fc1'],
                    'inspect': block.mlp,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
                {
                    'layers': {'mlp.fc2': block.mlp.fc2},
                    'prev_op': [block.mlp.fc1],
                    'input': ['mlp.fc2'],
                    'inspect': block.mlp.fc2,
                    'has_kwargs': False,
                    'is_mlp': True,
                },
            ]
        else:
            raise Exception(f'{self.get_modality()} modality not supported!')

    # def batch_process(self, audio_qas, calib_or_eval='eval', apply_chat_template=True, return_inputs=True): # noqa
    #     assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
    #     assert apply_chat_template
    #     messages = []
    #     answers = []
    #     for idx in range(len(audio_qas)):
    #         audio_path = audio_qas[idx]['audio']
    #         if audio_path is not None:
    #             content = []
    #             if not isinstance(audio_path, list):
    #                 audio_path = [audio_path]
    #             for audio_idx in range(len(audio_path)):
    #                 content.append({'type': 'audio', 'audio': audio_path[audio_idx]})
    #             if 'question' in audio_qas[idx]:
    #                 content.append({'type': 'text', 'text': audio_qas[idx]['question']})
    #             message = [{'role': 'user', 'content': content}]
    #         else:
    #             message = [
    #                 {
    #                     'role': 'user',
    #                     'content': [
    #                         {'type': 'text', 'text': audio_qas[idx]['question']}
    #                     ],
    #                 }
    #             ]
    #         messages.append(message)
    #         answers.append(audio_qas[idx]['answer'] + '<|im_end|>')
    #     texts = [
    #         self.processor.apply_chat_template(
    #             msg, tokenize=False, add_generation_prompt=True
    #         )
    #         for msg in messages
    #     ]
    #     if calib_or_eval == 'calib' and self.config['calib'].get('add_answer', False):
    #         texts = [texts[n] + answers[n] for n in range(len(texts))]
    #     if calib_or_eval == 'calib':
    #         logger.info(f'Calib data is:\n{texts}')
    #     if not return_inputs:
    #         return texts
    #     audios = []
    #     for conversation in messages:
    #         for message in conversation:
    #             if isinstance(message['content'], list):
    #                 for ele in message['content']:
    #                     if ele['type'] == 'audio':
    #                         audios.append(
    #                             librosa.load(
    #                                 ele['audio'],
    #                                 sr=self.processor.feature_extractor.sampling_rate,
    #                             )[0]
    #                         )

    #     inputs = self.processor(
    #         text=texts, audios=audios, return_tensors='pt', padding=True
    #     ).to(next(self.alm_model.parameters()).dtype)
    #     return inputs
