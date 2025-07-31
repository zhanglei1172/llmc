import json
import os
from abc import ABCMeta
import random
import functools

import torch
from datasets import load_dataset, load_from_disk
from loguru import logger
from PIL import Image
from torch.nn import functional as F

from .specified_preproc import PREPROC_REGISTRY


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, tokenizer, calib_cfg, batch_process=None, processor=None):
        # calib_cfg
        logger.info(f'calib_cfg : {calib_cfg}')
        self.tokenizer = tokenizer
        self.batch_process = batch_process
        self.processor = processor
        self.calib_dataset_name = calib_cfg['name']
        self.padding = calib_cfg.get('padding', False)
        if self.calib_dataset_name == 'ultrachat':
            assert self.padding
        self.download = calib_cfg['download']
        self.load_from_txt = calib_cfg.get('load_from_txt', False)
        self.calib_dataset_path = calib_cfg.get('path', None)
        self.apply_chat_template = calib_cfg.get('apply_chat_template', False)
        self.n_samples = calib_cfg.get('n_samples', None)
        self.calib_bs = calib_cfg['bs']
        if self.calib_dataset_name in ['t2v', 'i2v']:
            assert self.calib_bs == 1
        self.seq_len = calib_cfg.get('seq_len', None)
        self.preproc = calib_cfg.get('preproc', False)
        if self.calib_dataset_name == 'ultrachat':
            assert self.preproc == 'ultrachat_general'
        if self.preproc == 'original_txt':
            assert self.seq_len is None
        self.seed = calib_cfg['seed']
        self.special_config = calib_cfg.get('special', {})
        self.calib_dataset_field_map = {
            'pileval': 'text',
            'c4': 'text',
            'wikitext2': 'text',
            'ptb': 'sentence',
        }
        if self.calib_dataset_name in self.calib_dataset_field_map:
            self.key = self.calib_dataset_field_map[self.calib_dataset_name]
        self.build_calib_dataset()

    def build_calib_dataset(self):
        if self.download:
            if self.calib_dataset_name == 'pileval':
                self.calib_dataset = load_dataset(
                    'mit-han-lab/pile-val-backup', split='validation'
                )
            elif self.calib_dataset_name == 'c4':
                self.calib_dataset = load_dataset(
                    'allenai/c4',
                    data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                    split='train',
                )
            elif self.calib_dataset_name == 'wikitext2':
                self.calib_dataset = load_dataset(
                    'wikitext', 'wikitext-2-raw-v1', split='train'
                )
            elif self.calib_dataset_name == 'ptb':
                self.calib_dataset = load_dataset(
                    'ptb_text_only', 'penn_treebank', split='train'
                )
            elif self.calib_dataset_name == 'ultrachat':
                self.calib_dataset = load_dataset(
                    'HuggingFaceH4/ultrachat_200k', split='train_sft'
                )
            else:
                raise Exception(f'Not support {self.calib_dataset_name} dataset.')
        else:
            if self.calib_dataset_name.startswith('V4_'):
                try:
                    from xq_eval.task_utils import TASK2EVAL
                except Exception as e:
                    logger.error(f'Plese make sure git submodule updated!')
                    raise e
                self.task_clss = TASK2EVAL[self.calib_dataset_name.strip('V4_')]
                self.calib_dataset = None
            elif self.calib_dataset_name in [
                'custom_txt',
                'custom_mm',
                'images',
                't2v',
                'i2v',
            ]:
                self.calib_dataset = self.get_custom_dataset(self.calib_dataset_path)
            else:
                self.calib_dataset = load_from_disk(self.calib_dataset_path)

    def get_calib_model_inputs(self, samples):
        if not self.padding:
            if self.calib_dataset_name in ['t2v', 'i2v']:
                calib_model_inputs = samples
            elif self.calib_dataset_name == 'images':
                calib_model_inputs = self.get_batch_process(samples)
            else:
                assert not self.calib_dataset_name == 'custom_mm'
                if self.calib_dataset_name == 'custom_txt':
                    txts = self.batch_process(
                        samples,
                        calib_or_eval='calib',
                        apply_chat_template=self.apply_chat_template,
                        return_inputs=False,
                    )
                else:
                    txts = self.calib_dataset
                preproc = PREPROC_REGISTRY[self.preproc]
                preproc_param_dict = {
                    'calib_dataset': txts,
                    'tokenizer': self.tokenizer,
                    'n_samples': self.n_samples,
                    'seq_len': self.seq_len,
                }
                if self.preproc == 'txt_general_preproc':
                    preproc_param_dict['key'] = self.key
                elif self.preproc == 'v4_general_preproc':
                    preproc_param_dict['task_clss'] = self.task_clss
                    preproc_param_dict['processor'] = self.processor
                    preproc_param_dict['data_path'] = self.calib_dataset_path
                    if self.special_config:
                        preproc_param_dict.update(self.special_config)
                    return preproc(**preproc_param_dict)
                samples = preproc(**preproc_param_dict)
                calib_model_inputs = []
                if self.calib_bs == -1:
                    batch = torch.cat(samples, dim=0)
                    calib_model_inputs.append({'input_ids': batch})
                elif self.calib_bs == 1:
                    for i in range(len(samples)):
                        calib_model_inputs.append({'input_ids': samples[i]})
                elif self.calib_bs > 1:
                    for i in range(0, len(samples), self.calib_bs):
                        start = i
                        end = min(i + self.calib_bs, len(samples))
                        batch = samples[start:end]
                        batch = torch.cat(batch, dim=0)
                        calib_model_inputs.append({'input_ids': batch})
        else:
            assert (
                self.calib_dataset_name == 'custom_txt'
                or self.calib_dataset_name == 'custom_mm'
            )
            calib_model_inputs = self.get_batch_process(
                samples if self.n_samples == -1 else random.choices(samples, k=self.n_samples)
            )
        return calib_model_inputs

    def get_batch_process(self, samples):
        calib_model_inputs = []
        if self.calib_bs == -1:
            calib_model_inputs.append(
                self.batch_process(
                    samples,
                    calib_or_eval='calib',
                    apply_chat_template=self.apply_chat_template,
                )
            )
        elif self.calib_bs == 1:
            calib_model_inputs = [
                self.batch_process(
                    [sample],
                    calib_or_eval='calib',
                    apply_chat_template=self.apply_chat_template,
                )
                for sample in samples
            ]
        elif self.calib_bs > 1:
            for i in range(0, len(samples), self.calib_bs):
                start = i
                end = min(i + self.calib_bs, len(samples))
                batch = samples[start:end]
                calib_model_inputs.append(
                    self.batch_process(
                        batch,
                        calib_or_eval='calib',
                        apply_chat_template=self.apply_chat_template,
                    )
                )
        return calib_model_inputs

    def get_calib_dataset(self):
        if self.calib_dataset is not None:
            samples = self.calib_dataset[
                int(os.environ['RANK'])::int(os.environ['WORLD_SIZE'])
            ]
            logger.info(f'len(samples) rank : {len(samples)}')
        else:
            samples = None
        calib_model_inputs = self.get_calib_model_inputs(samples)
        logger.info(f'len(calib_model_inputs) : {len(calib_model_inputs)}')
        if self.padding:
            padding_mask = [
                calib_model_input['attention_mask']
                for calib_model_input in calib_model_inputs
            ]
        else:
            padding_mask = None
        return calib_model_inputs, padding_mask

    def get_custom_dataset(self, custom_dataset_path):
        custom_data_samples = []
        # audio_img_qa_json = os.path.join(custom_dataset_path, 'samples.json')
        if os.path.isdir(custom_dataset_path):
            audio_img_qa_jsons = [os.path.join(custom_dataset_path, x) for x in os.listdir(custom_dataset_path)]
        else:
            audio_img_qa_jsons = [custom_dataset_path]
        for audio_img_qa_json in audio_img_qa_jsons:
            if audio_img_qa_json.endswith('.json'):
                with open(audio_img_qa_json) as fp:
                    custom_data_samples.extend(json.load(fp))
        for idx in range(len(custom_data_samples)):
            if 'audio' in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]['audio'], list):
                    for audio_idx in range(len(custom_data_samples[idx]['audio'])):
                        custom_data_samples[idx]['audio'][audio_idx] = os.path.join(
                            custom_dataset_path, custom_data_samples[idx]['audio'][audio_idx]
                        )
                else:
                    custom_data_samples[idx]['audio'] = os.path.join(
                        custom_dataset_path, custom_data_samples[idx]['audio']
                    )
            else:
                custom_data_samples[idx]['audio'] = None
            if 'image' in custom_data_samples[idx]:
                if isinstance(custom_data_samples[idx]['image'], list):
                    for img_idx in range(len(custom_data_samples[idx]['image'])):
                        custom_data_samples[idx]['image'][img_idx] = os.path.join(
                            custom_dataset_path, custom_data_samples[idx]['image'][img_idx]
                        )
                elif not custom_data_samples[idx]['image'].startswith('data:image;base64,'):
                    # custom_data_samples[idx]['image'] = os.path.join(
                    #     custom_dataset_path, custom_data_samples[idx]['image']
                    # )
                    custom_data_samples[idx]['image'] = f"data:image;base64,{custom_data_samples[idx]['image']}"
            else:
                custom_data_samples[idx]['image'] = None
            if 'question' not in custom_data_samples[idx]:
                if 'prompt' in custom_data_samples[idx]:
                    custom_data_samples[idx]['question'] = custom_data_samples[idx]['prompt']
                else:
                    custom_data_samples[idx]['question'] = ''
            if 'answer' not in custom_data_samples[idx]:
                if 'label' in custom_data_samples[idx]:
                    custom_data_samples[idx]['answer'] = custom_data_samples[idx]['label']
                else:
                    custom_data_samples[idx]['answer'] = ''
            if isinstance(custom_data_samples[idx]['answer'], dict):
                custom_data_samples[idx]['answer'] = json.dumps(custom_data_samples[idx]['answer'],ensure_ascii=False)
            if 'prompt' not in custom_data_samples[idx]:
                custom_data_samples[idx]['prompt'] = ''
            if 'negative_prompt' not in custom_data_samples[idx]:
                custom_data_samples[idx]['negative_prompt'] = ''
        return custom_data_samples

class MixDataset(BaseDataset):
    def __init__(self, tokenizer, calib_cfg, batch_process=None, processor=None):
        if isinstance(calib_cfg, dict):
            calib_cfg = [calib_cfg]
        self.datasets = []
        for cfg in calib_cfg:
            dataset = BaseDataset(tokenizer, cfg, batch_process, processor)
            self.datasets.append(dataset)
    
    def get_calib_dataset(self):
        calib_model_inputs = []
        padding_mask = []
        for dataset in self.datasets:
            inputs, masks = dataset.get_calib_dataset()
            calib_model_inputs.extend(inputs)
            if masks is not None:
                padding_mask.extend(masks)
        if padding_mask:
            assert len(calib_model_inputs) == len(padding_mask), \
                "The length of calib_model_inputs and padding_mask must be the same."
        else:
            padding_mask = None
        if len(calib_model_inputs) == 0:
            raise ValueError("No samples found in the mixed datasets.")
        logger.info(f'len(calib_model_inputs) : {len(calib_model_inputs)}')
        return calib_model_inputs, padding_mask
