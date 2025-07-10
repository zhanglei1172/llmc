import base64
import io
from io import BytesIO
from PIL import Image
import os
import random
import shutil

import numpy as np
import torch
from loguru import logger


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_config(config):
    if config.get('sparse', False):
        logger.info('Use sparsification method')
    else:

        def check_weight_setting(weight_setting):
            if weight_setting.granularity == 'per_group':
                assert weight_setting.group_size > 0
            elif weight_setting.granularity == 'per_head':
                assert weight_setting.head_num > 0

        for _, modality_config in config.quant.items():
            if not isinstance(modality_config, dict) or not modality_config.get('weight', False):
                continue
            if modality_config.weight.get('granularity', False):
                weight_setting = modality_config.weight
                check_weight_setting(weight_setting)
            if modality_config.weight.get('w_1', False):
                weight_setting = modality_config.weight.w_1
                check_weight_setting(weight_setting)
            if modality_config.weight.get('w_2', False):
                weight_setting = modality_config.weight.w_2
                check_weight_setting(weight_setting)
    if config.model.get('tokenizer_mode', False):
        assert (
            config.model.tokenizer_mode == 'slow'
            or config.model.tokenizer_mode == 'fast'
        ), 'Tokenizer_mode should be slow or fast.'
        logger.info(f'Tokenizer_mode is set to {config.model.tokenizer_mode}.')
    else:
        config.model.tokenizer_mode = 'slow'
        logger.info('Tokenizer_mode is set to slow.')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception(f'{path} existed before. Need check.')


def copy_files(source_dir, target_dir, substring):
    for filename in os.listdir(source_dir):
        if substring in filename:
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            shutil.copy(source_file, target_file)
            logger.info(f'Copied {filename} to {target_dir}')


def print_important_package_version():
    from importlib.metadata import version
    logger.info(f"torch : {version('torch')}")
    logger.info(f"transformers : {version('transformers')}")
    logger.info(f"tokenizers : {version('tokenizers')}")
    logger.info(f"huggingface-hub : {version('huggingface-hub')}")
    logger.info(f"datasets : {version('datasets')}")


def get_modality(config):
    modalities = []
    modality_configs = []
    compression_config = config.quant if 'quant' in config else config.sparse
    for modality in ['vision', 'language', 'video_gen']:
        if modality in compression_config:
            compression_config[modality].modality = modality
            modalities.append(modality)
            modality_configs.append(compression_config[modality])
    if not modalities:
        compression_config.modality = 'language'
        return ['language'], [compression_config]
    return modalities, modality_configs


def deploy_all_modality(blockwise_opts, quant_format):
    for blockwise_opt in blockwise_opts:
        blockwise_opt.deploy(quant_format)

def resize_image(input_base64, width, height, keep_aspect_ratio=True):
    """
    对图片进行resize操作

    :param input_base64: 输入图片的Base64字符串
    :param width: 目标宽度
    :param height: 目标高度
    :param keep_aspect_ratio: 是否保持图片原比例，默认为False
    :return: 调整大小后的图片的Base64字符串
    """
    try:
        post_processed = False
        if input_base64.startswith('data:image;base64,'):
            post_processed = True
            # 如果Base64字符串包含前缀，去掉前缀
            input_base64 = input_base64.split('base64,')[-1]
        # 将Base64字符串解码为字节数据
        image_data = base64.b64decode(input_base64)
        # 使用字节数据打开图片
        image = Image.open(io.BytesIO(image_data))

        # 如果图像模式为RGBA，转换为RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        if keep_aspect_ratio:
            # 计算缩放比例
            ratio = min(width / image.width, height / image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            # 调整图片大小
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)

            # 创建一个新的空白图像，大小为目标尺寸
            new_image = Image.new("RGB", (width, height))
            # 计算粘贴位置，使图片居中
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            # top = 0
            # 将调整大小后的图片粘贴到新图像的中心
            new_image.paste(resized_image, (left, top))
        else:
            # 不保持原比例，直接调整图片大小
            new_image = image.resize((width, height), Image.LANCZOS)

        # 将调整后的图片保存为字节数据
        buffer = io.BytesIO()
        new_image.save(buffer, format="JPEG")
        buffer.seek(0)
        # 将字节数据编码为Base64字符串
        output_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        if post_processed:
            output_base64 = 'data:image;base64,' + output_base64

        return output_base64
    except Exception as e:
        print(f"处理图片时出现错误: {e}")
        return None
