import base64
import io
from io import BytesIO
from PIL import Image
import os
import random
import shutil
import contextlib
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
    # else:
    #     raise Exception(f'{path} existed before. Need check.')


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
    for modality in ['vision', 'language', 'video_gen', 'audio']:
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

import copy


def get_non_persistent_buffers(module: torch.nn.Module, recurse: bool = False, fqns: bool = False):
    """
    Gather all non persistent buffers of a given modules into a set

    Args:
        module (`nn.Module`):
            The module we want the non persistent buffers on.
        recurse (`bool`, *optional*, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct non persistent buffers.
        fqns (`bool`, *optional*, defaults to `False`):
            Whether or not to return the fully-qualified names of the non persistent buffers.
    """

    non_persistent_buffers_set = module._non_persistent_buffers_set
    if recurse:
        for n, m in module.named_modules():
            if fqns:
                non_persistent_buffers_set |= {n + "." + b for b in m._non_persistent_buffers_set}
            else:
                non_persistent_buffers_set |= m._non_persistent_buffers_set

    return non_persistent_buffers_set

@contextlib.contextmanager
def patch_attr(base: object, attr: str, value):
    """
    Patch the value of an object attribute. Original value is restored upon exit

    :param base: object which has the attribute to patch
    :param attr: name of the the attribute to patch
    :param value: used to replace original value

    Usage:
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace()
    >>> with patch_attr(obj, "attribute", "value"):
    ...     assert obj.attribute == "value"
    >>> assert not hasattr(obj, "attribute")
    """
    _sentinel = object()
    original_value = getattr(base, attr, _sentinel) # 针对类方法、属性、实例属性
    #setattr(base, attr, module_to_cuda.__get__(base)) 针对实例方法

    setattr(base, attr, value)
    try:
        yield
    finally:
        if original_value is not _sentinel:
            setattr(base, attr, original_value)
        else:
            delattr(base, attr)

@contextlib.contextmanager
def patch_module_to_cpu(base: object):
    """
    Patch the value of an object attribute. Original value is restored upon exit

    :param base: object which has the attribute to patch
    :param attr: name of the the attribute to patch
    :param value: used to replace original value

    Usage:
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace()
    >>> with patch_attr(obj, "attribute", "value"):
    ...     assert obj.attribute == "value"
    >>> assert not hasattr(obj, "attribute")
    """
    # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    attr = "cpu"
    _sentinel = object()
    original_value = getattr(base, attr, _sentinel)

    setattr(base, attr, module_to_cpu)
    try:
        yield
    finally:
        if original_value is not _sentinel:
            setattr(base, attr, original_value)
        else:
            delattr(base, attr)

@contextlib.contextmanager
def patch_module_to_cuda(base: object):
    """
    Patch the value of an object attribute. Original value is restored upon exit

    :param base: object which has the attribute to patch
    :param attr: name of the the attribute to patch
    :param value: used to replace original value

    Usage:
    >>> from types import SimpleNamespace
    >>> obj = SimpleNamespace()
    >>> with patch_attr(obj, "attribute", "value"):
    ...     assert obj.attribute == "value"
    >>> assert not hasattr(obj, "attribute")
    """
    # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    attr = "cuda"
    _sentinel = object()
    original_value = getattr(base, attr, _sentinel)

    # setattr(base, attr, module_to_cuda.__get__(base))
    setattr(base, attr, module_to_cuda)
    try:
        yield
    finally:
        if original_value is not _sentinel:
            setattr(base, attr, original_value)
        else:
            delattr(base, attr)

@torch.no_grad()
def module_to_cuda(self: torch.nn.Module):
    """
    Move a module to CUDA
    :param module: module to move
    """
    if not isinstance(self, torch.nn.Module):
        return self.to('cuda')
    src_rank = 0
    device = "cuda"
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    non_persistent_buffer_fqns = get_non_persistent_buffers(self, recurse=True, fqns=True)

    
    _sd = {}
    meta_sd = self.state_dict()

    for name, sd_param in meta_sd.items():
        if rank == src_rank:
            send_tensor = sd_param.to(device)
            torch.distributed.broadcast(send_tensor, src=src_rank)
            _sd[name] = send_tensor
        else:
            recv_tensor = torch.empty_like(sd_param, device=device)
            torch.distributed.broadcast(recv_tensor, src=src_rank)
            _sd[name] = recv_tensor
    self.load_state_dict(_sd, assign=True)
        
    del _sd
    
    original_non_persistent_buffers = copy.deepcopy(
        {k: v for k, v in self.named_buffers() if k in non_persistent_buffer_fqns}
    )
    
    for fqn, buffer_tensor in original_non_persistent_buffers.items():
        if rank == src_rank:
            buffer_tensor = buffer_tensor.to(device)
        else:
            buffer_tensor = torch.empty_like(buffer_tensor, device=device)

        if "." in fqn:
            parent_fqn, local_buffer_name = fqn.rsplit(".", 1)
            parent_module = self.get_submodule(parent_fqn)
        else:
            local_buffer_name = fqn
            parent_module = self

        parent_module.register_buffer(local_buffer_name, buffer_tensor, persistent=False)

    return self


def module_to_cpu(self: torch.nn.Module):
    """
    Move a module to CPU

    :param module: module to move
    """
    if not isinstance(self, torch.nn.Module):
        return self.to('cpu')
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        return self.to('cpu')
    else:
        return self.to('meta')