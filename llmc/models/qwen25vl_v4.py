import inspect
from typing import Optional, Union
from importlib.metadata import version
import packaging

import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedType
from loguru import logger
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    logger.warning(
        "Can not import Qwen2_5_VLForConditionalGeneration. "
        "If you need it, please upgrade transformers."
    )

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    logger.warning(
        "Can not import qwen_vl_utils. "
        "If you need it, please pip install qwen-vl-utils"
    )

from llmc.utils.registry_factory import MODEL_REGISTRY
from llmc.utils import resize_image

from .qwen25vl import Qwen25VL


@MODEL_REGISTRY
class Qwen25VL_V4(Qwen25VL):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.eval_name = "Qwen25VLEval"
        self.vlm_model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if not self.use_cache:
            if hasattr(self.vlm_model_config, "use_cache"):
                self.vlm_model_config.use_cache = False
        logger.info(f"self.vlm_model_config : {self.vlm_model_config}")
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.vlm_model_config,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager", # TODO for quant_attn
        )

        class ExpandVocabLinear(nn.Module):
            def __init__(self, ori_module):
                super(ExpandVocabLinear, self).__init__()

                self.ori_module = ori_module
                self.mask_start_id = 152064

            def forward(self, x):
                # out = F.linear(x, self.ori_module.weight, bias=self.ori_module.bias)
                out = self.ori_module(x)
                out[:, :, self.mask_start_id :] = -10000.0
                return out

        self.vlm_model._tied_weights_keys = ["lm_head.ori_module.weight"]
        self.vlm_model._tied_weights_keys = None  # []

        self.vlm_model.lm_head = ExpandVocabLinear(self.vlm_model.lm_head)

        self.mm_model = self.vlm_model
        logger.info(f"self.vlm_model : {self.vlm_model}")

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
        logger.warning(
            "You can refer to the link https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct "
            "to get more info of image resolution for performance boost."
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
        )
        self.processor.tokenizer.padding_side = "left"

    def get_head_layers(self):
        return [self.model.lm_head.ori_module]

    def get_layers_except_blocks(self):
        if packaging.version.parse(version("transformers")) >= packaging.version.parse(
            "4.45.0"
        ):
            return [
                self.embed_tokens,
                self.rotary_emb,
                self.model.language_model.norm,
                self.model.lm_head.ori_module,
            ]  # noqa
        else:
            return [
                self.embed_tokens,
                self.model.language_model.norm,
                self.model.lm_head.ori_module,
            ]

    def skip_layer_name(self):
        return ["lm_head.ori_module"]

    def before_save_model(self):
        self.vlm_model.lm_head.register_parameter(
            "weight",
            nn.Parameter(self.vlm_model.lm_head.ori_module.weight.data.clone()),
        )
        if (
            hasattr(self.vlm_model.lm_head.ori_module, "bias")
            and self.vlm_model.lm_head.ori_module.bias is not None
        ):
            self.vlm_model.lm_head.register_parameter(
                "bias",
                nn.Parameter(self.vlm_model.lm_head.ori_module.bias.data.clone()),
            )


try:
    from lmms_eval.api.model import lmms
    from lmms_eval.models.qwen2_vl import Qwen2_VL

    @MODEL_REGISTRY
    class Qwen25_V4VLEval(Qwen2_VL):
        def __init__(
            self,
            llmc_model,
            pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
            device: Optional[str] = "cuda",
            device_map: Optional[str] = "cuda",
            batch_size: Optional[Union[int, str]] = 1,
            use_cache=True,
            use_flash_attention_2: Optional[bool] = False,
            max_pixels: int = 12845056,
            min_pixels: int = 3136,
            max_num_frames: int = 32,
            **kwargs,
        ) -> None:
            lmms.__init__(self)
            # Do not use kwargs for now
            assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

            accelerator = Accelerator()
            if accelerator.num_processes > 1:
                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                self.device_map = f"cuda:{accelerator.local_process_index}"
            elif accelerator.num_processes == 1 and device_map == "auto":
                self._device = torch.device(device)
                self.device_map = device_map
            else:
                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                self.device_map = f"cuda:{accelerator.local_process_index}"

            self._model = llmc_model.eval().cuda()
            self.processor = AutoProcessor.from_pretrained(
                pretrained, max_pixels=max_pixels, min_pixels=min_pixels
            )
            self.max_pixels = max_pixels
            self.min_pixels = min_pixels
            self.max_num_frames = max_num_frames
            self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

            self._config = self.model.config
            self.batch_size_per_gpu = int(batch_size)
            self.use_cache = use_cache

            if accelerator.num_processes > 1:
                assert accelerator.distributed_type in [
                    DistributedType.FSDP,
                    DistributedType.MULTI_GPU,
                ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
                if accelerator.distributed_type == DistributedType.FSDP:
                    self._model = accelerator.prepare(self.model)
                else:
                    self._model = accelerator.prepare_model(
                        self.model, evaluation_mode=True
                    )
                self.accelerator = accelerator
                if self.accelerator.is_local_main_process:
                    logger.info(
                        f"Using {accelerator.num_processes} devices with data parallelism"
                    )
                self._rank = self.accelerator.local_process_index
                self._world_size = self.accelerator.num_processes
            else:
                self._rank = 0
                self._world_size = 1

except Exception:
    logger.warning(
        "Can not import lmms_eval. " "If you need it, please upgrade transformers."
    )
