import soundfile as sf

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeRMSNorm
from qwen_omni_utils import process_mm_info
import torch.nn as nn

# MODEL_PATH = "/code/Qwen3-Omni-30B-A3B-Instruct/"
# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
MODEL_PATH = "/code/lim42@xiaopeng.com/qwen3omini_vision_quarot/transformed_model/"
# MODEL_PATH = "/code/lim42@xiaopeng.com/qwen3omini_vision_quarot_eye/transformed_model/"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)


# ============ 替换所有 visual LayerNorm 为 RMSNorm ============
def replace_layernorm_with_rmsnorm(module, attr_name):
    """替换指定属性的 LayerNorm 为 RMSNorm"""
    if hasattr(module, attr_name):
        old_norm = getattr(module, attr_name)
        if isinstance(old_norm, nn.LayerNorm):
            new_norm = Qwen3OmniMoeRMSNorm(
                hidden_size=old_norm.normalized_shape[0],
                eps=old_norm.eps
            ).to(old_norm.weight.device, old_norm.weight.dtype)
            # 复制权重
            new_norm.weight.data.copy_(old_norm.weight.data)
            setattr(module, attr_name, new_norm)
            return True
    return False

# 1. 替换 visual blocks 中的 norm1 和 norm2
for block in model.thinker.visual.blocks:
    replace_layernorm_with_rmsnorm(block, 'norm1')
    replace_layernorm_with_rmsnorm(block, 'norm2')

# 2. 替换 merger_list 中的 ln_q
for merger in model.thinker.visual.merger_list:
    replace_layernorm_with_rmsnorm(merger, 'ln_q')

# 3. 替换 merger 中的 ln_q
replace_layernorm_with_rmsnorm(model.thinker.visual.merger, 'ln_q')

conversation = [
    {
        "role": "user",
        "content": [
            # {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"}
            {"type": "image", "image": "/dataset/workspace/lim42/cars.jpg"},
            # {"type": "audio", "audio": "/dataset/workspace/lim42/audio.wav"},
            {"type": "text", "text": "Tell me what you see?"},
        ],
    },
]

# Set whether to use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, 
                                 speaker="Ethan", 
                                 thinker_return_dict_in_generate=True,
                                 use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                              skip_special_tokens=True,
                              clean_up_tokenization_spaces=False)
print(text)
if audio is not None:
    sf.write(
        "output.wav",
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )
