MEASUREMENT = "snr"
ROTATE_DEV = None # None or 'cpu'
QK_USE_FLOAT = True
USE_COMPILE = False
ATTN_IMPL = "sdpa" #"sdpa"  # eager # flash_attention_2

def set_rotate_device(device):
    global ROTATE_DEV
    ROTATE_DEV = device

def set_attn_impl(attn_impl):
    global ATTN_IMPL
    ATTN_IMPL = attn_impl

# def get_rotate_device():
#     global _ROTATE_DEV