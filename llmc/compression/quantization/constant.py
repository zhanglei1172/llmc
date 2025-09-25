MEASUREMENT = "snr"
ROTATE_DEV = None # None or 'cpu'
QK_USE_FLOAT = True
USE_COMPILE = False
ATTN_IMPL = "flash_attention_2" #"sdpa"  # eager # flash_attention_2

def set_rotate_device(device):
    global ROTATE_DEV
    ROTATE_DEV = device

# def get_rotate_device():
#     global _ROTATE_DEV