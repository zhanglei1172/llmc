MEASUREMENT = "snr"
ROTATE_DEV = None # None or 'cpu'

def set_rotate_device(device):
    global ROTATE_DEV
    ROTATE_DEV = device

# def get_rotate_device():
#     global _ROTATE_DEV