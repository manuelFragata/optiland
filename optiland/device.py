import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_device(device_name: str):
    """
    Set the global device to 'cpu' or 'cuda'.

    Args:
        device_name: The device name. Choose from 'cpu' or 'cuda'.
    """
    global device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this system.")
    device = torch.device(device_name)


def get_device():
    """
    Get the current global device.
    """
    return device
