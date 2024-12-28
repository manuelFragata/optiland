import torch


_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_device(device: str):
    global _device
    _device = device


def get_device():
    return _device
