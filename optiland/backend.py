import numpy as np
import torch
from optiland.device import get_device


_backends = {'numpy': np, 'torch': torch}
_current_backend = _backends['numpy']  # Default backend is numpy


def set_backend(name: str):
    global _current_backend
    if name not in _backends:
        raise ValueError(f'Backend "{name}" is not supported. '
                         f'Choose from {list(_backends.keys())}')
    _current_backend = _backends[name]


def get_backend():
    return _current_backend


# Functions for common operations
def array(x):
    """Create an array/tensor."""
    if _current_backend == torch:
        return torch.tensor(x, device=get_device(), dtype=torch.float32)
    return np.asarray(x)


def sqrt(x):
    return _current_backend.sqrt(x)


def sin(x):
    return _current_backend.sin(x)


def cos(x):
    return _current_backend.cos(x)


def exp(x):
    return _current_backend.exp(x)


def mean(x):
    return _current_backend.mean(x)
