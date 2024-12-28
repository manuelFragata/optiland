"""Backend Module

This module provides a unified interface for performing numerical operations
using either NumPy or PyTorch as the backend. The default backend is NumPy,
but it can be switched to PyTorch using the `set_backend` function.

Kramer Harrison, 2024
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
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


def zeros(shape):
    """Create an array/tensor filled with zeros."""
    if _current_backend == torch:
        return torch.zeros(shape, device=get_device(), dtype=torch.float32)
    return np.zeros(shape)


def ones(shape):
    """Create an array/tensor filled with ones."""
    if _current_backend == torch:
        return torch.ones(shape, device=get_device(), dtype=torch.float32)
    return np.ones(shape)


def from_matrix(matrix):
    if _current_backend == torch:
        raise NotImplementedError('from_matrix is not implemented for torch.')
    return R.from_matrix(matrix)


def from_euler(euler):
    if _current_backend == torch:
        raise NotImplementedError('from_euler is not implemented for torch.')
    return R.from_euler('xyz', euler)


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
