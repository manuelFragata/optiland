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


def __getattr__(name):
    """Dynamically forward attribute requests to the current backend.

    This is used to access constants and functions from the current backend
    without importing them explicitly. For example, `be.sin` will return
    `np.sin` or `torch.sin` based on the current backend. Likewise, `be.pi`
    will return `np.pi` or `torch.pi`.
    """
    if hasattr(_current_backend, name):
        return getattr(_current_backend, name)
    raise AttributeError(f'"{_current_backend.__name__}" backend has no '
                         f'attribute "{name}"')


# Functions for common operations
def array(x):
    """Create an array/tensor."""
    if _current_backend == torch:
        return torch.tensor(x, device=get_device(), dtype=torch.float32)
    return np.asarray(x, dtype=float)


def zeros(shape):
    """Create an array/tensor filled with zeros."""
    if _current_backend == torch:
        return torch.zeros(shape, device=get_device(), dtype=torch.float32)
    return np.zeros(shape)


def zeros_like(x):
    """Create an array/tensor filled with zeros with the same shape as x."""
    if _current_backend == torch:
        return torch.zeros_like(x, device=get_device(), dtype=torch.float32)
    return np.zeros_like(x)


def ones(shape):
    """Create an array/tensor filled with ones."""
    if _current_backend == torch:
        return torch.ones(shape, device=get_device(), dtype=torch.float32)
    return np.ones(shape)


def ones_like(x):
    """Create an array/tensor filled with ones with the same shape as x."""
    if _current_backend == torch:
        return torch.ones_like(x, device=get_device(), dtype=torch.float32)
    return np.ones_like(x)


def full_like(x, fill_value):
    """
    Create an array/tensor filled with fill_value with the same shape as x.
    """
    if _current_backend == torch:
        return torch.full_like(x, fill_value, device=get_device(),
                               dtype=torch.float32)
    return np.full_like(x, fill_value)


def from_matrix(matrix):
    if _current_backend == torch:
        raise NotImplementedError('from_matrix is not implemented for torch.')
    return R.from_matrix(matrix)


def from_euler(euler):
    if _current_backend == torch:
        raise NotImplementedError('from_euler is not implemented for torch.')
    return R.from_euler('xyz', euler)


def copy(x):
    if _current_backend == torch:
        return x.clone()
    return x.copy()
