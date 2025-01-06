"""Backend Module

This module provides a unified interface for performing numerical operations
using either NumPy or PyTorch as the backend. The default backend is NumPy,
but it can be switched to PyTorch using the `set_backend` function.

Kramer Harrison, 2024
"""
from optiland.backend import (
    numpy_backend,
    torch_backend
)
from optiland.backend.utils import to_numpy  # noqa: F401
from optiland.backend.torch_backend import grad_mode  # noqa: F401


_backends = {
    'numpy': numpy_backend,
    'torch': torch_backend,
}
_current_backend = 'numpy'  # Default backend is numpy


def set_backend(name: str):
    """Set the current backend."""
    global _current_backend
    if name not in _backends:
        raise ValueError(f'Unknown backend "{name}". '
                         f'Available: {list(_backends.keys())}')
    _current_backend = name


def get_backend():
    """Get the current backend module."""
    return _current_backend


def __getattr__(name):
    """Dynamically retrieve attributes (functions) from the current backend.

    This is used to access constants and functions from the current backend
    without importing them explicitly. For example, `be.sin` will return
    `np.sin` or `torch.sin` based on the current backend. Likewise, `be.pi`
    will return `np.pi` or `torch.pi`.
    """
    backend = _backends[_current_backend]
    if hasattr(backend, name):
        return getattr(backend, name)

    elif hasattr(backend._lib, name):
        return getattr(backend._lib, name)

    raise AttributeError(
        f'"{backend.__name__}" backend has no attribute "{name}"'
    )
