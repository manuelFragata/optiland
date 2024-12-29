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
        if isinstance(x, torch.Tensor):
            return x
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


def full(shape, fill_value):
    """Create an array/tensor filled with fill_value."""
    if _current_backend == torch:
        return torch.full(shape, fill_value, device=get_device(),
                          dtype=torch.float32)
    return np.full(shape, fill_value)


def full_like(x, fill_value):
    """
    Create an array/tensor filled with fill_value with the same shape as x.
    """
    if _current_backend == torch:
        return torch.full_like(x, fill_value, device=get_device(),
                               dtype=torch.float32)
    return np.full_like(x, fill_value)


def linspace(start, stop, num=50):
    """Create an array/tensor of evenly spaced values."""
    if _current_backend == torch:
        return torch.linspace(start, stop, num, device=get_device(),
                              dtype=torch.float32)
    return np.linspace(start, stop, num)


def to_numpy(x):
    """Converts input scalar or array to NumPy array, regardless of backend."""
    if isinstance(x, (int, float, np.ndarray)):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        raise TypeError(f'Unsupported type for conversion to '
                        f'NumPy: {type(x)}')


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


def polyfit(x, y, degree):
    if _current_backend == torch:
        X = torch.stack([x**i for i in range(degree + 1)], dim=1)
        coeffs, _ = torch.lstsq(y.unsqueeze(1), X)
        return coeffs[:degree + 1].squeeze()
    return np.polyfit(x, y, degree)


def polyval(coeffs, x):
    if _current_backend == torch:
        return sum(c * x**i for i, c in enumerate(coeffs))
    return np.polyval(coeffs, x)


def load(filename):
    array = np.load(filename)
    if _current_backend == torch:
        array = torch.from_numpy(array)
    return array


def hstack(arrays):
    if _current_backend == torch:
        return torch.cat(arrays, dim=1)
    return np.hstack(arrays)


def vstack(arrays):
    if _current_backend == torch:
        return torch.cat(arrays, dim=0)
    return np.vstack(arrays)


def torch_interp(x, xp, fp):
    """
    Mimics numpy.interp for 1D linear interpolation in PyTorch.

    Args:
        x (torch.Tensor): Points to interpolate.
        xp (torch.Tensor): Known x-coordinates.
        fp (torch.Tensor): Known y-coordinates.

    Returns:
        torch.Tensor: Interpolated values.
    """
    # Ensure tensors are float for arithmetic operations
    x = torch.as_tensor(x, dtype=torch.float32, device=get_device())
    xp = torch.as_tensor(xp, dtype=torch.float32, device=get_device())
    fp = torch.as_tensor(fp, dtype=torch.float32, device=get_device())

    # Sort xp and fp based on xp
    sorted_indices = torch.argsort(xp)
    xp = xp[sorted_indices]
    fp = fp[sorted_indices]

    # Clip x to be within the range of xp
    x_clipped = torch.clip(x, xp[0], xp[-1])

    # Find indices where each x would be inserted to maintain order
    indices = torch.searchsorted(xp, x_clipped, right=True)
    indices = torch.clamp(indices, 1, len(xp) - 1)

    # Get the x-coordinates and y-coordinates for interpolation
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]

    # Linear interpolation formula
    interpolated = y0 + (y1 - y0) * (x_clipped - x0) / (x1 - x0)
    return interpolated


def interp(x, xp, fp):
    if _current_backend == torch:
        return torch_interp(x, xp, fp)
    return np.interp(x, xp, fp)


def atleast_1d(x):
    if _current_backend == torch:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 0:  # Scalar -> (1,)
            return x.unsqueeze(0)
        return x  # Already 1D or higher
    return np.atleast_1d(x)


def atleast_2d(x):
    if _current_backend == torch:
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 0:  # Scalar -> (1, 1)
            return x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 1:  # 1D array -> (1, N)
            return x.unsqueeze(0)
        return x  # Already 2D or higher
    return np.atleast_2d(x)


def size(x):
    if _current_backend == torch:
        return torch.numel(x)
    return x.size


def default_rng(seed=None):
    if _current_backend == torch:
        return torch.Generator(device=get_device()).manual_seed(seed)
    return np.random.default_rng(seed)


def random_uniform(low=0.0, high=1.0, size=None, generator=None):
    if _current_backend == torch:
        if generator is None:
            return torch.empty(size).uniform_(low, high)
        else:
            return torch.empty(size, generator=generator).uniform_(low, high)
    return np.random.uniform(low, high, size)
