import numpy as np
import torch
from optiland.backend.device import get_device


# Link to the underlying library
_lib = torch


def array(x):
    """Create an array/tensor."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, device=get_device(), dtype=torch.float32,
                        requires_grad=True)


def zeros(shape):
    """Create an array/tensor filled with zeros."""
    return torch.zeros(shape, device=get_device(), dtype=torch.float32,
                       requires_grad=True)


def zeros_like(x):
    """Create an array/tensor filled with zeros with the same shape as x."""
    return torch.zeros_like(x, device=get_device(), dtype=torch.float32,
                            requires_grad=True)


def ones(shape):
    """Create an array/tensor filled with ones."""
    return torch.ones(shape, device=get_device(), dtype=torch.float32,
                      requires_grad=True)


def ones_like(x):
    """Create an array/tensor filled with ones with the same shape as x."""
    return torch.ones_like(x, device=get_device(), dtype=torch.float32,
                           requires_grad=True)


def full(shape, fill_value):
    """Create an array/tensor filled with fill_value."""
    return torch.full(shape, fill_value, device=get_device(),
                      dtype=torch.float32, requires_grad=True)


def full_like(x, fill_value):
    """
    Create an array/tensor filled with fill_value with the same shape as x.
    """
    if isinstance(fill_value, torch.Tensor):
        fill_value = fill_value.item()
    return torch.full_like(x, fill_value, device=get_device(),
                           dtype=torch.float32, requires_grad=True)


def linspace(start, stop, num=50):
    """Create an array/tensor of evenly spaced values."""
    return torch.linspace(start, stop, num, device=get_device(),
                          dtype=torch.float32, requires_grad=True)


def from_matrix(matrix):
    raise NotImplementedError('from_matrix is not implemented for torch. '
                              'Please use the NumPy backend.')


def from_euler(euler):
    raise NotImplementedError('from_euler is not implemented for torch.')


def copy(x):
    return x.clone()


def polyfit(x, y, degree):
    X = torch.stack([x**i for i in range(degree + 1)], dim=1)
    coeffs, _ = torch.lstsq(y.unsqueeze(1), X)
    return coeffs[:degree + 1].squeeze()


def polyval(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))


def load(filename):
    array = np.load(filename)
    return torch.from_numpy(array)


def hstack(arrays):
    return torch.cat(arrays, dim=1)


def vstack(arrays):
    return torch.cat(arrays, dim=0)


def interp(x, xp, fp):
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


def atleast_1d(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 0:  # Scalar -> (1,)
        return x.unsqueeze(0)
    return x  # Already 1D or higher


def atleast_2d(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 0:  # Scalar -> (1, 1)
        return x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 1:  # 1D array -> (1, N)
        return x.unsqueeze(0)
    return x  # Already 2D or higher


def size(x):
    return torch.numel(x)


def default_rng(seed=None):
    if seed is None:
        seed = torch.initial_seed()
    return torch.Generator(device=get_device()).manual_seed(seed)


def random_uniform(low=0.0, high=1.0, size=None, generator=None):
    if generator is None:
        return torch.empty(size, device=get_device()).uniform_(low, high)
    else:
        return torch.empty(size, device=get_device()).uniform_(
            low, high, generator=generator
        )


def repeat(x, repeats):
    return torch.repeat_interleave(x, repeats)


def flip(x):
    return torch.flip(x, dims=(0,))


def meshgrid(*arrays):
    return torch.meshgrid(*arrays, indexing='ij')
