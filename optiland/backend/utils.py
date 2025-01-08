"""
Utility functions for working with different backends.

To add support for a new backend, add a conversion function to the CONVERTERS
list.

Kramer Harrison, 2024
"""
import numpy as np
import importlib


# Conversion functions for backends
def torch_to_numpy(obj):
    if importlib.util.find_spec("torch"):
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    raise TypeError


CONVERTERS = [torch_to_numpy]


def to_numpy(obj):
    """Converts input scalar or array to NumPy array, regardless of backend."""
    if isinstance(obj, (int, float, np.ndarray)):
        return obj

    for converter in CONVERTERS:
        try:
            return converter(obj)
        except TypeError:
            continue
    raise TypeError(f"Unsupported object type: {type(obj)}")
