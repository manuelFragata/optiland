import numpy as np
import torch


def to_numpy(x):
    """Converts input scalar or array to NumPy array, regardless of backend."""
    if isinstance(x, (int, float, np.ndarray)):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        raise TypeError(f'Unsupported type for conversion to '
                        f'NumPy: {type(x)}')
