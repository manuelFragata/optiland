import numpy as np
import torch


def assert_allclose(a, b, atol=1e-6, rtol=1e-5):
    """Assert that two arrays or tensors are element-wise equal within
    tolerance.
    """
    # Convert inputs to NumPy arrays
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    assert np.allclose(a, b, atol=atol, rtol=rtol)
