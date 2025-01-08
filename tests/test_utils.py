import numpy as np
from optiland.backend import to_numpy


def assert_allclose(a, b, atol=1e-6, rtol=1e-5):
    """Assert that two arrays or tensors are element-wise equal within
    tolerance.
    """
    a = to_numpy(a)
    b = to_numpy(b)
    assert np.allclose(a, b, atol=atol, rtol=rtol)
