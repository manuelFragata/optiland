import numpy as np
import optiland.backend as be


def assert_allclose(a, b):
    """Assert that two arrays or tensors are element-wise equal within
    tolerance.
    """
    backend = be.get_backend()
    if backend == 'torch':
        atol = 1e-4
    else:
        atol = 1e-7

    a = be.to_numpy(a)
    b = be.to_numpy(b)
    assert np.allclose(a, b, atol=atol)
