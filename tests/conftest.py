import pytest
import numpy as np
import optiland.backend as be


# Tolerance for comparing values in tests
_atol = 1e-9
_rtol = 1e-8


def assert_allclose(a, b, atol=_atol, rtol=_rtol):
    """Assert that two arrays or tensors are element-wise equal within
    tolerance.
    """
    a = be.to_numpy(a)
    b = be.to_numpy(b)
    assert np.allclose(a, b, atol=atol, rtol=rtol)


@pytest.fixture(
    params=be.list_available_backends(), ids=lambda b: f'backend={b}'
)
def backend(request):
    """Fixture to set the backend for each test and ensure proper device
    configuration.
    """
    global _atol, _rtol

    backend_name = request.param
    be.set_backend(backend_name)

    if backend_name == 'torch':
        # torch uses 32-bit floats by default,
        # so we need to adjust the tolerance
        _atol = 1e-6
        _rtol = 1e-5
        be.set_device('cpu')  # Use CPU for tests
        be.grad_mode.disable()  # Disable gradient tracking
    yield

    # Reset the backend to numpy after the test
    be.set_backend('numpy')
    _atol = 1e-9
    _rtol = 1e-8


def pytest_configure(config):
    """
    Add markers dynamically to indicate tests for specific backends.
    """
    config.addinivalue_line(
        "markers", "backend: mark tests that use different backends"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify collected tests to skip tests for unavailable backends.
    """
    available_backends = be.list_available_backends()
    for item in items:
        backend_marker = item.get_closest_marker("backend")
        if backend_marker:
            backend_name = backend_marker.args[0]
            if backend_name not in available_backends:
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{backend_name} backend not available"
                    )
                )
