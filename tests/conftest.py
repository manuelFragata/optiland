import pytest
import optiland.backend as be


AVAILABLE_BACKENDS = ['numpy', 'torch']


@pytest.fixture(params=AVAILABLE_BACKENDS, ids=lambda b: f'backend={b}')
def backend(request):
    """Fixture to set the backend for each test."""
    backend_name = request.param
    be.set_backend(backend_name)
    return backend_name
