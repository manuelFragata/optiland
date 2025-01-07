import pytest
import optiland.backend as be


@pytest.fixture(
    params=be.list_available_backends(), ids=lambda b: f'backend={b}'
)
def backend(request):
    """Fixture to set the backend for each test and ensure proper device
    configuration.
    """
    backend_name = request.param
    be.set_backend(backend_name)
    if backend_name == 'torch':
        be.set_device('cpu')  # Ensure torch uses the CPU for testing
    return backend_name


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
