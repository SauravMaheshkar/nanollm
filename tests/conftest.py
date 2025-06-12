import os

import jax
import pytest


def pytest_configure():
    """Set up JAX to simulate multiple devices for testing."""
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


@pytest.fixture(scope="session")
def jax_devices():
    devices = jax.devices()
    assert len(devices) == 8, f"Expected 8 devices, got {len(devices)}"
    return devices
