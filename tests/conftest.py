import os

import jax
import pytest
from flax import nnx

from model import NanoLLM


def pytest_configure():
    """Set up JAX to simulate multiple devices for testing."""
    if "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


@pytest.fixture(scope="session")
def jax_devices():
    devices = jax.devices()
    assert len(devices) == 8, f"Expected 8 devices, got {len(devices)}"
    return devices


@pytest.fixture
def small_model():
    """Create a small test model."""
    return NanoLLM(
        rngs=nnx.Rngs(0),
        vocab_size=100,
        num_layers=2,
        num_heads=4,
        head_size=16,
        dropout_rate=0.1,
        embed_size=64,
        sequence_length=16,
    )
