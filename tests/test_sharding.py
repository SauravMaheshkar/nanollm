import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from model import NanoLLM


@pytest.fixture
def mesh(jax_devices):
    return Mesh(devices=np.array(jax_devices).reshape(2, 4), axis_names=("data", "model"))


@pytest.fixture
def sharded_model(mesh):
    @nnx.jit
    def create_sharded_model():
        model = NanoLLM(
            rngs=nnx.Rngs(0),
            vocab_size=100,
            num_layers=2,
            num_heads=4,
            head_size=16,
            embed_size=64,
            sequence_length=32,
        )
        state = nnx.state(model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(model, sharded_state)
        return model

    with mesh:
        return create_sharded_model()


@pytest.mark.sharding
def test_basic_sharding(sharded_model):
    state = nnx.state(sharded_model)
    sharded_params = []

    for name, param in jax.tree.leaves_with_path(state):
        if hasattr(param, "sharding"):
            sharded_params.append(name)
            assert param.sharding is not None, f"Parameter {name} should have sharding"

    # Ensure we have at least some sharded parameters
    assert len(sharded_params) > 0, "No sharded parameters found"


@pytest.mark.sharding
def test_model_forward(mesh, sharded_model):
    data_sharding = NamedSharding(mesh, PartitionSpec("data", None))
    input_data = jax.device_put(jnp.ones((8, 32), dtype=jnp.int32), data_sharding)

    with mesh:
        output = sharded_model(input_data)

    assert output.shape == (8, 32, 100), (
        f"Expected shape (8, 32, 100), got {output.shape}"
    )
    assert hasattr(output, "sharding"), "Output should have sharding information"
    assert output.sharding is not None, "Output sharding should not be None"


@pytest.mark.sharding
def test_mesh_creation(mesh):
    assert mesh.axis_names == ("data", "model"), (
        f"Expected axis names ('data', 'model'), got {mesh.axis_names}"
    )
