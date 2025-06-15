import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from configs.sharding import ShardingConfig
from model import NanoLLM


@pytest.fixture
def mesh(jax_devices):
    return Mesh(devices=np.array(jax_devices).reshape(2, 4), axis_names=("fsdp", "tp"))


@pytest.fixture
def sharded_model(mesh):
    @nnx.jit
    def create_sharded_model():
        shd_config = ShardingConfig.get_default_sharding()
        model = NanoLLM(
            rngs=nnx.Rngs(0),
            vocab_size=100,
            num_layers=2,
            num_heads=4,
            head_size=16,
            embed_size=64,
            sequence_length=32,
            shd_config=shd_config,
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
    data_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))
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
    assert mesh.axis_names == ("fsdp", "tp"), (
        f"Expected axis names ('fsdp', 'tp'), got {mesh.axis_names}"
    )


@pytest.mark.sharding
def test_sharding_config_creation():
    """Test that ShardingConfig can be created with different options."""
    # Test default sharding
    default_config = ShardingConfig.get_default_sharding()
    assert default_config.emb_vd == ("tp", "fsdp")
    assert default_config.attn_weight_dd == ("tp", "fsdp")
    assert default_config.linear_in_df == ("fsdp", "tp")
    assert default_config.linear_out_fd == ("tp", "fsdp")
    assert default_config.layer_norm_d == ("tp",)
    assert default_config.act_btd == ("fsdp", None, "tp")
    assert default_config.act_btf == ("fsdp", None, "tp")

    # Test sampling sharding (no FSDP)
    sampling_config = ShardingConfig.get_default_sharding(is_sampling=True)
    assert sampling_config.emb_vd == ("tp", None)
    assert sampling_config.attn_weight_dd == ("tp", None)
    assert sampling_config.linear_in_df == (None, "tp")
    assert sampling_config.linear_out_fd == ("tp", None)
    assert sampling_config.layer_norm_d == ("tp",)
    assert sampling_config.act_btd == (None, None, "tp")
    assert sampling_config.act_btf == (None, None, "tp")

    # Test minimal sharding
    minimal_config = ShardingConfig.get_minimal_sharding()
    assert minimal_config.emb_vd == (None, None)
    assert minimal_config.attn_weight_dd == (None, None)
    assert minimal_config.linear_in_df == (None, None)
    assert minimal_config.linear_out_fd == (None, None)
    assert minimal_config.layer_norm_d == (None,)
    assert minimal_config.act_btd == (None, None, None)
    assert minimal_config.act_btf == (None, None, None)


@pytest.mark.sharding
def test_model_with_minimal_sharding():
    """Test that model works with minimal sharding configuration."""
    minimal_config = ShardingConfig.get_minimal_sharding()
    model = NanoLLM(
        rngs=nnx.Rngs(0),
        vocab_size=100,
        num_layers=1,
        num_heads=2,
        head_size=8,
        embed_size=32,
        sequence_length=16,
        shd_config=minimal_config,
    )

    # Test forward pass
    input_data = jnp.ones((2, 16), dtype=jnp.int32)
    output = model(input_data)
    assert output.shape == (2, 16, 100)
