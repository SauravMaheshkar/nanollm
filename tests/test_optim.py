import copy

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from flax.traverse_util import flatten_dict

from opt_factory import get_optimizer
from train import train_step


def create_synthetic_data(
    batch_size: int, sequence_length: int, vocab_size: int, rng_key
):
    """Create synthetic training data for testing."""
    rng_key, subkey = jax.random.split(rng_key)

    x = jax.random.randint(
        subkey, shape=(batch_size, sequence_length), minval=0, maxval=vocab_size
    )

    rng_key, subkey = jax.random.split(rng_key)
    y = jax.random.randint(
        subkey, shape=(batch_size, sequence_length), minval=0, maxval=vocab_size
    )

    return x, y


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_optimizer_training(optimizer_name, small_model):
    """Test that optimizers work and reduce loss during training."""
    optimizer = get_optimizer(small_model, opt=optimizer_name, learning_rate=0.001)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    rng = jax.random.PRNGKey(42)
    losses = []

    small_model.train()
    for _ in range(10):
        rng, subkey = jax.random.split(rng)
        batch = create_synthetic_data(4, 16, 100, subkey)
        train_step(small_model, optimizer, metrics, batch)
        current_loss = metrics.compute()["loss"]
        losses.append(float(current_loss))

        metrics.reset()

    # verify loss is decreasing
    assert losses[-1] < losses[0], (
        f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )

    # verify loss values are finite
    assert all(jnp.isfinite(loss) for loss in losses), "All loss values should be finite"

    # verify variation in loss values
    assert len(set(losses)) > 1, "Loss values should vary during training"


def test_invalid_optimizer(small_model):
    """Test that invalid optimizer raises ValueError."""
    with pytest.raises(ValueError, match="Invalid optimizer"):
        get_optimizer(small_model, opt="invalid_optimizer")


def test_model_parameters_update(small_model):
    """Test that model parameters are actually being updated during training."""
    initial_params = {k: v for k, v in flatten_dict(small_model.state_dict).items()}

    optimizer = get_optimizer(small_model, opt="adamw", learning_rate=0.001)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    rng = jax.random.PRNGKey(42)
    small_model.train()

    for _ in range(5):
        rng, subkey = jax.random.split(rng)
        batch = create_synthetic_data(4, 16, 100, subkey)
        train_step(small_model, optimizer, metrics, batch)
        metrics.reset()

    final_params = {k: v for k, v in flatten_dict(small_model.state_dict).items()}

    # verify some parameters have changed
    param_changed = False
    for key in initial_params:
        if not jnp.allclose(initial_params[key], final_params[key]):
            param_changed = True
            break

    assert param_changed, "Model parameters should be updated during training"


def test_optimizer_state_consistency(small_model):
    """Test that optimizer state is consistent and changes during training."""
    optimizer = get_optimizer(small_model, opt="adamw", learning_rate=0.001)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    initial_state = copy.deepcopy(optimizer.opt_state)

    rng = jax.random.PRNGKey(42)
    small_model.train()

    for step in range(3):
        rng, subkey = jax.random.split(rng)
        batch = create_synthetic_data(4, 16, 100, subkey)
        train_step(small_model, optimizer, metrics, batch)
        metrics.reset()

    # verify optimizer state has changed
    final_state = optimizer.opt_state

    def get_value(x):
        if hasattr(x, "value"):
            return x.value
        return x

    def compare(a, b):
        a_val, b_val = get_value(a), get_value(b)
        if isinstance(a_val, jnp.ndarray) and isinstance(b_val, jnp.ndarray):
            return jnp.allclose(a_val, b_val)
        return a_val == b_val

    changed = not jax.tree.all(jax.tree.map(compare, initial_state, final_state))
    assert changed, "Optimizer state should change during training"
