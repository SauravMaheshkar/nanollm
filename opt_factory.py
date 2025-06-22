from typing import Optional

import optax
from flax import nnx


def get_optimizer(
    model: nnx.Module, opt: Optional[str] = "adamw", **kwargs
) -> nnx.Optimizer:
    if opt == "adamw":
        base_optimizer = optax.adamw(**kwargs)
    elif opt == "muon":
        assert optax.__version__ >= "0.2.5", "Muon requires optax >= 0.2.5"
        base_optimizer = optax.contrib.muon(**kwargs)
    else:
        raise ValueError(f"Invalid optimizer: {opt}")

    return nnx.Optimizer(model, base_optimizer)
