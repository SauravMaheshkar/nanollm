import dataclasses
import os

import jax
import optax
from absl import logging
from flax import nnx

from configs import default
from input_pipeline import get_datasets
from model import NanoLLM


def loss_fn(
    model: nnx.Module, batch: tuple[jax.Array, jax.Array]
) -> tuple[float, jax.Array]:
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: tuple[jax.Array, jax.Array],
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, lables=batch[1])
    optimizer.update(grads)


def train_and_evaluate(config: default.Config, workdir: str) -> None:
    # Convert workdir to absolute path
    workdir = os.path.abspath(workdir)

    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=dataclasses.asdict(config),
        )

    train_data, _, vocab_size = get_datasets()

    dynamic_slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))

    def get_batch_fn(random_key, data, batch_size, sequence_length):
        ix = jax.random.randint(
            random_key,
            shape=(batch_size, 1),
            minval=0,
            maxval=len(data) - sequence_length,
        )
        x = dynamic_slice_vmap(data, ix, (sequence_length,))
        y = dynamic_slice_vmap(data, ix + 1, (sequence_length,))
        return x, y

    get_batch = jax.jit(get_batch_fn, static_argnames=["batch_size", "sequence_length"])

    model = NanoLLM(
        vocab_size=vocab_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_size=config.head_size,
        dropout_rate=config.dropout_rate,
        embed_size=config.embed_size,
        sequence_length=config.sequence_length,
        rngs=nnx.Rngs(0),
    )

    logging.info(f"Total number of parameters: {model.num_params:_}")
    if config.use_wandb:
        wandb.summary["num_params"] = model.num_params

    optimizer = nnx.Optimizer(model, optax.adamw(config.learning_rate, nesterov=True))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )
    rng = jax.random.PRNGKey(config.seed)
    metrics_history = {
        "train_loss": [],
    }

    model.train()
    for _ in range(config.num_epochs):
        for i in range(config.n_iterations):
            rng, subkey = jax.random.split(rng)
            batch = get_batch(
                subkey, train_data, config.batch_size, config.sequence_length
            )
            train_step(model, optimizer, metrics, batch)

            if (i + 1) % config.n_freq_train == 0:
                for metric, value in metrics.compute().items():
                    metrics_history[f"train_{metric}"].append(value)
                metrics.reset()

                # === logging ===
                print(f"Iteration {i + 1}, Loss: {metrics_history['train_loss'][-1]:.4f}")
                if config.use_wandb:
                    wandb.log(
                        {"train_loss": metrics_history["train_loss"][-1]}, step=i + 1
                    )

        model.save(workdir)
