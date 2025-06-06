import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    # batch size for training
    batch_size: int = 128
    # sequence length
    sequence_length: int = 64
    # number of epochs for training
    num_epochs: int = 1
    # learning rate
    learning_rate: float = 1e-3
    # random seed
    seed: int = 42
    # number of iterations
    n_iterations: int = 10_000
    # frequency of updating metrics
    n_freq_train: int = 100
    # dropout rate
    dropout_rate: float = 0.2
    # number of layers
    num_layers: int = 6
    # embedding size
    embed_size: int = 256
    # number of heads
    num_heads: int = 8
    # head size
    head_size: int = 32
    # whether to use weights and biases
    use_wandb: bool = False
    # weights and biases project
    wandb_project: str = "nanollm"
    # weights and biases entity
    wandb_entity: str | None = None

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def get_config():
    """Get the default hyperparameter configuration."""
    config = Config()
    return config
