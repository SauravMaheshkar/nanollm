import inspect
import json
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from huggingface_hub import HfApi, hf_hub_download
from jaxtyping import Array, Float


class MLP(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        embed_size: int,
        dropout_rate: float,
        sharding_rules: Optional[Tuple[Tuple[str, str], ...]] = None,
    ) -> None:
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate
        self.sharding_rules = sharding_rules

        # ==== Layers ====
        # Shard the input linear layer: (embed_size, 4*embed_size)
        # Use (None, 'model') to shard across model dimension
        self.input_linear = nnx.Linear(
            in_features=embed_size,
            out_features=4 * embed_size,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "model")
            ),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model",)),
            rngs=rngs,
        )

        # Shard the output linear layer: (4*embed_size, embed_size)
        # Use ('model', None) to shard across model dimension
        self.output_linear = nnx.Linear(
            in_features=4 * embed_size,
            out_features=embed_size,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), ("model", None)
            ),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(
        self, x: Float[Array, "batch seq_len embed_size"]
    ) -> Float[Array, "batch seq_len embed_size"]:
        x = self.input_linear(x)  # shape: (batch, seq_len, 4 * embed_size)
        x = nnx.relu(x)
        x = self.dropout(x)
        x = self.output_linear(x)  # shape: (batch, seq_len, embed_size)
        return x


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_heads: int,
        head_size: int,
        dropout_rate: float,
        embed_size: int,
        sharding_rules: Optional[Tuple[Tuple[str, str], ...]] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.sharding_rules = sharding_rules

        # ==== Attention Layer ====
        # Shard attention weights across model dimension
        self.attn = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=embed_size,
            qkv_features=embed_size,
            out_features=embed_size,
            dropout_rate=self.dropout_rate,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), (None, "model")
            ),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("model",)),
            rngs=rngs,
        )

        # ==== Normalization Layers ====
        # LayerNorm parameters are typically not sharded
        self.pre_norm = nnx.LayerNorm(num_features=embed_size, use_bias=False, rngs=rngs)
        self.post_norm = nnx.LayerNorm(num_features=embed_size, use_bias=False, rngs=rngs)

        # ==== MLP Layer ====
        self.mlp = MLP(
            rngs=rngs,
            embed_size=embed_size,
            dropout_rate=dropout_rate,
            sharding_rules=sharding_rules,
        )

    def __call__(
        self, x: Float[Array, "batch seq_len embed_size"]
    ) -> Float[Array, "batch seq_len embed_size"]:
        batch_size, seq_len = x.shape[:2]

        # Create causal attention mask
        # shape: (seq_len, seq_len)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = jnp.expand_dims(mask, axis=(0, 1))
        mask = jnp.broadcast_to(mask, (batch_size, self.num_heads, seq_len, seq_len))
        # shape: (batch_size, num_heads, seq_len, seq_len)

        x_norm = self.pre_norm(x)
        attn_outputs = self.attn(
            x_norm, x_norm, mask=mask, decode=False, deterministic=False
        )
        x = x + attn_outputs

        x = self.post_norm(x)
        x = x + self.mlp(x)
        return x


class NanoLLM(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        vocab_size: int,
        num_layers: int = 6,
        num_heads: int = 8,
        head_size: int = 32,
        dropout_rate: float = 0.2,
        embed_size: int = 256,
        sequence_length: int = 64,
        sharding_rules: Optional[Tuple[Tuple[str, str], ...]] = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.sequence_length = sequence_length
        self.sharding_rules = sharding_rules

        # ==== Embedding Layers ====
        self.token_emb = nnx.Embed(
            num_embeddings=self.vocab_size,
            features=self.embed_size,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), (None, "model")
            ),
            rngs=rngs,
        )
        self.pos_emb = nnx.Embed(
            num_embeddings=self.sequence_length,
            features=self.embed_size,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), (None, None)
            ),
            rngs=rngs,
        )

        # ==== Transformer Blocks ====
        self.blocks = [
            TransformerBlock(
                rngs=rngs,
                num_heads=self.num_heads,
                head_size=self.head_size,
                dropout_rate=self.dropout_rate,
                embed_size=self.embed_size,
                sharding_rules=sharding_rules,
            )
            for _ in range(self.num_layers)
        ]

        # ==== Output Layer ====
        # Output layer: (embed_size, vocab_size) - shard across model dimension
        self.output_layer = nnx.Linear(
            in_features=self.embed_size,
            out_features=self.vocab_size,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), ("model", None)
            ),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
            rngs=rngs,
        )

    def __call__(
        self, x: Float[Array, "batch seq_len"]
    ) -> Float[Array, "batch seq_len vocab_size"]:
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)  # shape: (seq_len, )

        # ==== Embeddings ====
        position_emb = self.pos_emb(positions)  # shape: (seq_len, embed_size)
        token_emb = self.token_emb(x)  # shape: (batch, seq_len, embed_size)
        x = token_emb + position_emb

        # ==== Transformer Blocks ====
        for block in self.blocks:
            x = block(x)

        # ==== Output Layer ====
        # shape: (batch, seq_len, vocab_size)
        return self.output_layer(x)

    @property
    def state(self) -> nnx.State:
        """Splits state from the graph and returns it"""
        return nnx.split(self, nnx.Param, ...)[1]

    @property
    def state_dict(self) -> dict[str, jnp.ndarray]:
        """Splits state from the graph and returns it as a dictionary.

        It can be used for serialization with orbax."""
        state = self.state
        pure_dict_state = nnx.to_pure_dict(state)
        return pure_dict_state

    @property
    def num_params(self) -> int:
        return sum(p.size for p in jax.tree.leaves(self.state))

    def save(self, path: str, **kwargs) -> None:
        """Saves the model state to a directory.

        Args:
            path: The directory path to save the model state to.
        """
        state = nnx.state(self)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(f"{path}/nanollm", state, **kwargs)

    def push_to_wandb(
        self, artifact_name: str, save_path: str, metadata: dict[str, Any]
    ) -> None:
        """Pushes the model ckpt to wandb.

        Args:
            artifact_name: The name of the artifact to push to wandb.
            save_path: The path to save the model ckpt to.
            metadata: The metadata to push to wandb.
        """
        import wandb

        if not wandb.run:
            wandb.init(project="nanollm", job_type="upload-model")

        metadata = {**metadata, "vocab_size": self.vocab_size}

        artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
        self.save(
            path=save_path,
            force=True,
            custom_metadata=metadata,
        )
        artifact.add_dir(save_path)
        wandb.log_artifact(artifact)

    def push_to_hub(
        self,
        repo_id: str,
        save_path: str,
        metadata: dict[str, Any],
        commit_message: str = "Upload model",
        token: Optional[str] = None,
    ) -> None:
        """Pushes the model and config to the Hugging Face Hub.

        Args:
            repo_id: The repository ID on the Hugging Face Hub
                (e.g. "SauravMaheshkar/nanollm")
            save_path: Local path to temporarily save the model
            metadata: Additional metadata to include in the config
            commit_message: Message for the commit
            token: Hugging Face API token.
        """
        self.save(path=save_path, force=True)

        config = {**metadata, "vocab_size": self.vocab_size}

        config_path = f"{save_path}/config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        api = HfApi(token=token)

        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"Note: Could not create repo: {e}")

        api.upload_folder(
            folder_path=save_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

    def load(self, path: str) -> "NanoLLM":
        """Loads the model state from a directory.

        Args:
            path: The directory path to load the model state from.
        """
        checkpointer = ocp.PyTreeCheckpointer()
        state = checkpointer.restore(f"{path}/nanollm", item=nnx.state(self))
        nnx.update(self, state)
        return self

    @classmethod
    def from_wandb_artifact(cls, artifact_name: str) -> "NanoLLM":
        """Loads a model ckpt from a wandb artifact.

        Args:
            artifact_name: The name of the artifact to load the model ckpt from.

        Example:
            ```python
            model = NanoLLM.from_wandb_artifact("sauravmaheshkar/nanollm/nanollm:v0")
            ```
        """
        import wandb

        api = wandb.Api()
        artifact = api.artifact(artifact_name)
        artifact_dir = artifact.download()

        init_params = inspect.signature(cls.__init__).parameters.keys()
        config = {k: v for k, v in {**artifact.metadata}.items() if k in init_params}

        rngs = nnx.Rngs(0)
        dummy_model = cls(rngs=rngs, **config)

        checkpointer = ocp.PyTreeCheckpointer()
        state = checkpointer.restore(
            f"{artifact_dir}/nanollm", item=nnx.state(dummy_model)
        )
        nnx.update(dummy_model, state)
        return dummy_model

    @classmethod
    def load_pretrained(
        cls,
        repo_id: str,
        save_path: Optional[str] = "artifacts",
        token: Optional[str] = None,
    ) -> "NanoLLM":
        """Loads a pretrained model from the Hugging Face Hub.

        Args:
            repo_id: The repository ID on the Hugging Face Hub
                (e.g. "SauravMaheshkar/nanollm")
            save_path: Local path to save the downloaded model
            token: Hugging Face API token.

        Example:
        ```python
        model = NanoLLM.load_pretrained("SauravMaheshkar/nanollm")
        ```
        """
        import os

        save_path = os.path.abspath(save_path)
        os.makedirs(save_path, exist_ok=True)

        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model",
            token=token,
        )
        with open(config_path, "r") as f:
            config = json.load(f)

        init_params = inspect.signature(cls.__init__).parameters.keys()
        config = {k: v for k, v in config.items() if k in init_params}

        api = HfApi(token=token)
        api.snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=save_path,
            token=token,
        )

        rngs = nnx.Rngs(0)
        dummy_model = cls(rngs=rngs, **config)

        checkpointer = ocp.PyTreeCheckpointer()
        state = checkpointer.restore(f"{save_path}/nanollm", item=nnx.state(dummy_model))
        nnx.update(dummy_model, state)
        return dummy_model

    def generate(self):
        """
        TODO(@saurav): Implement method to generate text.
        """
        pass
