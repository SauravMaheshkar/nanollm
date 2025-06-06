import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx


class MLP(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        embed_size: int,
        dropout_rate: float,
    ) -> None:
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate

        # ==== Layers ====
        self.input_linear = nnx.Linear(
            in_features=embed_size,
            out_features=4 * embed_size,
            rngs=rngs,
        )
        self.output_linear = nnx.Linear(
            in_features=4 * embed_size,
            out_features=embed_size,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.input_linear(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        x = self.output_linear(x)
        return x


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_heads: int,
        head_size: int,
        dropout_rate: float,
        embed_size: int,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size

        # ==== Attention Layer ====
        self.attn = nnx.MultiHeadAttention(
            num_heads=self.num_heads,
            in_features=embed_size,
            qkv_features=embed_size,
            out_features=embed_size,
            dropout_rate=self.dropout_rate,
            rngs=rngs,
        )

        # ==== Normalization Layers ====
        self.pre_norm = nnx.LayerNorm(num_features=embed_size, use_bias=False, rngs=rngs)
        self.post_norm = nnx.LayerNorm(num_features=embed_size, use_bias=False, rngs=rngs)

        # ==== MLP Layer ====
        self.mlp = MLP(
            rngs=rngs,
            embed_size=embed_size,
            dropout_rate=dropout_rate,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_len = x.shape[:2]

        # Create causal attention mask
        # Shape: (seq_len, seq_len)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        mask = jnp.expand_dims(mask, axis=(0, 1))
        mask = jnp.broadcast_to(mask, (batch_size, self.num_heads, seq_len, seq_len))

        x_norm = self.pre_norm(x)
        attn_outputs = self.attn(x_norm, x_norm, mask=mask, decode=False)
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
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.sequence_length = sequence_length

        # ==== Embedding Layers ====
        self.token_emb = nnx.Embed(
            num_embeddings=self.vocab_size, features=self.embed_size, rngs=rngs
        )
        self.pos_emb = nnx.Embed(
            num_embeddings=self.sequence_length, features=self.embed_size, rngs=rngs
        )

        # ==== Transformer Blocks ====
        self.blocks = [
            TransformerBlock(
                rngs=rngs,
                num_heads=self.num_heads,
                head_size=self.head_size,
                dropout_rate=self.dropout_rate,
                embed_size=self.embed_size,
            )
            for _ in range(self.num_layers)
        ]

        # ==== Output Layer ====
        self.output_layer = nnx.Linear(
            in_features=self.embed_size, out_features=self.vocab_size, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)

        # ==== Embeddings ====
        position_emb = self.pos_emb(positions)
        token_emb = self.token_emb(x)
        x = token_emb + position_emb

        # ==== Transformer Blocks ====
        for block in self.blocks:
            x = block(x)

        # ==== Output Layer ====
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

    def save(self, path: str) -> None:
        """Saves the model state to a directory.

        Args:
            path: The directory path to save the model state to.
        """
        state = nnx.state(self)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(f"{path}/nanollm", state)

    def generate(self):
        """
        TODO(@saurav): Implement method to generate text.
        """
        pass
