import dataclasses
from typing import Tuple


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
    """Sharding configuration for NanoLLM transformer."""

    # Embedding sharding: (vocab_size, embed_size)
    emb_vd: Tuple[str | None, ...]

    # Attention weights: (embed_size, embed_size) for QKV projection
    attn_weight_dd: Tuple[str | None, ...]

    # Linear input weights: (in_features, out_features)
    linear_in_df: Tuple[str | None, ...]

    # Linear output weights: (out_features, in_features)
    linear_out_fd: Tuple[str | None, ...]

    # LayerNorm weights: (embed_size,)
    layer_norm_d: Tuple[str | None, ...]

    # Activations: (batch, seq_len, embed_size)
    act_btd: Tuple[str | None, ...]

    # Activations: (batch, seq_len, hidden_size)
    act_btf: Tuple[str | None, ...]

    @staticmethod
    def get_default_sharding(is_sampling: bool = False):
        """Get default sharding configuration.

        Args:
            is_sampling: Whether this is for sampling/inference mode.
                        If True, removes data parallelism for better performance.
        """
        fsdp = "fsdp" if not is_sampling else None

        return ShardingConfig(
            emb_vd=("tp", fsdp),
            attn_weight_dd=("tp", fsdp),
            linear_in_df=(fsdp, "tp"),
            linear_out_fd=("tp", fsdp),
            layer_norm_d=("tp",),
            act_btd=(fsdp, None, "tp"),
            act_btf=(fsdp, None, "tp"),
        )

    @staticmethod
    def get_minimal_sharding():
        """
        Get minimal sharding configuration for small models or single-device training
        """
        return ShardingConfig(
            emb_vd=(None, None),
            attn_weight_dd=(None, None),
            linear_in_df=(None, None),
            linear_out_fd=(None, None),
            layer_norm_d=(None,),
            act_btd=(None, None, None),
            act_btf=(None, None, None),
        )
