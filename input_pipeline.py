import datasets
import jax.numpy as jnp


def get_datasets() -> tuple[jnp.ndarray, jnp.ndarray, int]:
    ds = datasets.load_dataset(
        "karpathy/tiny_shakespeare",
        split=["train", "test", "validation"],
        trust_remote_code=True,
    )
    train_ds, test_ds, val_ds = ds

    train_text = train_ds[0]["text"]
    vocab = sorted(set(train_text))

    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    def encode(sentence: str) -> list[int]:
        return [stoi[c] for c in sentence]

    def decode(latents: list[int]) -> str:
        return "".join([itos[i] for i in latents])

    train_ds = datasets.concatenate_datasets([train_ds, test_ds])

    def encode_fn(example):
        return {"text": encode(example["text"])}

    train_ds = train_ds.map(encode_fn)
    val_ds = val_ds.map(encode_fn)

    train_ds = train_ds.with_format("jax")
    val_ds = val_ds.with_format("jax")

    train_data = train_ds["text"][0]
    val_data = val_ds["text"][0]

    vocab_size = len(vocab)

    return train_data, val_data, vocab_size
