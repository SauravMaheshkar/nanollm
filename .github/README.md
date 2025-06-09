<a href="https://colab.research.google.com/github/SauravMaheshkar/nanollm/blob/main/notebooks/nanollm_tiny_shakespeare.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> [![CI/CD](https://github.com/SauravMaheshkar/nanollm/actions/workflows/ci.yml/badge.svg)](https://github.com/SauravMaheshkar/nanollm/actions/workflows/ci.yml)

JAX LLM playground

### Setup

```shell
uv sync --all-groups
```

### Getting started

```shell
python main.py --workdir=artifacts/
```

Log training data and model checkpoints to wandb

```shell
python main.py --workdir=artifacts/ \
    --config.use_wandb \
    --config.wandb_project=nanollm \
    --config.wandb_entity=sauravmaheshkar
```

Log model checkpoints to huggingface

```shell
python main.py --workdir=artifacts/ \
    --config.push_to_hub \
    --config.repo_id="SauravMaheshkar/nanollm"
```

### References

* https://github.com/google-deepmind/nanodo
* https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html
* https://optax.readthedocs.io/en/stable/_collections/examples/nanolm.html
