# Pallas Puzzles

[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) is JAX's kernel language for writing custom operations that run on TPU — what [Triton](https://triton-lang.org/) is for GPUs. Kernel languages provide low-level access to hardware, allowing optimizations outside of the compiler's reach.

This repo contains progressive puzzle notebooks that build from Pallas basics towards real open-source kernels. All puzzles run on **free Google Colab CPU instances** via `interpret=True` — no TPU needed.

**Blog post**: [vorushin.github.io/blog/pallas-puzzles](https://vorushin.github.io/blog/pallas-puzzles)

## Learning paths

### From Pallas basics to SplashAttention

SplashAttention — SParse version of fLASH attention — is an efficient implementation of attention on TPUs.

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/basics.ipynb?flush_caches=true) **basics**: how to write Pallas kernels, up to batched matmuls.
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/splash_attention.ipynb?flush_caches=true) **splash_attention**: from vanilla softmax to the block-sparse implementation.

### From Pallas basics to grouped matrix multiplications

Grouped matrix multiplications are the core building blocks of modern MoEs.

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/basics.ipynb?flush_caches=true) **basics**: same as above.
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/grouped_matmul.ipynb?flush_caches=true) **grouped_matmul**: how to split tokens into blocks and multiply them efficiently with expert weights.

**Prerequisites**: solid JAX/NumPy. No prior Pallas experience required.

## Create your own puzzles

The notebooks were created with Claude Code. The project contains guidelines in [CLAUDE.md](CLAUDE.md) on how to create new notebooks with progressive puzzles — could be a good starting point for creating interactive study materials tailored for your needs.

## Quick start

Click a Colab badge above, or run locally:

```bash
pip install jax jaxtyping
jupyter notebook basics.ipynb
```
