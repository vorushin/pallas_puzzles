# Pallas Puzzles

Progressive exercises for learning [Pallas](https://docs.jax.dev/en/latest/pallas/index.html), JAX's kernel language for TPU and GPU.

## Notebooks

| Notebook | Puzzles | Focus |
|----------|---------|-------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/basics.ipynb?flush_caches=true) **basics** | 1-11 | Pallas foundations: Refs, grids, BlockSpec, tiled matmul, fusion |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/ragged_dot.ipynb?flush_caches=true) **ragged_dot** | 1-9 | Scalar prefetch, group metadata, grouped matmul for MoE |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/splash_attention.ipynb?flush_caches=true) **splash_attention** | 1-8 | Online softmax, flash attention, causal & block-sparse masks, splash attention |

All puzzles run on **CPU** via `interpret=True` — no TPU needed.

**Prerequisites**: solid JAX/NumPy. No prior Pallas required.

## Quick start

Click a Colab badge above, or run locally:

```bash
pip install jax jaxtyping
jupyter notebook basics.ipynb
```
