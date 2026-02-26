# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/splash_attention.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # Pallas Puzzles: Splash Attention
#
# **8 progressive puzzles** building from basic dot-product attention to
# **splash attention** — JAX's production kernel for efficient
# block-sparse attention on TPU. Along the way you'll implement online
# softmax, flash attention, causal masking, and block-sparse dispatch.
#
# Every puzzle runs on **CPU** via `interpret=True` — no TPU needed.
#
# **Prerequisites**: Complete **basics.py** first (Pallas foundations and
# tiled matmul patterns). Puzzles 7–8 optionally reference scalar prefetch
# from **ragged_dot.py**.
#
# **Key references**:
# - [Flash Attention paper](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
# - [Online normalizer calculation](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018)
# - [JAX Pallas docs](https://docs.jax.dev/en/latest/pallas/index.html)
# - [JAX splash attention source](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention)
#
# | Part | Puzzles | Focus |
# |------|---------|-------|
# | I — Flash Attention | 1–5 | Attention, online softmax, tiled flash attention |
# | II — Splash Attention | 6–8 | Causal masks, block-sparse dispatch, full splash |

# %% [markdown]
# ## Setup

# %%
#@title Install dependencies
# !pip install -q jax jaxtyping

# %%
#@title Imports
import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
print(f"JAX {jax.__version__}")


# %%
#@title check() helper
def check(kernel_fn, spec_fn, inputs, *, grid=(), in_specs=None, out_specs=None,
          out_shape=None, scratch_shapes=(), atol=1e-3, rtol=1e-3, **kwargs):
    """Run a Pallas kernel in interpret mode and compare against a reference spec.

    Args:
        kernel_fn: The Pallas kernel to test.
        spec_fn: Reference function computing the expected output in pure JAX.
        inputs: Tuple of input arrays.
        grid: Pallas grid tuple.
        in_specs: List of BlockSpec for inputs (None = no blocking).
        out_specs: BlockSpec for output (None = no blocking).
        out_shape: jax.ShapeDtypeStruct for the output.
        scratch_shapes: Scratch memory specs (empty by default).
        atol, rtol: Tolerance for comparison.
        **kwargs: Extra args to pl.pallas_call.
    """
    expected = spec_fn(*inputs)
    if out_shape is None:
        out_shape = jax.ShapeDtypeStruct(expected.shape, expected.dtype)

    # Handle default specs
    if in_specs is None:
        in_specs = [pl.BlockSpec(memory_space=pl.ANY)] * len(inputs)
    if out_specs is None:
        out_specs = pl.BlockSpec(memory_space=pl.ANY)

    actual = pl.pallas_call(
        kernel_fn,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        interpret=True,
        **kwargs,
    )(*inputs)

    if jnp.allclose(actual, expected, atol=atol, rtol=rtol):
        print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
    else:
        diff = jnp.abs(actual - expected)
        max_err = float(jnp.max(diff))
        worst_idx = jnp.unravel_index(jnp.argmax(diff), diff.shape)
        print(f"FAILED ✗  max error: {max_err:.6f} at index {tuple(int(i) for i in worst_idx)}")
        n = min(4, expected.shape[0])
        print(f"  Expected (first {n}):\n{expected[:n]}")
        print(f"  Got      (first {n}):\n{actual[:n]}")


# %% [markdown]
# ---
# # Part I: Flash Attention (Puzzles 1–5)

# %% [markdown]
# ---
# ## Puzzle 1: Dot-Product Attention
#
# **Goal**: Implement the standard attention equation in pure JAX (no Pallas
# yet). This is the **reference spec** that all later kernels must match.
#
# ### Theory
#
# Attention maps a query against a set of key-value pairs:
#
# $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
#
# where $Q, K, V \in \mathbb{R}^{T \times d}$, $T$ is the sequence length
# and $d$ is the head dimension.
#
# The score matrix $S = Q K^T / \sqrt{d}$ has shape $(T, T)$ — that's the
# **O(T²) memory bottleneck** we'll learn to eliminate. For a 4K-token
# sequence with 64-dim heads, $S$ alone is 64 MB in float32. At 128K tokens
# (common in modern LLMs), it would be **64 GB**. Clearly we can't
# materialize this matrix.
#
# ```
# Q (T×d)     K^T (d×T)       S (T×T)          P (T×T)          O (T×d)
# ┌──────┐   ┌──────────┐   ┌───────────┐    ┌───────────┐    ┌──────┐
# │      │   │          │   │           │    │ softmax   │    │      │
# │      │ @ │          │ = │  S / √d   │ →  │  rows     │ @  V  =  │  O   │
# │      │   │          │   │           │    │           │    │      │
# └──────┘   └──────────┘   └───────────┘    └───────────┘    └──────┘
#  T × d       d × T          T × T            T × T           T × d
#                            ← O(T²) memory! →
# ```
#
# Let's start by implementing this naive version, then spend the rest of the
# notebook learning to **never materialize S**.

# %%
#@title Diagram: Dot-Product Attention
from IPython.display import SVG, display
display(SVG(data='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 340" font-family="monospace" font-size="13">
  <rect width="720" height="340" fill="white"/>

  <!-- Q matrix -->
  <rect x="30" y="60" width="60" height="120" rx="4" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="60" y="130" text-anchor="middle" fill="#1e40af" font-size="12">Q</text>
  <text x="60" y="200" text-anchor="middle" fill="#6b7280" font-size="10">T × d</text>

  <!-- × sign -->
  <text x="110" y="125" text-anchor="middle" fill="#374151" font-size="18">@</text>

  <!-- K^T matrix -->
  <rect x="130" y="80" width="120" height="60" rx="4" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="190" y="117" text-anchor="middle" fill="#1e40af" font-size="12">K^T</text>
  <text x="190" y="160" text-anchor="middle" fill="#6b7280" font-size="10">d × T</text>

  <!-- = sign -->
  <text x="270" y="125" text-anchor="middle" fill="#374151" font-size="18">=</text>

  <!-- S matrix (big red square - THE BOTTLENECK) -->
  <rect x="290" y="40" width="140" height="140" rx="4" fill="#fef2f2" stroke="#ef4444" stroke-width="2.5"/>
  <text x="360" y="115" text-anchor="middle" fill="#b91c1c" font-weight="bold" font-size="14">S / √d</text>
  <text x="360" y="135" text-anchor="middle" fill="#dc2626" font-size="11">T × T</text>
  <text x="360" y="205" text-anchor="middle" fill="#dc2626" font-weight="bold" font-size="13">⚠ O(T²) memory!</text>

  <!-- Arrow down to P -->
  <line x1="360" y1="185" x2="360" y2="225" stroke="#6b7280" stroke-width="1" marker-end="url(#arrow)"/>
  <text x="385" y="215" fill="#6b7280" font-size="10">softmax</text>

  <!-- Arrow to P @ V -->
  <text x="460" y="270" text-anchor="middle" fill="#374151" font-size="14">→</text>

  <!-- P matrix -->
  <rect x="290" y="230" width="140" height="50" rx="4" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="360" y="260" text-anchor="middle" fill="#166534" font-size="12">P (T×T)</text>

  <!-- × V -->
  <text x="445" y="260" text-anchor="middle" fill="#374151" font-size="14">@</text>
  <rect x="465" y="230" width="50" height="50" rx="4" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="490" y="260" text-anchor="middle" fill="#1e40af" font-size="12">V</text>

  <!-- = O -->
  <text x="530" y="260" text-anchor="middle" fill="#374151" font-size="14">=</text>
  <rect x="545" y="230" width="60" height="50" rx="4" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="575" y="260" text-anchor="middle" fill="#166534" font-size="12">O</text>
  <text x="575" y="300" text-anchor="middle" fill="#6b7280" font-size="10">T × d</text>

  <!-- Title -->
  <text x="360" y="25" text-anchor="middle" fill="#111827" font-weight="bold" font-size="15">Dot-Product Attention</text>

  <!-- Arrow marker -->
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
  </defs>
</svg>'''))

# %%
T1 = 128    # sequence length
d1 = 64     # head dimension (per head)

Q1 = jax.random.normal(jax.random.key(0), (T1, d1))
K1 = jax.random.normal(jax.random.key(1), (T1, d1))
V1 = jax.random.normal(jax.random.key(2), (T1, d1))


# --- Reference ---
def attention_spec(Q, K, V):
    """Standard dot-product attention: softmax(Q @ K.T / sqrt(d)) @ V"""
    d = Q.shape[-1]
    S = Q @ K.T / jnp.sqrt(d).astype(Q.dtype)
    P = jax.nn.softmax(S, axis=-1)
    return P @ V


# --- Your implementation ---
def my_attention(Q, K, V):
    """Implement: softmax(Q @ K.T / sqrt(d)) @ V

    Steps:
      1. Compute scores S = Q @ K^T, scaled by 1/sqrt(d)
      2. Apply softmax along the last axis (over keys)
      3. Multiply the attention weights P by V
    """
    pass  # YOUR CODE HERE


# %%
expected1 = attention_spec(Q1, K1, V1)
actual1 = my_attention(Q1, K1, V1)

if actual1 is not None and jnp.allclose(actual1, expected1, atol=1e-3):
    print(f"PASSED ✓  (shape={actual1.shape}, dtype={actual1.dtype})")
else:
    print("FAILED ✗")
    if actual1 is not None:
        print(f"  Max error: {float(jnp.max(jnp.abs(actual1 - expected1))):.6f}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Step by step</summary>
#
# ```python
# d = Q.shape[-1]
# S = Q @ K.T / jnp.sqrt(d).astype(Q.dtype)   # (T, T) scores
# P = jax.nn.softmax(S, axis=-1)                # (T, T) weights
# return P @ V                                   # (T, d) output
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# def my_attention(Q, K, V):
#     d = Q.shape[-1]
#     S = Q @ K.T / jnp.sqrt(d).astype(Q.dtype)
#     P = jax.nn.softmax(S, axis=-1)
#     return P @ V
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 2: Tiled Softmax — The Denominator Problem
#
# **Goal**: Compute `softmax(x)` for a long vector using a Pallas kernel
# that processes tiles of the input, **accumulating the global max and
# sum(exp)** across tiles.
#
# ### Theory
#
# Numerically stable softmax has three steps:
#
# 1. **Global max**: $m = \max(x)$ — for numerical stability
# 2. **Sum of exponentials**: $\ell = \sum_i \exp(x_i - m)$
# 3. **Normalize**: $\text{softmax}(x)_i = \exp(x_i - m) / \ell$
#
# When `x` is too large for on-chip memory, we tile it. But here's the
# catch: step 1 needs to see *all* tiles before step 2 can start, and step
# 2 needs to finish before step 3. That means **three separate passes**
# over the data from HBM:
#
# ```
# Pass 1: HBM → SRAM → HBM    Find global max m
#          ──────────────→
#
# Pass 2: HBM → SRAM → HBM    Compute ℓ = Σ exp(xᵢ - m)
#          ──────────────→
#
# Pass 3: HBM → SRAM → HBM    Write exp(xᵢ - m) / ℓ
#          ──────────────→
# ```
#
# Each pass reads the entire input from slow HBM. For attention, x is a
# *row* of the score matrix — and we do this for every row. That's a lot
# of HBM traffic. Can we do better? (Spoiler: yes — Puzzle 3.)
#
# For now, let's implement the honest 3-pass version. We'll use a Pallas
# kernel for the **reduction** part (passes 1 and 2 together — compute
# max and sum_exp in one pass since they can be combined), then apply the
# normalization.
#
# This kernel tiles over the K dimension using the zero/accumulate pattern
# from basics.py Puzzle 7:
# - `@pl.when(k == 0)`: initialize max and sum_exp
# - Every tile: update running max, accumulate sum of exponentials
#
# The kernel outputs **two scalars**: the global max `m` and the
# log-sum-exp `l = log(sum(exp(x - m)))`.

# %%
N2 = 512          # vector length
bn2 = 128         # tile size
tiles_k2 = N2 // bn2

# --- Reference ---
def softmax_spec(x):
    """x: (N2,) → (N2,)"""
    return jax.nn.softmax(x)


# --- Kernel: compute (max, sum_exp) via tiled reduction ---
def softmax_stats_kernel(x_ref, m_ref, l_ref):
    """Tile over x to compute global max m and sum_exp l.

    x_ref: (bn2,) — one tile of x
    m_ref: ()     — running global max (scalar output)
    l_ref: ()     — running sum of exp(x - m) (scalar output)

    Grid: (tiles_k2,) — iterates over tiles of x
    """
    k = pl.program_id(0)
    pass  # YOUR CODE HERE
    # 1. On first tile (k == 0): set m = max(tile), l = sum(exp(tile - m))
    # 2. On later tiles: update m = max(m, max(tile)),
    #    correct l for the new max, add new exponentials


# %%
x2 = jax.random.normal(jax.random.key(10), (N2,))

# Run the stats kernel
m2_shape = jax.ShapeDtypeStruct((), jnp.float32)
l2_shape = jax.ShapeDtypeStruct((), jnp.float32)

m2, l2 = pl.pallas_call(
    softmax_stats_kernel,
    grid=(tiles_k2,),
    in_specs=[pl.BlockSpec((bn2,), lambda k: (k,))],
    out_specs=(
        pl.BlockSpec(memory_space=pl.ANY),  # m: scalar, no blocking
        pl.BlockSpec(memory_space=pl.ANY),  # l: scalar, no blocking
    ),
    out_shape=(m2_shape, l2_shape),
    interpret=True,
)(x2)

# Now use m and l to compute softmax (this part is given)
softmax2 = jnp.exp(x2 - m2) / l2

expected2 = softmax_spec(x2)
if jnp.allclose(softmax2, expected2, atol=1e-3):
    print(f"PASSED ✓  (m={float(m2):.3f}, l={float(l2):.3f})")
else:
    print(f"FAILED ✗  max error: {float(jnp.max(jnp.abs(softmax2 - expected2))):.6f}")
    print(f"  m={float(m2):.3f} (expected {float(jnp.max(x2)):.3f})")
    print(f"  l={float(l2):.3f} (expected {float(jnp.sum(jnp.exp(x2 - jnp.max(x2)))):.3f})")

# %% [markdown]
# <details><summary>Hint 1 of 3 — First tile</summary>
#
# On the first tile, just compute the local statistics:
# ```python
# @pl.when(k == 0)
# def _():
#     tile = x_ref[...]
#     m_ref[...] = jnp.max(tile)
#     l_ref[...] = jnp.sum(jnp.exp(tile - jnp.max(tile)))
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — Later tiles (the tricky part)</summary>
#
# When a new tile has a larger max, you need to **correct** the running sum:
# ```python
# @pl.when(k > 0)
# def _():
#     tile = x_ref[...]
#     m_old = m_ref[...]
#     m_new = jnp.maximum(m_old, jnp.max(tile))
#     # Old exponentials were computed with m_old — rescale them
#     correction = jnp.exp(m_old - m_new)
#     l_ref[...] = l_ref[...] * correction + jnp.sum(jnp.exp(tile - m_new))
#     m_ref[...] = m_new
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# def softmax_stats_kernel(x_ref, m_ref, l_ref):
#     k = pl.program_id(0)
#
#     @pl.when(k == 0)
#     def _():
#         tile = x_ref[...]
#         m_ref[...] = jnp.max(tile)
#         l_ref[...] = jnp.sum(jnp.exp(tile - jnp.max(tile)))
#
#     @pl.when(k > 0)
#     def _():
#         tile = x_ref[...]
#         m_old = m_ref[...]
#         m_new = jnp.maximum(m_old, jnp.max(tile))
#         correction = jnp.exp(m_old - m_new)
#         l_ref[...] = l_ref[...] * correction + jnp.sum(jnp.exp(tile - m_new))
#         m_ref[...] = m_new
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 3: Online Softmax — One Pass to Rule Them All
#
# **Goal**: Compute softmax in a **single pass** by maintaining running
# statistics that get corrected on-the-fly as new tiles arrive.
#
# ### Theory
#
# The 3-pass softmax from Puzzle 2 works, but it reads the data from HBM
# multiple times. **Online softmax** (Milakov & Gimelshein, 2018) is THE
# breakthrough that makes flash attention possible — it computes softmax
# in a **single pass** by maintaining running statistics `(m, ℓ)` that
# self-correct:
#
# ```
# Initialize: m = -∞,  ℓ = 0
#
# For each tile xᵢ:
#     m_new = max(m, max(xᵢ))           ← update running max
#     correction = exp(m - m_new)        ← rescale factor for old stats
#     ℓ = ℓ · correction                 ← correct old sum
#       + Σⱼ exp(xᵢⱼ - m_new)           ← add new exponentials
#     m = m_new
# ```
#
# The **correction factor** `exp(m_old - m_new)` is the magic. When a new
# tile has a bigger max, all previous exponentials need to be rescaled.
# Instead of going back and recomputing them, we just multiply the running
# sum by this factor. It works because:
#
# ```
# exp(x - m_old) · exp(m_old - m_new) = exp(x - m_new)
# ```
#
# After processing all tiles, `m` is the global max and `ℓ` is
# `sum(exp(x - m))` — exactly what softmax needs.
#
# Doesn't this look a lot like what you wrote in Puzzle 2? It is! But
# there's a crucial difference: in Puzzle 2 we used separate `@pl.when`
# branches for `k == 0` and `k > 0`. Here we initialize `m = -∞` and
# `ℓ = 0` and let the same update rule handle all tiles uniformly —
# including the first one. This unified formulation is what we'll use in
# flash attention.
#
# ```
# Tile 0          Tile 1          Tile 2          Tile 3
# ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐
# │ x[0] │  ───→ │ x[1] │  ───→ │ x[2] │  ───→ │ x[3] │
# └──────┘       └──────┘       └──────┘       └──────┘
#    │               │               │               │
# m=-∞, ℓ=0     update m,ℓ     update m,ℓ     update m,ℓ
#    │               │               │               │
# m=3.2, ℓ=47   m=3.5, ℓ=89   m=3.5, ℓ=134  m=3.7, ℓ=201
#                  ↑ correction!              ↑ correction!
# ```
#
# **Your task**: Implement a Pallas kernel that computes `m` and `ℓ` in a
# single pass, then use them to compute the final softmax output.

# %%
#@title Diagram: Online Softmax
from IPython.display import SVG, display
display(SVG(data='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 300" font-family="monospace" font-size="12">
  <rect width="760" height="300" fill="white"/>

  <!-- Title -->
  <text x="380" y="25" text-anchor="middle" fill="#111827" font-weight="bold" font-size="15">Online Softmax — Single Pass</text>

  <!-- Initial state -->
  <text x="15" y="90" fill="#6b7280" font-size="11">m = -∞</text>
  <text x="15" y="108" fill="#6b7280" font-size="11">ℓ = 0</text>

  <!-- Tile 0 -->
  <rect x="100" y="60" width="120" height="60" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="160" y="85" text-anchor="middle" fill="#1e40af" font-weight="bold">Tile 0</text>
  <text x="160" y="105" text-anchor="middle" fill="#3b82f6" font-size="10">x[0:128]</text>
  <text x="160" y="145" text-anchor="middle" fill="#1e3a5f" font-size="11">m=2.1</text>
  <text x="160" y="163" text-anchor="middle" fill="#1e3a5f" font-size="11">ℓ=47.3</text>

  <!-- Arrow 0→1 -->
  <line x1="225" y1="90" x2="265" y2="90" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Tile 1 -->
  <rect x="270" y="60" width="120" height="60" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="330" y="85" text-anchor="middle" fill="#1e40af" font-weight="bold">Tile 1</text>
  <text x="330" y="105" text-anchor="middle" fill="#3b82f6" font-size="10">x[128:256]</text>
  <text x="330" y="145" text-anchor="middle" fill="#1e3a5f" font-size="11">m=3.4</text>
  <text x="330" y="163" text-anchor="middle" fill="#1e3a5f" font-size="11">ℓ=89.1</text>

  <!-- Correction star 1 -->
  <text x="330" y="188" text-anchor="middle" fill="#f59e0b" font-size="18">★</text>
  <text x="330" y="205" text-anchor="middle" fill="#d97706" font-size="10" font-weight="bold">correction!</text>
  <text x="330" y="218" text-anchor="middle" fill="#92400e" font-size="9">ℓ *= exp(2.1 - 3.4)</text>

  <!-- Arrow 1→2 -->
  <line x1="395" y1="90" x2="435" y2="90" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Tile 2 -->
  <rect x="440" y="60" width="120" height="60" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="500" y="85" text-anchor="middle" fill="#1e40af" font-weight="bold">Tile 2</text>
  <text x="500" y="105" text-anchor="middle" fill="#3b82f6" font-size="10">x[256:384]</text>
  <text x="500" y="145" text-anchor="middle" fill="#1e3a5f" font-size="11">m=3.4</text>
  <text x="500" y="163" text-anchor="middle" fill="#1e3a5f" font-size="11">ℓ=134.7</text>
  <text x="500" y="183" text-anchor="middle" fill="#6b7280" font-size="9">(no correction)</text>

  <!-- Arrow 2→3 -->
  <line x1="565" y1="90" x2="605" y2="90" stroke="#9ca3af" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Tile 3 -->
  <rect x="610" y="60" width="120" height="60" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>
  <text x="670" y="85" text-anchor="middle" fill="#1e40af" font-weight="bold">Tile 3</text>
  <text x="670" y="105" text-anchor="middle" fill="#3b82f6" font-size="10">x[384:512]</text>
  <text x="670" y="145" text-anchor="middle" fill="#1e3a5f" font-size="11">m=3.7</text>
  <text x="670" y="163" text-anchor="middle" fill="#1e3a5f" font-size="11">ℓ=201.5</text>

  <!-- Correction star 3 -->
  <text x="670" y="188" text-anchor="middle" fill="#f59e0b" font-size="18">★</text>
  <text x="670" y="205" text-anchor="middle" fill="#d97706" font-size="10" font-weight="bold">correction!</text>
  <text x="670" y="218" text-anchor="middle" fill="#92400e" font-size="9">ℓ *= exp(3.4 - 3.7)</text>

  <!-- Final result -->
  <rect x="200" y="250" width="360" height="35" rx="6" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="380" y="273" text-anchor="middle" fill="#166534" font-weight="bold" font-size="12">Final: m = global max, ℓ = Σ exp(xᵢ - m)  ✓ one pass!</text>

  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#9ca3af"/>
    </marker>
  </defs>
</svg>'''))

# %%
N3 = 512
bn3 = 128
tiles_k3 = N3 // bn3

# --- Reference ---
def softmax_spec3(x):
    """x: (N3,) → (N3,)"""
    return jax.nn.softmax(x)


# --- Kernel: online softmax stats ---
def online_softmax_kernel(x_ref, m_ref, l_ref):
    """Single-pass softmax stats: m = global max, l = sum(exp(x - m)).

    x_ref: (bn3,) — one tile of x
    m_ref: ()     — running global max (scalar)
    l_ref: ()     — running sum of exponentials (scalar)

    Grid: (tiles_k3,) — iterates over tiles of x

    Unlike Puzzle 2, we use a unified update rule for ALL tiles:
    - Initialize m = -inf, l = 0 on first tile
    - Same correction logic handles both first and subsequent tiles
    """
    k = pl.program_id(0)
    pass  # YOUR CODE HERE
    # 1. On first tile: initialize m_ref = -inf, l_ref = 0
    # 2. On ALL tiles (including first): read tile, compute new max,
    #    apply correction to l, add new exponentials


# %%
x3 = jax.random.normal(jax.random.key(20), (N3,))

m3, l3 = pl.pallas_call(
    online_softmax_kernel,
    grid=(tiles_k3,),
    in_specs=[pl.BlockSpec((bn3,), lambda k: (k,))],
    out_specs=(
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
    ),
    out_shape=(
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.float32),
    ),
    interpret=True,
)(x3)

softmax3 = jnp.exp(x3 - m3) / l3

expected3 = softmax_spec3(x3)
if jnp.allclose(softmax3, expected3, atol=1e-3):
    print(f"PASSED ✓  (m={float(m3):.3f}, l={float(l3):.3f})")
else:
    print(f"FAILED ✗  max error: {float(jnp.max(jnp.abs(softmax3 - expected3))):.6f}")
    print(f"  m={float(m3):.3f} (expected {float(jnp.max(x3)):.3f})")
    print(f"  l={float(l3):.3f} (expected {float(jnp.sum(jnp.exp(x3 - jnp.max(x3)))):.3f})")

# %% [markdown]
# <details><summary>Hint 1 of 3 — The unified update rule</summary>
#
# Initialize on the first tile, then run the same update code on ALL tiles:
# ```python
# @pl.when(k == 0)
# def _init():
#     m_ref[...] = jnp.float32(-jnp.inf)
#     l_ref[...] = jnp.float32(0.0)
#
# # This runs on EVERY tile (including first!)
# tile = x_ref[...]
# m_new = jnp.maximum(m_ref[...], jnp.max(tile))
# ...
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — Correction factor</summary>
#
# ```python
# correction = jnp.exp(m_ref[...] - m_new)
# l_ref[...] = l_ref[...] * correction + jnp.sum(jnp.exp(tile - m_new))
# m_ref[...] = m_new
# ```
# When `k == 0`: `m_ref = -inf`, so `correction = exp(-inf - m_new) = 0`,
# and `l_ref = 0 * 0 + sum(exp(tile - m_new))`. It just works!
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# def online_softmax_kernel(x_ref, m_ref, l_ref):
#     k = pl.program_id(0)
#
#     @pl.when(k == 0)
#     def _init():
#         m_ref[...] = jnp.float32(-jnp.inf)
#         l_ref[...] = jnp.float32(0.0)
#
#     tile = x_ref[...]
#     m_new = jnp.maximum(m_ref[...], jnp.max(tile))
#     correction = jnp.exp(m_ref[...] - m_new)
#     l_ref[...] = l_ref[...] * correction + jnp.sum(jnp.exp(tile - m_new))
#     m_ref[...] = m_new
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 4: Tiled Attention — One Q Block
#
# **Goal**: Compute the attention output for a **single block of Q rows**,
# iterating over all K,V blocks with online softmax to avoid materializing
# the full score matrix.
#
# ### Theory
#
# Now we combine online softmax with tiled matmul. This is the **core loop**
# of flash attention — processing one Q block against all KV blocks.
#
# For a Q block of shape `(bq, d)`, we iterate over KV blocks:
#
# ```
#   Q block       K blocks (iterate →)      Output
#   ┌──────┐     ┌──────┬──────┬──────┬──────┐     ┌──────┐
#   │      │     │      │      │      │      │     │      │
#   │ bq×d │  @  │ bk×d │ bk×d │ bk×d │ bk×d │  →  │ bq×d │
#   │      │     │      │      │      │      │     │      │
#   └──────┘     └──────┴──────┴──────┴──────┘     └──────┘
#                 kv=0    kv=1   kv=2   kv=3
# ```
#
# For each KV block, the kernel:
# 1. Computes scores: `s = Q_block @ K_block.T / sqrt(d)` → `(bq, bk)`
# 2. Updates online softmax stats `(m, ℓ)` per row
# 3. **Corrects** the running output accumulator for the new max
# 4. Adds new contribution: `acc += P_block @ V_block`
# 5. After last KV block: normalizes by `1/ℓ`
#
# The key insight is step 3: when the max changes, we must **rescale**
# the entire accumulator. Without this correction, outputs from earlier
# KV blocks would have the wrong scale:
#
# ```python
# correction = exp(m_old - m_new)     # (bq,) per-row correction
# acc = acc * correction[:, None]     # rescale all d columns
# acc += P_block @ V_block            # add new contribution
# ```
#
# After the last KV block, we normalize: `output = acc / ℓ[:, None]`.
# (This is because we've been accumulating unnormalized `exp(s - m) @ V`,
# and need to divide by the total `ℓ = sum(exp(s - m))` at the end.)
#
# **Scratch memory** holds three things:
# - `acc`: `(bq, d)` — running output accumulator
# - `m`: `(bq,)` — running max per row
# - `l`: `(bq,)` — running sum of exponentials per row

# %%
T4 = 128      # sequence length
d4 = 64       # head dimension
bq4 = 32      # Q block size
bk4 = 32      # KV block size
tiles_kv4 = T4 // bk4

Q4 = jax.random.normal(jax.random.key(30), (T4, d4))
K4 = jax.random.normal(jax.random.key(31), (T4, d4))
V4 = jax.random.normal(jax.random.key(32), (T4, d4))

# --- Reference: attention for just the first Q block ---
def attention_one_block_spec(Q, K, V):
    """Attention output for first bq4 rows only."""
    q_block = Q[:bq4]                          # (bq4, d4)
    S = q_block @ K.T / jnp.sqrt(d4).astype(Q.dtype)  # (bq4, T4)
    P = jax.nn.softmax(S, axis=-1)             # (bq4, T4)
    return P @ V                               # (bq4, d4)


# --- Kernel: tiled attention for one Q block ---
def tiled_attention_one_block_kernel(
    q_ref,      # (bq4, d4) — the Q block (same for all KV iterations)
    k_ref,      # (bk4, d4) — one KV block
    v_ref,      # (bk4, d4) — one KV block
    o_ref,      # (bq4, d4) — output
    acc_ref,    # (bq4, d4) — scratch: running output accumulator
    m_ref,      # (bq4,)    — scratch: running row max
    l_ref,      # (bq4,)    — scratch: running row sum_exp
):
    """Process one KV block for a single Q block using online softmax.

    Grid: (tiles_kv4,) — iterates over KV blocks
    """
    kv = pl.program_id(0)
    pass  # YOUR CODE HERE
    # 1. On first KV block: init acc=0, m=-inf, l=0
    # 2. Compute scores: s = q @ k.T / sqrt(d4)       → (bq4, bk4)
    # 3. Compute row-wise max of scores: m_tile        → (bq4,)
    # 4. Update running max: m_new = max(m, m_tile)    → (bq4,)
    # 5. Correction factor: corr = exp(m - m_new)      → (bq4,)
    # 6. Rescale accumulator: acc *= corr[:, None]
    # 7. Compute P_block = exp(s - m_new[:, None])     → (bq4, bk4)
    # 8. Update l: l = l * corr + P_block.sum(axis=-1)
    # 9. Accumulate: acc += P_block @ v
    # 10. Update m = m_new
    # 11. On LAST KV block: o = acc / l[:, None]


# %%
expected4 = attention_one_block_spec(Q4, K4, V4)

actual4 = pl.pallas_call(
    tiled_attention_one_block_kernel,
    grid=(tiles_kv4,),
    in_specs=[
        pl.BlockSpec((bq4, d4), lambda kv: (0, 0)),       # Q: always first block
        pl.BlockSpec((bk4, d4), lambda kv: (kv, 0)),      # K: iterate over blocks
        pl.BlockSpec((bk4, d4), lambda kv: (kv, 0)),      # V: iterate over blocks
    ],
    out_specs=pl.BlockSpec((bq4, d4), lambda kv: (0, 0)),
    out_shape=jax.ShapeDtypeStruct((bq4, d4), jnp.float32),
    scratch_shapes=[
        pltpu.VMEM((bq4, d4), jnp.float32),    # acc
        pltpu.VMEM((bq4,), jnp.float32),        # m
        pltpu.VMEM((bq4,), jnp.float32),        # l
    ],
    interpret=True,
)(Q4, K4, V4)

if jnp.allclose(actual4, expected4, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual4.shape})")
else:
    diff4 = jnp.abs(actual4 - expected4)
    print(f"FAILED ✗  max error: {float(jnp.max(diff4)):.6f}")
    print(f"  Expected (first 2 rows):\n{expected4[:2, :8]}")
    print(f"  Got      (first 2 rows):\n{actual4[:2, :8]}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Initialization and scores</summary>
#
# ```python
# @pl.when(kv == 0)
# def _init():
#     acc_ref[...] = jnp.zeros((bq4, d4), dtype=jnp.float32)
#     m_ref[...] = jnp.full((bq4,), -jnp.inf, dtype=jnp.float32)
#     l_ref[...] = jnp.zeros((bq4,), dtype=jnp.float32)
#
# q = q_ref[...]
# k = k_ref[...]
# v = v_ref[...]
# s = q @ k.T / jnp.sqrt(d4).astype(jnp.float32)  # (bq4, bk4)
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — Online softmax update with output correction</summary>
#
# ```python
# m_tile = jnp.max(s, axis=-1)                    # (bq4,)
# m_new = jnp.maximum(m_ref[...], m_tile)          # (bq4,)
# corr = jnp.exp(m_ref[...] - m_new)               # (bq4,)
#
# acc_ref[...] = acc_ref[...] * corr[:, None]       # rescale old output
# p = jnp.exp(s - m_new[:, None])                   # (bq4, bk4)
# l_ref[...] = l_ref[...] * corr + p.sum(axis=-1)  # update sum_exp
# acc_ref[...] = acc_ref[...] + p @ v               # accumulate P @ V
# m_ref[...] = m_new
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# def tiled_attention_one_block_kernel(q_ref, k_ref, v_ref, o_ref,
#                                       acc_ref, m_ref, l_ref):
#     kv = pl.program_id(0)
#
#     @pl.when(kv == 0)
#     def _init():
#         acc_ref[...] = jnp.zeros((bq4, d4), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq4,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq4,), dtype=jnp.float32)
#
#     q = q_ref[...]
#     k = k_ref[...]
#     v = v_ref[...]
#     s = q @ k.T / jnp.sqrt(d4).astype(jnp.float32)
#
#     m_tile = jnp.max(s, axis=-1)
#     m_new = jnp.maximum(m_ref[...], m_tile)
#     corr = jnp.exp(m_ref[...] - m_new)
#
#     acc_ref[...] = acc_ref[...] * corr[:, None]
#     p = jnp.exp(s - m_new[:, None])
#     l_ref[...] = l_ref[...] * corr + p.sum(axis=-1)
#     acc_ref[...] = acc_ref[...] + p @ v
#     m_ref[...] = m_new
#
#     @pl.when(kv == tiles_kv4 - 1)
#     def _norm():
#         o_ref[...] = acc_ref[...] / l_ref[...][:, None]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 5: Flash Attention Forward
#
# **Goal**: Extend Puzzle 4 to process **all Q blocks** via a 2D grid.
# This is full **flash attention** — computing exact attention without
# materializing the T×T score matrix.
#
# ### Theory
#
# In Puzzle 4, we handled one Q block. Now we grid over all Q blocks.
# The grid has two dimensions:
#
# ```
# grid = (tiles_q, tiles_kv)
#         ↓         ↓
#     which Q    iterate over
#     block      all KV blocks
# ```
#
# For each Q block `i`, the kernel sweeps through ALL KV blocks
# `kv ∈ [0, tiles_kv)`, maintaining per-row online softmax statistics.
# Each Q block is completely independent — they don't share state.
#
# ```
#                    K blocks
#              kv=0  kv=1  kv=2  kv=3
#            ┌──────┬──────┬──────┬──────┐
#  Q    i=0  │ s0,0 │ s0,1 │ s0,2 │ s0,3 │ → O[0:bq]     ← grid point (0, *)
# blocks     ├──────┼──────┼──────┼──────┤
#       i=1  │ s1,0 │ s1,1 │ s1,2 │ s1,3 │ → O[bq:2*bq]  ← grid point (1, *)
#            ├──────┼──────┼──────┼──────┤
#       i=2  │ s2,0 │ s2,1 │ s2,2 │ s2,3 │ → O[2*bq:3*bq] ← grid point (2, *)
#            ├──────┼──────┼──────┼──────┤
#       i=3  │ s3,0 │ s3,1 │ s3,2 │ s3,3 │ → O[3*bq:4*bq] ← grid point (3, *)
#            └──────┴──────┴──────┴──────┘
#
# We never materialize the full score matrix!
# Each block s[i,j] = Q_block_i @ K_block_j.T / √d  is (bq × bk)
# and lives only in SRAM for the duration of that grid point.
# ```
#
# The kernel body is almost identical to Puzzle 4. The only differences:
# - `i = pl.program_id(0)` selects which Q block
# - `kv = pl.program_id(1)` iterates over KV blocks
# - BlockSpecs route Q/O by `i`, and K/V by `kv`
#
# **This is flash attention.** Same exact outputs as Puzzle 1, but O(T)
# memory instead of O(T²). The score matrix never exists in full — each
# `(bq, bk)` tile is computed, used, and discarded.

# %%
#@title Diagram: Flash Attention Grid
from IPython.display import SVG, display
display(SVG(data='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 580 460" font-family="monospace" font-size="12">
  <rect width="580" height="460" fill="white"/>

  <!-- Title -->
  <text x="290" y="25" text-anchor="middle" fill="#111827" font-weight="bold" font-size="15">Flash Attention — Tiled Grid</text>

  <!-- Column labels -->
  <text x="195" y="55" text-anchor="middle" fill="#6b7280" font-size="10">KV 0</text>
  <text x="275" y="55" text-anchor="middle" fill="#6b7280" font-size="10">KV 1</text>
  <text x="355" y="55" text-anchor="middle" fill="#6b7280" font-size="10">KV 2</text>
  <text x="435" y="55" text-anchor="middle" fill="#6b7280" font-size="10">KV 3</text>

  <!-- Row labels -->
  <text x="125" y="100" text-anchor="end" fill="#6b7280" font-size="10">Q block 0</text>
  <text x="125" y="180" text-anchor="end" fill="#6b7280" font-size="10">Q block 1</text>
  <text x="125" y="260" text-anchor="end" fill="#6b7280" font-size="10">Q block 2</text>
  <text x="125" y="340" text-anchor="end" fill="#6b7280" font-size="10">Q block 3</text>

  <!-- Grid cells - all blue since flash processes everything -->
  <!-- Row 0 -->
  <rect x="155" y="65" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
  <rect x="235" y="65" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
  <rect x="315" y="65" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
  <rect x="395" y="65" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>

  <!-- Row 1 - highlighted as active -->
  <rect x="155" y="145" width="70" height="65" rx="3" fill="#93c5fd" stroke="#3b82f6" stroke-width="2.5"/>
  <rect x="235" y="145" width="70" height="65" rx="3" fill="#93c5fd" stroke="#3b82f6" stroke-width="2.5"/>
  <rect x="315" y="145" width="70" height="65" rx="3" fill="#93c5fd" stroke="#3b82f6" stroke-width="2.5"/>
  <rect x="395" y="145" width="70" height="65" rx="3" fill="#93c5fd" stroke="#3b82f6" stroke-width="2.5"/>

  <!-- Active row arrow -->
  <line x1="160" y1="178" x2="458" y2="178" stroke="#2563eb" stroke-width="2" stroke-dasharray="6,3" marker-end="url(#bluearr)"/>
  <text x="490" y="170" fill="#2563eb" font-size="10" font-weight="bold">iterate</text>
  <text x="490" y="185" fill="#2563eb" font-size="10" font-weight="bold">all KV →</text>

  <!-- Row 2 -->
  <rect x="155" y="225" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
  <rect x="235" y="225" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
  <rect x="315" y="225" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
  <rect x="395" y="225" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>

  <!-- Row 3 -->
  <rect x="155" y="305" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
  <rect x="235" y="305" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
  <rect x="315" y="305" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>
  <rect x="395" y="305" width="70" height="65" rx="3" fill="#dbeafe" stroke="#93c5fd" stroke-width="1"/>

  <!-- Cell labels - bq×bk -->
  <text x="190" y="102" text-anchor="middle" fill="#3b82f6" font-size="9">bq×bk</text>
  <text x="270" y="102" text-anchor="middle" fill="#3b82f6" font-size="9">bq×bk</text>

  <!-- Bottom label -->
  <text x="290" y="400" text-anchor="middle" fill="#374151" font-size="11">Each (bq×bk) score tile computed in SRAM, then discarded</text>
  <text x="290" y="418" text-anchor="middle" fill="#374151" font-size="11">Score matrix never fully materialized</text>

  <!-- Memory badge -->
  <rect x="210" y="430" width="160" height="25" rx="12" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="290" y="448" text-anchor="middle" fill="#166534" font-weight="bold" font-size="12">O(T) memory ✓</text>

  <defs>
    <marker id="bluearr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
  </defs>
</svg>'''))

# %%
T5 = 128
d5 = 64
bq5 = 32
bk5 = 32
tiles_q5 = T5 // bq5
tiles_kv5 = T5 // bk5

Q5 = jax.random.normal(jax.random.key(40), (T5, d5))
K5 = jax.random.normal(jax.random.key(41), (T5, d5))
V5 = jax.random.normal(jax.random.key(42), (T5, d5))

# --- Reference ---
def flash_attention_spec(Q, K, V):
    """Full attention: softmax(Q @ K.T / sqrt(d)) @ V"""
    d = Q.shape[-1]
    S = Q @ K.T / jnp.sqrt(d).astype(Q.dtype)
    P = jax.nn.softmax(S, axis=-1)
    return P @ V


# --- Kernel ---
def flash_attention_kernel(
    q_ref,      # (bq5, d5)
    k_ref,      # (bk5, d5)
    v_ref,      # (bk5, d5)
    o_ref,      # (bq5, d5) — output
    acc_ref,    # (bq5, d5) — scratch: accumulator
    m_ref,      # (bq5,)    — scratch: running max
    l_ref,      # (bq5,)    — scratch: running sum_exp
):
    """Flash attention kernel.

    Grid: (tiles_q5, tiles_kv5)
      - program_id(0) = i: which Q block
      - program_id(1) = kv: which KV block (reduction dimension)
    """
    i = pl.program_id(0)
    kv = pl.program_id(1)
    pass  # YOUR CODE HERE
    # Same pattern as Puzzle 4, but now:
    # - Use kv (not i) for the KV iteration
    # - Init on kv == 0, normalize on kv == tiles_kv5 - 1
    # - The BlockSpecs handle routing Q by i, K/V by kv


# %%
check(flash_attention_kernel, flash_attention_spec, (Q5, K5, V5),
      grid=(tiles_q5, tiles_kv5),
      in_specs=[
          pl.BlockSpec((bq5, d5), lambda i, kv: (i, 0)),    # Q: route by i
          pl.BlockSpec((bk5, d5), lambda i, kv: (kv, 0)),   # K: route by kv
          pl.BlockSpec((bk5, d5), lambda i, kv: (kv, 0)),   # V: route by kv
      ],
      out_specs=pl.BlockSpec((bq5, d5), lambda i, kv: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((T5, d5), jnp.float32),
      scratch_shapes=[
          pltpu.VMEM((bq5, d5), jnp.float32),   # acc
          pltpu.VMEM((bq5,), jnp.float32),       # m
          pltpu.VMEM((bq5,), jnp.float32),       # l
      ],
      atol=1e-2, rtol=1e-2)

# %% [markdown]
# <details><summary>Hint 1 of 2 — It's Puzzle 4 with different program_id</summary>
#
# The kernel body is identical to Puzzle 4. Just:
# - Use `kv = pl.program_id(1)` instead of `pl.program_id(0)`
# - Init on `kv == 0`, normalize on `kv == tiles_kv5 - 1`
# - The BlockSpecs handle routing — you don't need to think about `i` inside the kernel
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# def flash_attention_kernel(q_ref, k_ref, v_ref, o_ref,
#                            acc_ref, m_ref, l_ref):
#     i = pl.program_id(0)
#     kv = pl.program_id(1)
#
#     @pl.when(kv == 0)
#     def _init():
#         acc_ref[...] = jnp.zeros((bq5, d5), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq5,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq5,), dtype=jnp.float32)
#
#     q = q_ref[...]
#     k = k_ref[...]
#     v = v_ref[...]
#     s = q @ k.T / jnp.sqrt(d5).astype(jnp.float32)
#
#     m_tile = jnp.max(s, axis=-1)
#     m_new = jnp.maximum(m_ref[...], m_tile)
#     corr = jnp.exp(m_ref[...] - m_new)
#
#     acc_ref[...] = acc_ref[...] * corr[:, None]
#     p = jnp.exp(s - m_new[:, None])
#     l_ref[...] = l_ref[...] * corr + p.sum(axis=-1)
#     acc_ref[...] = acc_ref[...] + p @ v
#     m_ref[...] = m_new
#
#     @pl.when(kv == tiles_kv5 - 1)
#     def _norm():
#         o_ref[...] = acc_ref[...] / l_ref[...][:, None]
# ```
# </details>

# %% [markdown]
# ---
# # Part II: Splash Attention (Puzzles 6–8)
#
# Flash attention processes every KV block for every Q block.
# But many attention patterns have **structure** — causal masking, sliding
# windows, block-sparse patterns. **Splash attention** exploits this
# structure by **skipping entire blocks** that would be fully masked.

# %% [markdown]
# ---
# ## Puzzle 6: Causal Masking
#
# **Goal**: Add a causal mask to flash attention, **skipping** blocks that
# are entirely above the diagonal.
#
# ### Theory
#
# In autoregressive models (GPT, LLaMA, etc.), token $i$ can only attend
# to tokens $j \le i$. This triangular mask zeros out the upper-right
# portion of the score matrix.
#
# When we tile the score matrix into blocks, each block falls into one of
# three categories:
#
# ```
#              KV blocks
#           kv=0  kv=1  kv=2  kv=3
#         ┌──────┬──────┬──────┬──────┐
#    i=0  │ PART │ SKIP │ SKIP │ SKIP │
#         ├──────┼──────┼──────┼──────┤
#    i=1  │ FULL │ PART │ SKIP │ SKIP │
#  Q      ├──────┼──────┼──────┼──────┤
# blocks  │ FULL │ FULL │ PART │ SKIP │
#    i=2  ├──────┼──────┼──────┼──────┤
#    i=3  │ FULL │ FULL │ FULL │ PART │
#         └──────┴──────┴──────┴──────┘
#
#  FULL  = all positions visible → compute normally
#  PART  = diagonal block → apply element-wise mask
#  SKIP  = all positions masked → skip entirely (don't load K,V!)
# ```
#
# **When to skip**: Block `(i, kv)` is entirely above the diagonal when
# the **first Q row** in the block (= `i * bq`) is less than the **last
# K column** in the block (= `(kv + 1) * bk - 1`). Simplifying:
# `i * bq < kv * bk` (when bq == bk).
#
# **When to mask**: For partial blocks (on the diagonal), we create a
# boolean mask using global row/column indices:
# ```python
# q_idx = i * bq + jnp.arange(bq)[:, None]     # (bq, 1)
# kv_idx = kv * bk + jnp.arange(bk)[None, :]   # (1, bk)
# causal_mask = q_idx >= kv_idx                  # (bq, bk) True where visible
# ```
# Then apply: `s = jnp.where(causal_mask, s, -jnp.inf)` — masked positions
# become $-\infty$ so `exp(−∞) = 0` in softmax.
#
# Skipping blocks is a huge win: for causal attention, ~half the blocks
# are skipped. And the `@pl.when` guard means K and V tiles for skipped
# blocks are **never loaded from HBM**.

# %%
#@title Diagram: Causal Block Mask
from IPython.display import SVG, display
display(SVG(data='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 480" font-family="monospace" font-size="12">
  <rect width="560" height="480" fill="white"/>

  <!-- Title -->
  <text x="280" y="25" text-anchor="middle" fill="#111827" font-weight="bold" font-size="15">Causal Block Mask</text>

  <!-- Column labels -->
  <text x="180" y="55" text-anchor="middle" fill="#6b7280" font-size="11">kv=0</text>
  <text x="260" y="55" text-anchor="middle" fill="#6b7280" font-size="11">kv=1</text>
  <text x="340" y="55" text-anchor="middle" fill="#6b7280" font-size="11">kv=2</text>
  <text x="420" y="55" text-anchor="middle" fill="#6b7280" font-size="11">kv=3</text>

  <!-- Row labels -->
  <text x="115" y="100" text-anchor="end" fill="#6b7280" font-size="11">i=0</text>
  <text x="115" y="180" text-anchor="end" fill="#6b7280" font-size="11">i=1</text>
  <text x="115" y="260" text-anchor="end" fill="#6b7280" font-size="11">i=2</text>
  <text x="115" y="340" text-anchor="end" fill="#6b7280" font-size="11">i=3</text>

  <!-- Row 0: P S S S -->
  <rect x="140" y="65" width="70" height="65" rx="3" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <text x="175" y="103" text-anchor="middle" fill="#854d0e" font-weight="bold" font-size="16">P</text>
  <rect x="220" y="65" width="70" height="65" rx="3" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="255" y="103" text-anchor="middle" fill="#991b1b" font-weight="bold" font-size="16">S</text>
  <rect x="300" y="65" width="70" height="65" rx="3" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="335" y="103" text-anchor="middle" fill="#991b1b" font-weight="bold" font-size="16">S</text>
  <rect x="380" y="65" width="70" height="65" rx="3" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="415" y="103" text-anchor="middle" fill="#991b1b" font-weight="bold" font-size="16">S</text>

  <!-- Row 1: F P S S -->
  <rect x="140" y="145" width="70" height="65" rx="3" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="175" y="183" text-anchor="middle" fill="#166534" font-weight="bold" font-size="16">F</text>
  <rect x="220" y="145" width="70" height="65" rx="3" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <text x="255" y="183" text-anchor="middle" fill="#854d0e" font-weight="bold" font-size="16">P</text>
  <rect x="300" y="145" width="70" height="65" rx="3" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="335" y="183" text-anchor="middle" fill="#991b1b" font-weight="bold" font-size="16">S</text>
  <rect x="380" y="145" width="70" height="65" rx="3" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="415" y="183" text-anchor="middle" fill="#991b1b" font-weight="bold" font-size="16">S</text>

  <!-- Row 2: F F P S -->
  <rect x="140" y="225" width="70" height="65" rx="3" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="175" y="263" text-anchor="middle" fill="#166534" font-weight="bold" font-size="16">F</text>
  <rect x="220" y="225" width="70" height="65" rx="3" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="255" y="263" text-anchor="middle" fill="#166534" font-weight="bold" font-size="16">F</text>
  <rect x="300" y="225" width="70" height="65" rx="3" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <text x="335" y="263" text-anchor="middle" fill="#854d0e" font-weight="bold" font-size="16">P</text>
  <rect x="380" y="225" width="70" height="65" rx="3" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="415" y="263" text-anchor="middle" fill="#991b1b" font-weight="bold" font-size="16">S</text>

  <!-- Row 3: F F F P -->
  <rect x="140" y="305" width="70" height="65" rx="3" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="175" y="343" text-anchor="middle" fill="#166534" font-weight="bold" font-size="16">F</text>
  <rect x="220" y="305" width="70" height="65" rx="3" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="255" y="343" text-anchor="middle" fill="#166534" font-weight="bold" font-size="16">F</text>
  <rect x="300" y="305" width="70" height="65" rx="3" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="335" y="343" text-anchor="middle" fill="#166534" font-weight="bold" font-size="16">F</text>
  <rect x="380" y="305" width="70" height="65" rx="3" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <text x="415" y="343" text-anchor="middle" fill="#854d0e" font-weight="bold" font-size="16">P</text>

  <!-- Legend -->
  <rect x="80" y="400" width="20" height="20" rx="3" fill="#dcfce7" stroke="#22c55e" stroke-width="1"/>
  <text x="108" y="415" fill="#374151" font-size="11">F = FULL — compute normally</text>

  <rect x="80" y="428" width="20" height="20" rx="3" fill="#fef9c3" stroke="#eab308" stroke-width="1"/>
  <text x="108" y="443" fill="#374151" font-size="11">P = PARTIAL — apply element mask</text>

  <rect x="80" y="456" width="20" height="20" rx="3" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="108" y="471" fill="#374151" font-size="11">S = SKIP — don't load K,V!</text>

  <!-- Stats -->
  <text x="460" y="415" fill="#6b7280" font-size="11">6 of 16 SKIP</text>
  <text x="460" y="433" fill="#dc2626" font-weight="bold" font-size="11">~38% skipped</text>
</svg>'''))

# %%
T6 = 128
d6 = 64
bq6 = 32
bk6 = 32
tiles_q6 = T6 // bq6
tiles_kv6 = T6 // bk6

Q6 = jax.random.normal(jax.random.key(50), (T6, d6))
K6 = jax.random.normal(jax.random.key(51), (T6, d6))
V6 = jax.random.normal(jax.random.key(52), (T6, d6))

# --- Reference ---
def causal_attention_spec(Q, K, V):
    """Attention with causal mask."""
    d = Q.shape[-1]
    T = Q.shape[0]
    S = Q @ K.T / jnp.sqrt(d).astype(Q.dtype)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    S = jnp.where(mask, S, -jnp.inf)
    P = jax.nn.softmax(S, axis=-1)
    return P @ V


# --- Kernel ---
def causal_flash_kernel(
    q_ref, k_ref, v_ref, o_ref,
    acc_ref, m_ref, l_ref,
):
    """Flash attention with causal masking.

    Grid: (tiles_q6, tiles_kv6)

    You need to handle three cases:
    - SKIP: i * bq6 < kv * bk6 → do nothing
    - FULL: (i + 1) * bq6 > (kv + 1) * bk6 → normal flash attention
    - PARTIAL: diagonal block → apply causal mask to scores
    """
    i = pl.program_id(0)
    kv = pl.program_id(1)
    pass  # YOUR CODE HERE
    # 1. Init on kv == 0 (same as Puzzle 5)
    # 2. Determine if this block should be skipped
    # 3. @pl.when(should_compute): compute scores, apply mask if needed,
    #    do online softmax update
    # 4. Normalize on last kv block


# %%
check(causal_flash_kernel, causal_attention_spec, (Q6, K6, V6),
      grid=(tiles_q6, tiles_kv6),
      in_specs=[
          pl.BlockSpec((bq6, d6), lambda i, kv: (i, 0)),
          pl.BlockSpec((bk6, d6), lambda i, kv: (kv, 0)),
          pl.BlockSpec((bk6, d6), lambda i, kv: (kv, 0)),
      ],
      out_specs=pl.BlockSpec((bq6, d6), lambda i, kv: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((T6, d6), jnp.float32),
      scratch_shapes=[
          pltpu.VMEM((bq6, d6), jnp.float32),
          pltpu.VMEM((bq6,), jnp.float32),
          pltpu.VMEM((bq6,), jnp.float32),
      ],
      atol=1e-2, rtol=1e-2)

# %% [markdown]
# <details><summary>Hint 1 of 3 — Block classification</summary>
#
# ```python
# should_compute = (i * bq6 >= kv * bk6)    # not above diagonal
# is_full = ((i + 1) * bq6 > (kv + 1) * bk6)  # entirely below diagonal
# ```
# When `should_compute` is False, the block is fully masked — skip it.
# When `is_full` is True, no per-element mask needed.
# </details>
#
# <details><summary>Hint 2 of 3 — Applying the causal mask</summary>
#
# For partial (diagonal) blocks, create the mask with global indices:
# ```python
# q_idx = i * bq6 + jnp.arange(bq6)[:, None]
# kv_idx = kv * bk6 + jnp.arange(bk6)[None, :]
# causal_mask = q_idx >= kv_idx
# s = jnp.where(causal_mask, s, -jnp.inf)
# ```
# For full blocks, just use `s` as-is.
# You can unify: `s = jnp.where(causal_mask | is_full, s, -jnp.inf)`
# or use `lax.cond`.
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# def causal_flash_kernel(q_ref, k_ref, v_ref, o_ref,
#                          acc_ref, m_ref, l_ref):
#     i = pl.program_id(0)
#     kv = pl.program_id(1)
#
#     @pl.when(kv == 0)
#     def _init():
#         acc_ref[...] = jnp.zeros((bq6, d6), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq6,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq6,), dtype=jnp.float32)
#
#     should_compute = (i * bq6 >= kv * bk6)
#
#     @pl.when(should_compute)
#     def _compute():
#         q = q_ref[...]
#         k = k_ref[...]
#         v = v_ref[...]
#         s = q @ k.T / jnp.sqrt(d6).astype(jnp.float32)
#
#         # Apply causal mask for partial blocks
#         q_idx = i * bq6 + jnp.arange(bq6)[:, None]
#         kv_idx = kv * bk6 + jnp.arange(bk6)[None, :]
#         causal_mask = q_idx >= kv_idx
#         s = jnp.where(causal_mask, s, -jnp.inf)
#
#         m_tile = jnp.max(s, axis=-1)
#         m_new = jnp.maximum(m_ref[...], m_tile)
#         corr = jnp.exp(m_ref[...] - m_new)
#
#         acc_ref[...] = acc_ref[...] * corr[:, None]
#         p = jnp.exp(s - m_new[:, None])
#         l_ref[...] = l_ref[...] * corr + p.sum(axis=-1)
#         acc_ref[...] = acc_ref[...] + p @ v
#         m_ref[...] = m_new
#
#     @pl.when(kv == tiles_kv6 - 1)
#     def _norm():
#         o_ref[...] = acc_ref[...] / l_ref[...][:, None]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 7: Block-Sparse Masks and Prefetch Maps
#
# **Goal**: Generalize causal masking to **arbitrary block-sparse patterns**.
# Use a `block_mask` array to classify blocks and a `data_next` map to
# iterate only over non-skipped blocks.
#
# ### Theory
#
# Causal masking (Puzzle 6) hardcoded the skip/compute logic into the
# kernel. But production attention uses many patterns: causal, sliding
# window, local + global, block-sparse for long documents, etc. We need
# a **general mechanism**.
#
# **Splash attention** represents the mask as a precomputed **block_mask**
# array. For each Q block `i` and KV block `kv`, `block_mask[i, kv]`
# says:
#
# | Value | Meaning | Action |
# |-------|---------|--------|
# | 0 | SKIP | Don't load K,V — block is fully masked |
# | 1 | PARTIAL | Apply per-element mask (partial visibility) |
# | 2 | FULL | Compute normally (all positions visible) |
#
# In this puzzle, we keep the grid shape `(tiles_q, tiles_kv)` and use
# `block_mask` to decide what to do at each grid point — just like
# Puzzle 6, but reading the classification from an array instead of
# computing it.
#
# ```
#    block_mask (precomputed)
#           kv=0  kv=1  kv=2  kv=3
#         ┌──────┬──────┬──────┬──────┐
#    i=0  │  1   │  0   │  0   │  0   │     0 = SKIP
#         ├──────┼──────┼──────┼──────┤     1 = PARTIAL
#    i=1  │  2   │  1   │  0   │  0   │     2 = FULL
#         ├──────┼──────┼──────┼──────┤
#    i=2  │  2   │  2   │  1   │  0   │
#         ├──────┼──────┼──────┼──────┤
#    i=3  │  2   │  2   │  2   │  1   │
#         └──────┴──────┴──────┴──────┘
# ```
#
# For the mask itself on partial blocks, we precompute per-element mask
# arrays and store them in `partial_mask_blocks`. The kernel looks up which
# partial mask to use for diagonal blocks.
#
# **Why precompute the mask?** On TPU, branches are expensive. By
# precomputing block_mask on the host, the kernel just reads integers and
# acts accordingly — no complex logic inside the kernel.
#
# For this puzzle, we'll use `block_mask` as a regular input (passed via
# BlockSpec). In Puzzle 8, we'll upgrade to `PrefetchScalarGridSpec` for
# production-grade dispatch.

# %%
T7 = 128
d7 = 64
bq7 = 32
bk7 = 32
tiles_q7 = T7 // bq7
tiles_kv7 = T7 // bk7

Q7 = jax.random.normal(jax.random.key(60), (T7, d7))
K7 = jax.random.normal(jax.random.key(61), (T7, d7))
V7 = jax.random.normal(jax.random.key(62), (T7, d7))

# --- Build causal block_mask ---
# 0 = SKIP, 1 = PARTIAL (diagonal), 2 = FULL (below diagonal)
block_mask7 = jnp.zeros((tiles_q7, tiles_kv7), dtype=jnp.int32)
for qi in range(tiles_q7):
    for kvi in range(tiles_kv7):
        if qi * bq7 < kvi * bk7:
            block_mask7 = block_mask7.at[qi, kvi].set(0)    # SKIP
        elif (qi + 1) * bq7 > (kvi + 1) * bk7:
            block_mask7 = block_mask7.at[qi, kvi].set(2)    # FULL
        else:
            block_mask7 = block_mask7.at[qi, kvi].set(1)    # PARTIAL

print("block_mask7:")
print(block_mask7)

# --- Build partial mask blocks ---
# For each diagonal block, precompute the per-element causal mask.
# We store them as a list indexed by the diagonal block number.
num_partial7 = int(min(tiles_q7, tiles_kv7))
partial_masks7 = jnp.zeros((num_partial7, bq7, bk7), dtype=jnp.bool_)
partial_idx = 0
for qi in range(tiles_q7):
    for kvi in range(tiles_kv7):
        if int(block_mask7[qi, kvi]) == 1:
            q_idx = qi * bq7 + jnp.arange(bq7)[:, None]
            kv_idx = kvi * bk7 + jnp.arange(bk7)[None, :]
            partial_masks7 = partial_masks7.at[partial_idx].set(q_idx >= kv_idx)
            partial_idx += 1

print(f"\npartial_masks7 shape: {partial_masks7.shape} ({partial_idx} partial blocks)")


# --- Reference ---
def block_sparse_attention_spec(Q, K, V, block_mask, partial_masks):
    """Attention with block-sparse mask (reference implementation)."""
    d = Q.shape[-1]
    T = Q.shape[0]
    S = Q @ K.T / jnp.sqrt(d).astype(Q.dtype)
    # Reconstruct full mask from block_mask + partial_masks
    full_mask = jnp.zeros((T, T), dtype=jnp.bool_)
    pidx = 0
    for qi in range(tiles_q7):
        for kvi in range(tiles_kv7):
            bm = int(block_mask[qi, kvi])
            r0, r1 = qi * bq7, (qi + 1) * bq7
            c0, c1 = kvi * bk7, (kvi + 1) * bk7
            if bm == 2:
                full_mask = full_mask.at[r0:r1, c0:c1].set(True)
            elif bm == 1:
                full_mask = full_mask.at[r0:r1, c0:c1].set(partial_masks[pidx])
                pidx += 1
    S = jnp.where(full_mask, S, -jnp.inf)
    P = jax.nn.softmax(S, axis=-1)
    return P @ V


# --- Kernel ---
def block_sparse_flash_kernel(
    q_ref, k_ref, v_ref,
    block_mask_ref,        # (tiles_kv7,) — mask row for this Q block
    partial_masks_ref,     # (num_partial7, bq7, bk7) — all partial masks
    o_ref,
    acc_ref, m_ref, l_ref,
):
    """Flash attention with block-sparse mask from block_mask array.

    Grid: (tiles_q7, tiles_kv7)

    block_mask_ref: row of block_mask for current Q block
      - 0 = skip, 1 = partial, 2 = full
    partial_masks_ref: precomputed element masks for partial blocks
    """
    i = pl.program_id(0)
    kv = pl.program_id(1)
    pass  # YOUR CODE HERE
    # 1. Init on kv == 0
    # 2. Read block_mask_ref[kv] to get block type
    # 3. @pl.when(block_type > 0): compute attention
    #    - If block_type == 1: apply partial mask from partial_masks_ref
    #    - If block_type == 2: no mask needed (full visibility)
    # 4. Normalize on last kv block


# %%
expected7 = block_sparse_attention_spec(Q7, K7, V7, block_mask7, partial_masks7)

# For partial block indexing, we need to know which partial mask index
# corresponds to each diagonal block. For causal: block (i,i) is the
# i-th partial block.
# We pass block_mask row-by-row via BlockSpec.
actual7 = pl.pallas_call(
    block_sparse_flash_kernel,
    grid=(tiles_q7, tiles_kv7),
    in_specs=[
        pl.BlockSpec((bq7, d7), lambda i, kv: (i, 0)),              # Q
        pl.BlockSpec((bk7, d7), lambda i, kv: (kv, 0)),             # K
        pl.BlockSpec((bk7, d7), lambda i, kv: (kv, 0)),             # V
        pl.BlockSpec((tiles_kv7,), lambda i, kv: (i,)),             # block_mask row
        pl.BlockSpec(memory_space=pl.ANY),                           # partial_masks (full)
    ],
    out_specs=pl.BlockSpec((bq7, d7), lambda i, kv: (i, 0)),
    out_shape=jax.ShapeDtypeStruct((T7, d7), jnp.float32),
    scratch_shapes=[
        pltpu.VMEM((bq7, d7), jnp.float32),
        pltpu.VMEM((bq7,), jnp.float32),
        pltpu.VMEM((bq7,), jnp.float32),
    ],
    interpret=True,
)(Q7, K7, V7, block_mask7, partial_masks7)

if jnp.allclose(actual7, expected7, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual7.shape})")
else:
    diff7 = jnp.abs(actual7 - expected7)
    print(f"FAILED ✗  max error: {float(jnp.max(diff7)):.6f}")
    print(f"  Expected (first 2 rows):\n{expected7[:2, :8]}")
    print(f"  Got      (first 2 rows):\n{actual7[:2, :8]}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Reading the block mask</summary>
#
# `block_mask_ref` contains the entire mask row for this Q block. Read
# the entry for the current KV block:
# ```python
# block_type = block_mask_ref[kv]    # 0, 1, or 2
# should_compute = block_type > 0
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — Applying partial masks</summary>
#
# For causal masking, the partial block for Q block `i` is at index `i`
# in partial_masks_ref:
# ```python
# mask = partial_masks_ref[i]  # (bq7, bk7) boolean mask
# s = jnp.where(mask, s, -jnp.inf)
# ```
# For the full case, skip the mask step (or use `True` mask).
# Unify with: `is_partial = (block_type == 1)`
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# def block_sparse_flash_kernel(q_ref, k_ref, v_ref,
#                                block_mask_ref, partial_masks_ref,
#                                o_ref, acc_ref, m_ref, l_ref):
#     i = pl.program_id(0)
#     kv = pl.program_id(1)
#
#     @pl.when(kv == 0)
#     def _init():
#         acc_ref[...] = jnp.zeros((bq7, d7), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq7,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq7,), dtype=jnp.float32)
#
#     block_type = block_mask_ref[kv]
#     should_compute = block_type > 0
#
#     @pl.when(should_compute)
#     def _compute():
#         q = q_ref[...]
#         k = k_ref[...]
#         v = v_ref[...]
#         s = q @ k.T / jnp.sqrt(d7).astype(jnp.float32)
#
#         # Apply partial mask for diagonal blocks
#         is_partial = (block_type == 1)
#         mask = partial_masks_ref[i]
#         # Use mask only when partial; for full blocks, keep all scores
#         s = jnp.where(is_partial & ~mask, -jnp.inf, s)
#
#         m_tile = jnp.max(s, axis=-1)
#         m_new = jnp.maximum(m_ref[...], m_tile)
#         corr = jnp.exp(m_ref[...] - m_new)
#
#         acc_ref[...] = acc_ref[...] * corr[:, None]
#         p = jnp.exp(s - m_new[:, None])
#         l_ref[...] = l_ref[...] * corr + p.sum(axis=-1)
#         acc_ref[...] = acc_ref[...] + p @ v
#         m_ref[...] = m_new
#
#     @pl.when(kv == tiles_kv7 - 1)
#     def _norm():
#         o_ref[...] = acc_ref[...] / l_ref[...][:, None]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 8: Splash Attention
#
# **Goal**: Put it all together — build the full **splash attention** kernel
# with block-sparse masks and `PrefetchScalarGridSpec` for efficient
# block dispatch.
#
# ### Theory
#
# In Puzzle 7, we iterated over ALL KV blocks and checked the mask at
# runtime. But if 80% of blocks are SKIP, we're wasting 80% of grid
# iterations just to check a flag and do nothing. On TPU, each grid
# iteration has overhead even if the kernel body is skipped.
#
# **Splash attention** solves this with **compacted iteration**: instead of
# iterating over all `tiles_kv` blocks for each Q block, we iterate only
# over the **non-skip** blocks. A precomputed `data_next` array tells
# each grid iteration which KV block to process.
#
# ```
#    Regular grid (Puzzle 7):          Compacted grid (Puzzle 8):
#
#    kv: 0  1  2  3                    step: 0    1
#    ┌──┬──┬──┬──┐                     ┌──────┬──────┐
#    │ P│ S│ S│ S│  → 4 iterations     │ kv=0 │      │  → 2 iterations
#    ├──┼──┼──┼──┤                     │(PART)│      │    (skip nothing!)
#    │ F│ P│ S│ S│  → 4 iterations     ├──────┼──────┤
#    ├──┼──┼──┼──┤                     │ kv=0 │ kv=1 │  → 2 iterations
#    │ F│ F│ P│ S│  → 4 iterations     │(FULL)│(PART)│
#    ├──┼──┼──┼──┤                     ├──────┼──────┤
#    │ F│ F│ F│ P│  → 4 iterations     │ kv=0 │ kv=1 │
#    └──┴──┴──┴──┘                     │(FULL)│(FULL)│  etc.
#    16 total iterations               └──────┴──────┘
#    (10 are SKIP!)                    10 total iterations
#                                      (0 wasted!)
# ```
#
# **How it works**: We precompute:
# - `data_next[i, step]`: which KV block index to load at step `step`
#    for Q block `i`
# - `mask_next[i, step]`: block type (0=skip, 1=partial, 2=full) at this step
# - `grid_width`: max number of non-skip blocks across all Q blocks
#
# The grid becomes `(tiles_q, grid_width)` instead of `(tiles_q, tiles_kv)`.
#
# To pass `data_next` and `mask_next` to the kernel, we use
# **`PrefetchScalarGridSpec`** (from ragged_dot.py) — small metadata arrays
# are loaded into scalar memory (SMEM) and accessible to both index maps
# and the kernel body. The index maps use `data_next` to route K,V loads
# to the correct blocks.
#
# This is the production pattern from `jax.experimental.pallas.ops.tpu.splash_attention`.

# %%
#@title Diagram: Compacted Iteration (Splash)
from IPython.display import SVG, display
display(SVG(data='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 420" font-family="monospace" font-size="12">
  <rect width="780" height="420" fill="white"/>

  <!-- Title -->
  <text x="390" y="25" text-anchor="middle" fill="#111827" font-weight="bold" font-size="15">Splash Attention: Compacted Iteration</text>

  <!-- LEFT: Regular grid -->
  <text x="150" y="55" text-anchor="middle" fill="#6b7280" font-weight="bold" font-size="12">Regular Grid (Puzzle 7)</text>

  <!-- Left grid - 4×4 causal -->
  <!-- Row 0: P S S S -->
  <rect x="50" y="70" width="45" height="45" rx="2" fill="#fef9c3" stroke="#eab308" stroke-width="1"/>
  <text x="72" y="98" text-anchor="middle" fill="#854d0e" font-size="10">P</text>
  <rect x="100" y="70" width="45" height="45" rx="2" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="122" y="98" text-anchor="middle" fill="#b91c1c" font-size="10">S</text>
  <rect x="150" y="70" width="45" height="45" rx="2" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="172" y="98" text-anchor="middle" fill="#b91c1c" font-size="10">S</text>
  <rect x="200" y="70" width="45" height="45" rx="2" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="222" y="98" text-anchor="middle" fill="#b91c1c" font-size="10">S</text>

  <!-- Row 1: F P S S -->
  <rect x="50" y="120" width="45" height="45" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1"/>
  <text x="72" y="148" text-anchor="middle" fill="#166534" font-size="10">F</text>
  <rect x="100" y="120" width="45" height="45" rx="2" fill="#fef9c3" stroke="#eab308" stroke-width="1"/>
  <text x="122" y="148" text-anchor="middle" fill="#854d0e" font-size="10">P</text>
  <rect x="150" y="120" width="45" height="45" rx="2" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="172" y="148" text-anchor="middle" fill="#b91c1c" font-size="10">S</text>
  <rect x="200" y="120" width="45" height="45" rx="2" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="222" y="148" text-anchor="middle" fill="#b91c1c" font-size="10">S</text>

  <!-- Row 2: F F P S -->
  <rect x="50" y="170" width="45" height="45" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1"/>
  <text x="72" y="198" text-anchor="middle" fill="#166534" font-size="10">F</text>
  <rect x="100" y="170" width="45" height="45" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1"/>
  <text x="122" y="198" text-anchor="middle" fill="#166534" font-size="10">F</text>
  <rect x="150" y="170" width="45" height="45" rx="2" fill="#fef9c3" stroke="#eab308" stroke-width="1"/>
  <text x="172" y="198" text-anchor="middle" fill="#854d0e" font-size="10">P</text>
  <rect x="200" y="170" width="45" height="45" rx="2" fill="#fee2e2" stroke="#fca5a5" stroke-width="1"/>
  <text x="222" y="198" text-anchor="middle" fill="#b91c1c" font-size="10">S</text>

  <!-- Row 3: F F F P -->
  <rect x="50" y="220" width="45" height="45" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1"/>
  <text x="72" y="248" text-anchor="middle" fill="#166534" font-size="10">F</text>
  <rect x="100" y="220" width="45" height="45" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1"/>
  <text x="122" y="248" text-anchor="middle" fill="#166534" font-size="10">F</text>
  <rect x="150" y="220" width="45" height="45" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1"/>
  <text x="172" y="248" text-anchor="middle" fill="#166534" font-size="10">F</text>
  <rect x="200" y="220" width="45" height="45" rx="2" fill="#fef9c3" stroke="#eab308" stroke-width="1"/>
  <text x="222" y="248" text-anchor="middle" fill="#854d0e" font-size="10">P</text>

  <!-- Left label -->
  <text x="150" y="290" text-anchor="middle" fill="#b91c1c" font-size="11">16 iterations</text>
  <text x="150" y="305" text-anchor="middle" fill="#b91c1c" font-weight="bold" font-size="11">6 wasted on SKIP</text>

  <!-- BIG ARROW -->
  <line x1="270" y1="165" x2="360" y2="165" stroke="#7c3aed" stroke-width="3" marker-end="url(#bigarr)"/>
  <text x="315" y="150" text-anchor="middle" fill="#7c3aed" font-weight="bold" font-size="11">data_next</text>
  <text x="315" y="190" text-anchor="middle" fill="#7c3aed" font-size="10">maps steps</text>
  <text x="315" y="205" text-anchor="middle" fill="#7c3aed" font-size="10">→ KV blocks</text>

  <!-- RIGHT: Compacted grid -->
  <text x="560" y="55" text-anchor="middle" fill="#6b7280" font-weight="bold" font-size="12">Compacted Grid (Splash)</text>

  <!-- Row labels -->
  <text x="395" y="98" fill="#6b7280" font-size="10">i=0</text>
  <text x="395" y="148" fill="#6b7280" font-size="10">i=1</text>
  <text x="395" y="198" fill="#6b7280" font-size="10">i=2</text>
  <text x="395" y="248" fill="#6b7280" font-size="10">i=3</text>

  <!-- Step labels -->
  <text x="445" y="68" text-anchor="middle" fill="#6b7280" font-size="9">step 0</text>
  <text x="515" y="68" text-anchor="middle" fill="#6b7280" font-size="9">step 1</text>
  <text x="585" y="68" text-anchor="middle" fill="#6b7280" font-size="9">step 2</text>
  <text x="655" y="68" text-anchor="middle" fill="#6b7280" font-size="9">step 3</text>

  <!-- Compacted Row 0: 1 block -->
  <rect x="420" y="75" width="50" height="40" rx="2" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <text x="445" y="98" text-anchor="middle" fill="#854d0e" font-size="9">kv=0</text>

  <!-- Compacted Row 1: 2 blocks -->
  <rect x="420" y="125" width="50" height="40" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="445" y="148" text-anchor="middle" fill="#166534" font-size="9">kv=0</text>
  <rect x="490" y="125" width="50" height="40" rx="2" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <text x="515" y="148" text-anchor="middle" fill="#854d0e" font-size="9">kv=1</text>

  <!-- Compacted Row 2: 3 blocks -->
  <rect x="420" y="175" width="50" height="40" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="445" y="198" text-anchor="middle" fill="#166534" font-size="9">kv=0</text>
  <rect x="490" y="175" width="50" height="40" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="515" y="198" text-anchor="middle" fill="#166534" font-size="9">kv=1</text>
  <rect x="560" y="175" width="50" height="40" rx="2" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <text x="585" y="198" text-anchor="middle" fill="#854d0e" font-size="9">kv=2</text>

  <!-- Compacted Row 3: 4 blocks -->
  <rect x="420" y="225" width="50" height="40" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="445" y="248" text-anchor="middle" fill="#166534" font-size="9">kv=0</text>
  <rect x="490" y="225" width="50" height="40" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="515" y="248" text-anchor="middle" fill="#166534" font-size="9">kv=1</text>
  <rect x="560" y="225" width="50" height="40" rx="2" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>
  <text x="585" y="248" text-anchor="middle" fill="#166534" font-size="9">kv=2</text>
  <rect x="630" y="225" width="50" height="40" rx="2" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <text x="655" y="248" text-anchor="middle" fill="#854d0e" font-size="9">kv=3</text>

  <!-- Right label -->
  <text x="560" y="290" text-anchor="middle" fill="#166534" font-size="11">10 iterations</text>
  <text x="560" y="305" text-anchor="middle" fill="#166534" font-weight="bold" font-size="11">0 wasted!</text>

  <!-- Bottom explanation -->
  <rect x="140" y="330" width="500" height="75" rx="8" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.5"/>
  <text x="390" y="355" text-anchor="middle" fill="#5b21b6" font-weight="bold" font-size="12">PrefetchScalarGridSpec</text>
  <text x="390" y="375" text-anchor="middle" fill="#6d28d9" font-size="11">data_next[i, step] → which KV block to load at each step</text>
  <text x="390" y="393" text-anchor="middle" fill="#6d28d9" font-size="11">Index maps use data_next to route K,V to the right block</text>

  <defs>
    <marker id="bigarr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#7c3aed"/>
    </marker>
  </defs>
</svg>'''))

# %%
T8 = 128
d8 = 64
bq8 = 32
bk8 = 32
tiles_q8 = T8 // bq8
tiles_kv8 = T8 // bk8

Q8 = jax.random.normal(jax.random.key(70), (T8, d8))
K8 = jax.random.normal(jax.random.key(71), (T8, d8))
V8 = jax.random.normal(jax.random.key(72), (T8, d8))

# --- Build causal block_mask (same as Puzzle 7) ---
block_mask8 = jnp.zeros((tiles_q8, tiles_kv8), dtype=jnp.int32)
for qi in range(tiles_q8):
    for kvi in range(tiles_kv8):
        if qi * bq8 < kvi * bk8:
            block_mask8 = block_mask8.at[qi, kvi].set(0)
        elif (qi + 1) * bq8 > (kvi + 1) * bk8:
            block_mask8 = block_mask8.at[qi, kvi].set(2)
        else:
            block_mask8 = block_mask8.at[qi, kvi].set(1)

# --- Build compacted iteration maps ---
# For each Q block, list the non-skip KV blocks in order
grid_width8 = 0
for qi in range(tiles_q8):
    count = int(jnp.sum(block_mask8[qi] > 0))
    grid_width8 = max(grid_width8, count)

# data_next[i, step] = which KV block to process at step `step`
# mask_next[i, step] = block type at that step
data_next8 = jnp.zeros((tiles_q8, grid_width8), dtype=jnp.int32)
mask_next8 = jnp.zeros((tiles_q8, grid_width8), dtype=jnp.int32)

for qi in range(tiles_q8):
    step = 0
    for kvi in range(tiles_kv8):
        if int(block_mask8[qi, kvi]) > 0:
            data_next8 = data_next8.at[qi, step].set(kvi)
            mask_next8 = mask_next8.at[qi, step].set(int(block_mask8[qi, kvi]))
            step += 1

print(f"block_mask8:\n{block_mask8}")
print(f"\ndata_next8 (kv indices per step):\n{data_next8}")
print(f"\nmask_next8 (block types per step):\n{mask_next8}")
print(f"\ngrid_width8: {grid_width8} (max non-skip blocks per Q block)")

# --- Build partial masks ---
num_partial8 = int(min(tiles_q8, tiles_kv8))
partial_masks8 = jnp.zeros((num_partial8, bq8, bk8), dtype=jnp.bool_)
pidx8 = 0
for qi in range(tiles_q8):
    for kvi in range(tiles_kv8):
        if int(block_mask8[qi, kvi]) == 1:
            q_idx = qi * bq8 + jnp.arange(bq8)[:, None]
            kv_idx = kvi * bk8 + jnp.arange(bk8)[None, :]
            partial_masks8 = partial_masks8.at[pidx8].set(q_idx >= kv_idx)
            pidx8 += 1


# --- Reference ---
def splash_attention_spec(Q, K, V, data_next, mask_next, partial_masks):
    """Same as causal attention — splash is an optimization, not a different computation."""
    d = Q.shape[-1]
    T = Q.shape[0]
    S = Q @ K.T / jnp.sqrt(d).astype(Q.dtype)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    S = jnp.where(mask, S, -jnp.inf)
    P = jax.nn.softmax(S, axis=-1)
    return P @ V


# --- Index maps for PrefetchScalarGridSpec ---
# data_next_ref and mask_next_ref are scalar-prefetched.
# Index maps receive them as extra arguments.
def q_index_map(i, step, data_next_ref, mask_next_ref):
    return (i, 0)

def kv_index_map(i, step, data_next_ref, mask_next_ref):
    """Use data_next to look up which KV block to load."""
    kv_block = data_next_ref[i, step]
    return (kv_block, 0)

def o_index_map(i, step, data_next_ref, mask_next_ref):
    return (i, 0)


# --- Kernel ---
def splash_attention_kernel(
    data_next_ref,       # (tiles_q8, grid_width8) — scalar prefetch
    mask_next_ref,       # (tiles_q8, grid_width8) — scalar prefetch
    q_ref,               # (bq8, d8) — Q block
    k_ref,               # (bk8, d8) — KV block (routed by data_next)
    v_ref,               # (bk8, d8) — KV block (routed by data_next)
    partial_masks_ref,   # (num_partial8, bq8, bk8) — all partial masks
    o_ref,               # (bq8, d8) — output
    acc_ref,             # (bq8, d8) — scratch
    m_ref,               # (bq8,) — scratch
    l_ref,               # (bq8,) — scratch
):
    """Splash attention kernel with compacted block-sparse iteration.

    Grid: (tiles_q8, grid_width8)
      - program_id(0) = i: Q block index
      - program_id(1) = step: compacted iteration step

    Scalar-prefetched: data_next (kv block indices), mask_next (block types)
    Index maps use data_next to route K,V loads to the correct block.
    """
    i = pl.program_id(0)
    step = pl.program_id(1)
    pass  # YOUR CODE HERE
    # 1. Init on step == 0
    # 2. Read block type from mask_next_ref[i, step]
    # 3. @pl.when(block_type > 0): flash attention with optional mask
    #    - K,V are already routed to the right block by the index map!
    #    - For partial blocks, look up mask from partial_masks_ref[i]
    # 4. Normalize on step == grid_width8 - 1


# %%
expected8 = splash_attention_spec(Q8, K8, V8, data_next8, mask_next8, partial_masks8)

actual8 = pl.pallas_call(
    splash_attention_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,   # data_next and mask_next
        in_specs=[
            pl.BlockSpec((bq8, d8), q_index_map),              # Q
            pl.BlockSpec((bk8, d8), kv_index_map),             # K (routed!)
            pl.BlockSpec((bk8, d8), kv_index_map),             # V (routed!)
            pl.BlockSpec(memory_space=pl.ANY),                  # partial_masks (full)
        ],
        out_specs=pl.BlockSpec((bq8, d8), o_index_map),
        scratch_shapes=[
            pltpu.VMEM((bq8, d8), jnp.float32),   # acc
            pltpu.VMEM((bq8,), jnp.float32),       # m
            pltpu.VMEM((bq8,), jnp.float32),       # l
        ],
        grid=(tiles_q8, grid_width8),
    ),
    interpret=True,
)(data_next8, mask_next8, Q8, K8, V8, partial_masks8)

if jnp.allclose(actual8, expected8, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual8.shape})")
    print(f"\n🎉 Congratulations! You've built splash attention from scratch!")
    print(f"   Grid: ({tiles_q8}, {grid_width8}) instead of ({tiles_q8}, {tiles_kv8})")
    print(f"   Skipped {tiles_q8 * tiles_kv8 - tiles_q8 * grid_width8} "
          f"of {tiles_q8 * tiles_kv8} total block pairs")
else:
    diff8 = jnp.abs(actual8 - expected8)
    print(f"FAILED ✗  max error: {float(jnp.max(diff8)):.6f}")
    print(f"  Expected (first 2 rows):\n{expected8[:2, :8]}")
    print(f"  Got      (first 2 rows):\n{actual8[:2, :8]}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — The kernel is similar to Puzzle 7</summary>
#
# The main differences from Puzzle 7:
# - K,V are already loaded for the correct block (index maps handle routing)
# - Read block type from `mask_next_ref[i, step]` instead of `block_mask_ref[kv]`
# - Init/normalize on `step == 0` and `step == grid_width8 - 1`
# - No wasted iterations — every step does useful work (unless padded)
# </details>
#
# <details><summary>Hint 2 of 3 — Watch out for padding</summary>
#
# If a Q block has fewer non-skip blocks than `grid_width`, the remaining
# steps have `mask_next = 0` (skip). The `@pl.when(block_type > 0)` guard
# handles this automatically.
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# def splash_attention_kernel(data_next_ref, mask_next_ref,
#                              q_ref, k_ref, v_ref, partial_masks_ref,
#                              o_ref, acc_ref, m_ref, l_ref):
#     i = pl.program_id(0)
#     step = pl.program_id(1)
#
#     @pl.when(step == 0)
#     def _init():
#         acc_ref[...] = jnp.zeros((bq8, d8), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq8,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq8,), dtype=jnp.float32)
#
#     block_type = mask_next_ref[i, step]
#     should_compute = block_type > 0
#
#     @pl.when(should_compute)
#     def _compute():
#         q = q_ref[...]
#         k = k_ref[...]
#         v = v_ref[...]
#         s = q @ k.T / jnp.sqrt(d8).astype(jnp.float32)
#
#         is_partial = (block_type == 1)
#         mask = partial_masks_ref[i]
#         s = jnp.where(is_partial & ~mask, -jnp.inf, s)
#
#         m_tile = jnp.max(s, axis=-1)
#         m_new = jnp.maximum(m_ref[...], m_tile)
#         corr = jnp.exp(m_ref[...] - m_new)
#
#         acc_ref[...] = acc_ref[...] * corr[:, None]
#         p = jnp.exp(s - m_new[:, None])
#         l_ref[...] = l_ref[...] * corr + p.sum(axis=-1)
#         acc_ref[...] = acc_ref[...] + p @ v
#         m_ref[...] = m_new
#
#     @pl.when(step == grid_width8 - 1)
#     def _norm():
#         o_ref[...] = acc_ref[...] / l_ref[...][:, None]
# ```
# </details>

# %% [markdown]
# ---
# # TODO: Backward Pass
#
# The backward pass for splash attention is what makes it truly
# production-ready. Here's what those puzzles would cover:
#
# ### dQ Kernel
# For each Q block, iterate over KV blocks (same pattern as forward) and
# compute gradients with respect to Q:
# - **Recompute** the score matrix S from Q and K — don't store it!
#   This is the key memory optimization from the Flash Attention paper.
# - Use dO (the upstream gradient) and the saved statistics (m, l) from
#   the forward pass to compute dP, then dS, then dQ.
# - Accumulate dQ contributions across all KV blocks.
#
# ### dKV Kernel
# For each KV block, iterate over **Q blocks** (the reverse direction):
# - Compute S^T contributions and accumulate dK and dV.
# - This is the "transpose" of the forward kernel — instead of iterating
#   KV blocks for a fixed Q block, we iterate Q blocks for a fixed KV block.
#
# ### Block-Sparse Backward
# Skip the same blocks as the forward pass! The block_mask and data_next
# maps work in both directions:
# - dQ kernel: same mask as forward (skip KV blocks)
# - dKV kernel: needs a **transposed** version of data_next (skip Q blocks)
#
# The backward pass roughly doubles the kernel code, but the patterns are
# the same: online accumulation with `@pl.when` guards, scratch memory
# for accumulators, and block-sparse dispatch via precomputed maps.
#
# See `jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel`
# for the production implementation.

# %% [markdown]
# ---
# ## 🎉 Summary
#
# You've built splash attention from the ground up:
#
# | Puzzle | Concept | Key Insight |
# |--------|---------|-------------|
# | 1 | Dot-product attention | The O(T²) score matrix problem |
# | 2 | Tiled softmax | Computing max and sum_exp across tiles |
# | 3 | Online softmax | Single-pass with correction factor `exp(m_old - m_new)` |
# | 4 | Tiled attention (1 Q block) | Combining online softmax with tiled matmul |
# | 5 | Flash attention | Grid over all Q blocks — O(T) memory |
# | 6 | Causal masking | Block classification: FULL / PARTIAL / SKIP |
# | 7 | Block-sparse masks | Precomputed block_mask + partial_mask arrays |
# | 8 | Splash attention | Compacted iteration via data_next + PrefetchScalarGridSpec |
#
# **What makes splash attention "splash"?** The compacted iteration from
# Puzzle 8. Instead of checking a mask at every grid point, we precompute
# which blocks to visit and only iterate over those. For attention patterns
# with significant sparsity (causal, sliding window, block-sparse), this
# eliminates wasted computation entirely.
#
# **Next steps**:
# - Try changing the block_mask in Puzzle 8 to a **sliding window** pattern
# - Read the [production source](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention) — you now understand every concept it uses
# - Implement the backward pass (see TODO above) for full training support
# - Add multi-head support: it's just `jax.vmap` over the head dimension!
