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
#
# Collapse this section (› arrow next to the heading), then ▶ **Run all cells
# in section** from the menu to get everything ready in one click.

# %%
# !pip install -q jax

# %%
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
print(f"JAX {jax.__version__}")


# %%
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


# %%
def attention_spec(Q, K, V):
    """Standard dot-product attention: softmax(QK^T / sqrt(H)) @ V"""
    H = Q.shape[-1]
    S = jnp.einsum('th,sh->ts', Q, K) / jnp.sqrt(H).astype(Q.dtype)
    P = jax.nn.softmax(S, axis=-1)
    return jnp.einsum('ts,sh->th', P, V)


# %% [markdown]
# ---
# # Part I: Flash Attention (Puzzles 1–5)

# %% [markdown]
# ---
# ## Puzzle 1: Dot-Product Attention
#
# **Goal**: Implement the standard attention equation in pure JAX (no Pallas
# yet). This gives you hands-on familiarity with the formula before we start
# tiling it.
#
# ### Theory
#
# Attention maps a query against a set of key-value pairs:
#
# $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{H}}\right) V$$
#
# where $Q, K, V \in \mathbb{R}^{T \times H}$, $T$ is the sequence length
# and $H$ is the head dimension (following the naming convention from
# [How to Scale Your Model](https://jax-ml.github.io/scaling-book/transformers/)).
#
# The score matrix $S = Q K^T / \sqrt{H}$ has shape $(T, T)$ — that's the
# **O(T²) memory bottleneck** we'll learn to eliminate. For a 4K-token
# sequence with 64-dim heads, $S$ alone is 64 MB in float32. At 128K tokens
# (common in modern LLMs), it would be **64 GB**. Clearly we can't
# materialize this matrix.
#
# ```
# Q (T×H)    K^T (H×T)      S (T×T)         P (T×T)       V (T×H)   O (T×H)
# ┌──────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌──────┐  ┌──────┐
# │      │  │          │  │           │  │ softmax   │  │      │  │      │
# │      │@ │          │= │  S / √H   │→ │  rows     │@ │      │= │  O   │
# │      │  │          │  │           │  │           │  │      │  │      │
# └──────┘  └──────────┘  └───────────┘  └───────────┘  └──────┘  └──────┘
#  T × H      H × T         T × T          T × T        T × H     T × H
#                          ← O(T²) memory! →
# ```
#
# Let's start by implementing this naive version, then spend the rest of the
# notebook learning to **never materialize S**.


# %%
T1 = 128    # sequence length
H1 = 64     # head dimension

Q1 = jax.random.normal(jax.random.key(0), (T1, H1))
K1 = jax.random.normal(jax.random.key(1), (T1, H1))
V1 = jax.random.normal(jax.random.key(2), (T1, H1))


# --- Your implementation ---
def my_attention(Q, K, V):
    """Implement: softmax(QK^T / sqrt(H)) V

    Q: (T, H), K: (T, H), V: (T, H) → output: (T, H)

    Steps:
      1. Compute scores S = QK^T / sqrt(H) using jnp.einsum  → (T, T)
      2. Apply softmax along the last axis (over keys)        → (T, T)
      3. Multiply the attention weights P by V via einsum     → (T, H)
    """
    # YOUR CODE HERE


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
# <details><summary>Hint 1 of 2 — Which JAX functions?</summary>
#
# You need three operations:
# - `jnp.einsum('th,sh->ts', Q, K)` contracts over H → scores `(T, T)`
# - `jax.nn.softmax(..., axis=-1)` normalizes rows
# - `jnp.einsum('ts,sh->th', P, V)` contracts over S → output `(T, H)`
# Don't forget to scale scores by `1 / jnp.sqrt(H)`
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# def my_attention(Q, K, V):
#     H = Q.shape[-1]
#     S = jnp.einsum('th,sh->ts', Q, K) / jnp.sqrt(H).astype(Q.dtype)
#     P = jax.nn.softmax(S, axis=-1)
#     return jnp.einsum('ts,sh->th', P, V)
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 2: Tiled Softmax — The Multi-Pass Problem
#
# **Goal**: Compute `softmax(x)` for a long vector using **two Pallas
# kernels** — one to find the global max, one to sum the exponentials —
# then normalize. This is the straightforward approach and it shows why
# we'll need something better.
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
# For now, let's implement the honest multi-pass version with **two
# separate kernels**:
#
# **Kernel 1 — `tiled_max_kernel`**: Tile over x to find the global max.
# Uses the init/accumulate pattern from basics.py Puzzle 7:
# `@pl.when(k == 0)` initializes to `-inf`, then every tile updates the
# running max uniformly.
#
# **Kernel 2 — `tiled_sumexp_kernel`**: Tile over x again, now that we
# know `m`, to compute `l = sum(exp(x - m))`. This kernel receives `m`
# as an input (it was computed by kernel 1).
#
# The key limitation: kernel 2 **cannot start until kernel 1 finishes**
# because it needs the global max. This sequential dependency is what
# forces multiple passes.

# %%
N2 = 512          # vector length
bn2 = 128         # tile size
tiles_k2 = N2 // bn2

# --- Reference ---
def softmax_spec(x):
    """x: (N2,) → (N2,)"""
    return jax.nn.softmax(x)


# --- Kernel 1: find global max via tiled reduction ---
def tiled_max_kernel(x_ref, m_ref):
    """Tile over x to compute global max m.

    x_ref: (bn2,) — one tile of x
    m_ref: ()     — running global max (scalar output)

    Grid: (tiles_k2,) — iterates over tiles of x
    """
    k = pl.program_id(0)
    # YOUR CODE HERE
    # 1. On first tile (k == 0): initialize m to -infinity
    # 2. On ALL tiles: m = max(m, max(tile))


# --- Kernel 2: compute sum of exponentials (needs global max m) ---
def tiled_sumexp_kernel(x_ref, m_ref, l_ref):
    """Tile over x to compute l = sum(exp(x - m)), given global max m.

    x_ref: (bn2,) — one tile of x
    m_ref: ()     — global max (input, already computed by kernel 1)
    l_ref: ()     — running sum of exp(x - m) (scalar output)

    Grid: (tiles_k2,) — iterates over tiles of x
    """
    k = pl.program_id(0)
    # YOUR CODE HERE
    # 1. On first tile (k == 0): initialize l to 0
    # 2. On ALL tiles: l += sum(exp(tile - m))


# %%
x2 = jax.random.uniform(jax.random.key(10), (N2,), minval=-5.0, maxval=-1.0)

# Pass 1: find global max
m2 = pl.pallas_call(
    tiled_max_kernel,
    grid=(tiles_k2,),
    in_specs=[pl.BlockSpec((bn2,), lambda k: (k,))],
    out_specs=pl.BlockSpec(memory_space=pl.ANY),
    out_shape=jax.ShapeDtypeStruct((), jnp.float32),
    interpret=True,
)(x2)

# Pass 2: compute sum(exp(x - m)) — needs m from pass 1
l2 = pl.pallas_call(
    tiled_sumexp_kernel,
    grid=(tiles_k2,),
    in_specs=[
        pl.BlockSpec((bn2,), lambda k: (k,)),  # x: tiled
        pl.BlockSpec(memory_space=pl.ANY),       # m: scalar, no blocking
    ],
    out_specs=pl.BlockSpec(memory_space=pl.ANY),
    out_shape=jax.ShapeDtypeStruct((), jnp.float32),
    interpret=True,
)(x2, m2)

# Pass 3: normalize (just JAX, no kernel needed)
softmax2 = jnp.exp(x2 - m2) / l2

expected_m2 = jnp.max(x2)
expected_l2 = jnp.sum(jnp.exp(x2 - expected_m2))
expected2 = softmax_spec(x2)
m_ok = jnp.allclose(m2, expected_m2, atol=1e-3)
l_ok = jnp.allclose(l2, expected_l2, atol=1e-3)
s_ok = jnp.allclose(softmax2, expected2, atol=1e-3)
if m_ok and l_ok and s_ok:
    print(f"PASSED ✓  (m={float(m2):.3f}, l={float(l2):.3f})")
else:
    if not m_ok:
        print(f"FAILED ✗  m={float(m2):.3f} (expected {float(expected_m2):.3f})")
    if not l_ok:
        print(f"FAILED ✗  l={float(l2):.3f} (expected {float(expected_l2):.3f})")
    if not s_ok:
        print(f"FAILED ✗  softmax max error: {float(jnp.max(jnp.abs(softmax2 - expected2))):.6f}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — tiled_max_kernel</summary>
#
# Initialize to the identity element for max (`-inf`), then the same update
# runs on every tile — same pattern as basics.py Puzzle 7:
# ```python
# @pl.when(k == 0)
# def _():
#     m_ref[...] = jnp.float32(-jnp.inf)
# m_ref[...] = jnp.maximum(m_ref[...], jnp.max(x_ref[...]))
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — tiled_sumexp_kernel</summary>
#
# Initialize to 0, then accumulate uniformly:
# ```python
# @pl.when(k == 0)
# def _():
#     l_ref[...] = jnp.float32(0.0)
# l_ref[...] += jnp.sum(jnp.exp(x_ref[...] - m_ref[...]))
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# def tiled_max_kernel(x_ref, m_ref):
#     k = pl.program_id(0)
#     @pl.when(k == 0)
#     def _():
#         m_ref[...] = jnp.float32(-jnp.inf)
#     m_ref[...] = jnp.maximum(m_ref[...], jnp.max(x_ref[...]))
#
#
# def tiled_sumexp_kernel(x_ref, m_ref, l_ref):
#     k = pl.program_id(0)
#     @pl.when(k == 0)
#     def _():
#         l_ref[...] = jnp.float32(0.0)
#     l_ref[...] += jnp.sum(jnp.exp(x_ref[...] - m_ref[...]))
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 3: Online Softmax — One Pass to Rule Them All
#
# **Goal**: Compute `m` and `l` in a **single kernel** by maintaining
# running statistics that self-correct as new tiles arrive.
#
# ### Theory
#
# In Puzzle 2 we needed two separate kernels — one for max, one for
# sum_exp — because `sum(exp(x - m))` requires knowing the global max
# first. Two kernel launches means two full passes over the data from HBM.
#
# **Online softmax** (Milakov & Gimelshein, 2018) is THE breakthrough
# that makes flash attention possible. The key idea: what if we compute
# `sum(exp(x - m))` *while we're still discovering the max*? When a new
# tile reveals a bigger max, we **correct** the running sum instead of
# starting over:
#
# ```
# exp(x - m_old) · exp(m_old - m_new) = exp(x - m_new)
# ```
#
# This **correction factor** `exp(m_old - m_new)` retroactively fixes
# all previous exponentials without re-reading the data. It turns two
# sequential passes into one:
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
# Notice the initialization: `m = -∞` and `ℓ = 0`. On the first tile,
# `correction = exp(-∞ - m_new) = 0`, so the old sum (0) gets zeroed out
# and we're left with just the new exponentials. The same update rule
# handles all tiles uniformly — no `k == 0` vs `k > 0` branching needed.
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
# After processing all tiles, `m` is the global max and `ℓ` is
# `sum(exp(x - m))` — exactly what softmax needs. One kernel, one pass.
#
# **Your task**: Write a single kernel that computes both `m` and `ℓ`,
# then use them to compute the final softmax output.


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
    # YOUR CODE HERE
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
# For a Q block of shape `(bq, H)`, we iterate over KV blocks:
#
# ```
#   Q block       K blocks (iterate →)      Output
#   ┌──────┐     ┌──────┬──────┬──────┬──────┐     ┌──────┐
#   │      │     │      │      │      │      │     │      │
#   │ bq×H │  @  │ bk×H │ bk×H │ bk×H │ bk×H │  →  │ bq×H │
#   │      │     │      │      │      │      │     │      │
#   └──────┘     └──────┴──────┴──────┴──────┘     └──────┘
#                 kv=0    kv=1   kv=2   kv=3
# ```
#
# For each KV block, the kernel:
# 1. Computes scores: `s = einsum('qh,kh->qk', Q_block, K_block) / sqrt(H)` → `(bq, bk)`
# 2. Updates online softmax stats `(m, ℓ)` per row
# 3. **Corrects** the running output accumulator for the new max
# 4. Adds new contribution: `acc += einsum('qk,kh->qh', P_block, V_block)`
# 5. After last KV block: normalizes by `1/ℓ`
#
# The key insight is step 3: when the max changes, we must **rescale**
# the entire accumulator. Without this correction, outputs from earlier
# KV blocks would have the wrong scale:
#
# ```python
# correction = exp(m_old - m_new)     # (bq,) per-row correction
# acc = acc * correction[:, None]     # rescale all H columns
# acc += einsum('qk,kh->qh', P_block, V_block)  # add new contribution
# ```
#
# After the last KV block, we normalize: `output = acc / ℓ[:, None]`.
# (This is because we've been accumulating unnormalized `exp(s - m) · V`,
# and need to divide by the total `ℓ = sum(exp(s - m))` at the end.)
#
# **Scratch memory** holds three things:
# - `acc`: `(bq, H)` — running output accumulator
# - `m`: `(bq,)` — running max per row
# - `l`: `(bq,)` — running sum of exponentials per row

# %%
T4 = 128      # sequence length
H4 = 64       # head dimension
bq4 = 32      # Q block size
bk4 = 32      # KV block size
tiles_kv4 = T4 // bk4

Q4 = jax.random.normal(jax.random.key(30), (T4, H4))
K4 = jax.random.normal(jax.random.key(31), (T4, H4))
V4 = jax.random.normal(jax.random.key(32), (T4, H4))

# --- Reference: attention for just the first Q block ---
def attention_one_block_spec(Q, K, V):
    """Attention output for first bq4 rows only."""
    q_block = Q[:bq4]                                          # (bq4, H4)
    S = jnp.einsum('qh,kh->qk', q_block, K) / jnp.sqrt(H4).astype(Q.dtype)  # (bq4, T4)
    P = jax.nn.softmax(S, axis=-1)                             # (bq4, T4)
    return jnp.einsum('qk,kh->qh', P, V)                      # (bq4, H4)


# --- Kernel: tiled attention for one Q block ---
def tiled_attention_one_block_kernel(
    q_ref,      # (bq4, H4) — the Q block (same for all KV iterations)
    k_ref,      # (bk4, H4) — one KV block
    v_ref,      # (bk4, H4) — one KV block
    o_ref,      # (bq4, H4) — output
    acc_ref,    # (bq4, H4) — scratch: running output accumulator
    m_ref,      # (bq4,)    — scratch: running row max
    l_ref,      # (bq4,)    — scratch: running row sum_exp
):
    """Process one KV block for a single Q block using online softmax.

    Grid: (tiles_kv4,) — iterates over KV blocks
    """
    kv = pl.program_id(0)
    # YOUR CODE HERE
    # 1. On first KV block: init acc=0, m=-inf, l=0
    # 2. Scores: s = einsum('qh,kh->qk', q, k) / sqrt(H4)  → (bq4, bk4)
    # 3. Compute row-wise max of scores: m_tile        → (bq4,)
    # 4. Update running max: m_new = max(m, m_tile)    → (bq4,)
    # 5. Correction factor: corr = exp(m - m_new)      → (bq4,)
    # 6. Rescale accumulator: acc *= corr[:, None]
    # 7. Compute P_block = exp(s - m_new[:, None])     → (bq4, bk4)
    # 8. Update l: l = l * corr + P_block.sum(axis=-1)
    # 9. Accumulate: acc += einsum('qk,kh->qh', P_block, v)
    # 10. Update m = m_new
    # 11. On LAST KV block: o = acc / l[:, None]


# %%
expected4 = attention_one_block_spec(Q4, K4, V4)

actual4 = pl.pallas_call(
    tiled_attention_one_block_kernel,
    grid=(tiles_kv4,),
    in_specs=[
        pl.BlockSpec((bq4, H4), lambda kv: (0, 0)),       # Q: always first block
        pl.BlockSpec((bk4, H4), lambda kv: (kv, 0)),      # K: iterate over blocks
        pl.BlockSpec((bk4, H4), lambda kv: (kv, 0)),      # V: iterate over blocks
    ],
    out_specs=pl.BlockSpec((bq4, H4), lambda kv: (0, 0)),
    out_shape=jax.ShapeDtypeStruct((bq4, H4), jnp.float32),
    scratch_shapes=[
        pltpu.VMEM((bq4, H4), jnp.float32),    # acc
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
#     acc_ref[...] = jnp.zeros((bq4, H4), dtype=jnp.float32)
#     m_ref[...] = jnp.full((bq4,), -jnp.inf, dtype=jnp.float32)
#     l_ref[...] = jnp.zeros((bq4,), dtype=jnp.float32)
#
# q = q_ref[...]
# k = k_ref[...]
# v = v_ref[...]
# s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H4).astype(jnp.float32)  # (bq4, bk4)
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
# acc_ref[...] += jnp.einsum('qk,kh->qh', p, v)    # accumulate P @ V
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
#         acc_ref[...] = jnp.zeros((bq4, H4), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq4,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq4,), dtype=jnp.float32)
#
#     q = q_ref[...]
#     k = k_ref[...]
#     v = v_ref[...]
#     s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H4).astype(jnp.float32)
#
#     m_tile = jnp.max(s, axis=-1)
#     m_new = jnp.maximum(m_ref[...], m_tile)
#     corr = jnp.exp(m_ref[...] - m_new)
#
#     acc_ref[...] = acc_ref[...] * corr[:, None]
#     p = jnp.exp(s - m_new[:, None])
#     l_ref[...] = l_ref[...] * corr + p.sum(axis=-1)
#     acc_ref[...] += jnp.einsum('qk,kh->qh', p, v)
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
# Each block s[i,j] = einsum(Q_block_i, K_block_j) / √H  is (bq × bk)
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
T5 = 128
H5 = 64
bq5 = 32
bk5 = 32
tiles_q5 = T5 // bq5
tiles_kv5 = T5 // bk5

Q5 = jax.random.normal(jax.random.key(40), (T5, H5))
K5 = jax.random.normal(jax.random.key(41), (T5, H5))
V5 = jax.random.normal(jax.random.key(42), (T5, H5))

# --- Reference ---
def flash_attention_spec(Q, K, V):
    """Full attention: softmax(QK^T / sqrt(H)) @ V"""
    H = Q.shape[-1]
    S = jnp.einsum('th,sh->ts', Q, K) / jnp.sqrt(H).astype(Q.dtype)
    P = jax.nn.softmax(S, axis=-1)
    return jnp.einsum('ts,sh->th', P, V)


# --- Kernel ---
def flash_attention_kernel(
    q_ref,      # (bq5, H5)
    k_ref,      # (bk5, H5)
    v_ref,      # (bk5, H5)
    o_ref,      # (bq5, H5) — output
    acc_ref,    # (bq5, H5) — scratch: accumulator
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
    # YOUR CODE HERE
    # Same pattern as Puzzle 4, but now:
    # - Use kv (not i) for the KV iteration
    # - Init on kv == 0, normalize on kv == tiles_kv5 - 1
    # - The BlockSpecs handle routing Q by i, K/V by kv


# %%
check(flash_attention_kernel, flash_attention_spec, (Q5, K5, V5),
      grid=(tiles_q5, tiles_kv5),
      in_specs=[
          pl.BlockSpec((bq5, H5), lambda i, kv: (i, 0)),    # Q: route by i
          pl.BlockSpec((bk5, H5), lambda i, kv: (kv, 0)),   # K: route by kv
          pl.BlockSpec((bk5, H5), lambda i, kv: (kv, 0)),   # V: route by kv
      ],
      out_specs=pl.BlockSpec((bq5, H5), lambda i, kv: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((T5, H5), jnp.float32),
      scratch_shapes=[
          pltpu.VMEM((bq5, H5), jnp.float32),   # acc
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
#         acc_ref[...] = jnp.zeros((bq5, H5), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq5,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq5,), dtype=jnp.float32)
#
#     q = q_ref[...]
#     k = k_ref[...]
#     v = v_ref[...]
#     s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H5).astype(jnp.float32)
#
#     m_tile = jnp.max(s, axis=-1)
#     m_new = jnp.maximum(m_ref[...], m_tile)
#     corr = jnp.exp(m_ref[...] - m_new)
#
#     acc_ref[...] = acc_ref[...] * corr[:, None]
#     p = jnp.exp(s - m_new[:, None])
#     l_ref[...] = l_ref[...] * corr + p.sum(axis=-1)
#     acc_ref[...] += jnp.einsum('qk,kh->qh', p, v)
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
T6 = 128
H6 = 64
bq6 = 32
bk6 = 32
tiles_q6 = T6 // bq6
tiles_kv6 = T6 // bk6

Q6 = jax.random.normal(jax.random.key(50), (T6, H6))
K6 = jax.random.normal(jax.random.key(51), (T6, H6))
V6 = jax.random.normal(jax.random.key(52), (T6, H6))

# --- Reference ---
def causal_attention_spec(Q, K, V):
    """Attention with causal mask."""
    H = Q.shape[-1]
    T = Q.shape[0]
    S = jnp.einsum('th,sh->ts', Q, K) / jnp.sqrt(H).astype(Q.dtype)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    S = jnp.where(mask, S, -jnp.inf)
    P = jax.nn.softmax(S, axis=-1)
    return jnp.einsum('ts,sh->th', P, V)


# --- Kernel ---
def causal_flash_kernel(
    q_ref, k_ref, v_ref, o_ref,
    acc_ref, m_ref, l_ref,
):
    """Flash attention with causal masking.

    Grid: (tiles_q6, tiles_kv6)

    Block (i, kv) falls into one of two cases:
    - SKIP: i * bq6 < kv * bk6 → entirely above diagonal, do nothing
    - COMPUTE: otherwise → apply causal mask and do flash attention
      (the causal mask is all-True for blocks fully below the diagonal,
       so you don't need to special-case FULL vs PARTIAL)
    """
    i = pl.program_id(0)
    kv = pl.program_id(1)
    # YOUR CODE HERE
    # 1. Init on kv == 0 (same as Puzzle 5)
    # 2. Determine if this block should be skipped: i * bq6 < kv * bk6
    # 3. @pl.when(should_compute): compute scores, apply causal mask,
    #    do online softmax update
    # 4. Normalize on last kv block


# %%
check(causal_flash_kernel, causal_attention_spec, (Q6, K6, V6),
      grid=(tiles_q6, tiles_kv6),
      in_specs=[
          pl.BlockSpec((bq6, H6), lambda i, kv: (i, 0)),
          pl.BlockSpec((bk6, H6), lambda i, kv: (kv, 0)),
          pl.BlockSpec((bk6, H6), lambda i, kv: (kv, 0)),
      ],
      out_specs=pl.BlockSpec((bq6, H6), lambda i, kv: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((T6, H6), jnp.float32),
      scratch_shapes=[
          pltpu.VMEM((bq6, H6), jnp.float32),
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
#         acc_ref[...] = jnp.zeros((bq6, H6), dtype=jnp.float32)
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
#         s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H6).astype(jnp.float32)
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
#         acc_ref[...] += jnp.einsum('qk,kh->qh', p, v)
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
# **Goal**: Replace hardcoded causal logic with a **data-driven block mask**.
# Use a `block_mask` array to classify blocks and a `partial_masks` array
# for per-element masks on partial blocks.
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
# arrays and store them in `partial_masks`. For causal masking, there is
# exactly one partial block per Q row (the diagonal), so we can index
# the partial mask by `i` (the Q block index). A fully general
# implementation would need an additional index map from `(i, kv)` to
# partial mask index — the production splash attention code does this.
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
H7 = 64
bq7 = 32
bk7 = 32
tiles_q7 = T7 // bq7
tiles_kv7 = T7 // bk7

Q7 = jax.random.normal(jax.random.key(60), (T7, H7))
K7 = jax.random.normal(jax.random.key(61), (T7, H7))
V7 = jax.random.normal(jax.random.key(62), (T7, H7))

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
    H = Q.shape[-1]
    T = Q.shape[0]
    S = jnp.einsum('th,sh->ts', Q, K) / jnp.sqrt(H).astype(Q.dtype)
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
    return jnp.einsum('ts,sh->th', P, V)


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
    # YOUR CODE HERE
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
        pl.BlockSpec((bq7, H7), lambda i, kv: (i, 0)),              # Q
        pl.BlockSpec((bk7, H7), lambda i, kv: (kv, 0)),             # K
        pl.BlockSpec((bk7, H7), lambda i, kv: (kv, 0)),             # V
        pl.BlockSpec((tiles_kv7,), lambda i, kv: (i,)),             # block_mask row
        pl.BlockSpec(memory_space=pl.ANY),                           # partial_masks (full)
    ],
    out_specs=pl.BlockSpec((bq7, H7), lambda i, kv: (i, 0)),
    out_shape=jax.ShapeDtypeStruct((T7, H7), jnp.float32),
    scratch_shapes=[
        pltpu.VMEM((bq7, H7), jnp.float32),
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
# For causal masking, the diagonal block for Q block `i` is always the
# `i`-th partial mask (one partial block per row, on the diagonal):
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
#         acc_ref[...] = jnp.zeros((bq7, H7), dtype=jnp.float32)
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
#         s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H7).astype(jnp.float32)
#
#         # Apply partial mask for diagonal blocks.
#         # For causal: block i has partial mask i (one per Q row).
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
#         acc_ref[...] += jnp.einsum('qk,kh->qh', p, v)
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
T8 = 128
H8 = 64
bq8 = 32
bk8 = 32
tiles_q8 = T8 // bq8
tiles_kv8 = T8 // bk8

Q8 = jax.random.normal(jax.random.key(70), (T8, H8))
K8 = jax.random.normal(jax.random.key(71), (T8, H8))
V8 = jax.random.normal(jax.random.key(72), (T8, H8))

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
    H = Q.shape[-1]
    T = Q.shape[0]
    S = jnp.einsum('th,sh->ts', Q, K) / jnp.sqrt(H).astype(Q.dtype)
    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    S = jnp.where(mask, S, -jnp.inf)
    P = jax.nn.softmax(S, axis=-1)
    return jnp.einsum('ts,sh->th', P, V)


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
    q_ref,               # (bq8, H8) — Q block
    k_ref,               # (bk8, H8) — KV block (routed by data_next)
    v_ref,               # (bk8, H8) — KV block (routed by data_next)
    partial_masks_ref,   # (num_partial8, bq8, bk8) — all partial masks
    o_ref,               # (bq8, H8) — output
    acc_ref,             # (bq8, H8) — scratch
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
    # YOUR CODE HERE
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
            pl.BlockSpec((bq8, H8), q_index_map),              # Q
            pl.BlockSpec((bk8, H8), kv_index_map),             # K (routed!)
            pl.BlockSpec((bk8, H8), kv_index_map),             # V (routed!)
            pl.BlockSpec(memory_space=pl.ANY),                  # partial_masks (full)
        ],
        out_specs=pl.BlockSpec((bq8, H8), o_index_map),
        scratch_shapes=[
            pltpu.VMEM((bq8, H8), jnp.float32),   # acc
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
#         acc_ref[...] = jnp.zeros((bq8, H8), dtype=jnp.float32)
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
#         s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H8).astype(jnp.float32)
#
#         # For causal: Q block i → partial mask i (one per row)
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
#         acc_ref[...] += jnp.einsum('qk,kh->qh', p, v)
#         m_ref[...] = m_new
#
#     @pl.when(step == grid_width8 - 1)
#     def _norm():
#         o_ref[...] = acc_ref[...] / l_ref[...][:, None]
# ```
# </details>

# %% [markdown]
# ---
# # What's Next: Backward Pass
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
