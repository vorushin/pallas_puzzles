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
# Click › to collapse this section, then click ▶ to get everything ready.

# %%
# !pip install -q jax

# %%
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
print(f"JAX {jax.__version__}")


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
T = 128    # sequence length
H = 64     # head dimension

Q = jax.random.normal(jax.random.key(0), (T, H))
K = jax.random.normal(jax.random.key(1), (T, H))
V = jax.random.normal(jax.random.key(2), (T, H))


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
expected = attention_spec(Q, K, V)
actual = my_attention(Q, K, V)

if actual is not None and jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    print("FAILED ✗")
    if actual is not None:
        print(f"  Max error: {float(jnp.max(jnp.abs(actual - expected))):.6f}")

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
N = 512          # vector length
bn = 128         # tile size
tiles_k = N // bn

# --- Reference ---
def softmax_spec(x):
    """x: (N,) → (N,)"""
    return jax.nn.softmax(x)


# --- Kernel 1: find global max via tiled reduction ---
def tiled_max_kernel(x_ref, m_ref):
    """Tile over x to compute global max m.

    x_ref: (bn,) — one tile of x
    m_ref: ()    — running global max (scalar output)

    Grid: (tiles_k,) — iterates over tiles of x
    """
    k = pl.program_id(0)
    # YOUR CODE HERE
    # 1. On first tile (k == 0): initialize m to -infinity
    # 2. On ALL tiles: m = max(m, max(tile))


# --- Kernel 2: compute sum of exponentials (needs global max m) ---
def tiled_sumexp_kernel(x_ref, m_ref, l_ref):
    """Tile over x to compute l = sum(exp(x - m)), given global max m.

    x_ref: (bn,) — one tile of x
    m_ref: ()    — global max (input, already computed by kernel 1)
    l_ref: ()    — running sum of exp(x - m) (scalar output)

    Grid: (tiles_k,) — iterates over tiles of x
    """
    k = pl.program_id(0)
    # YOUR CODE HERE
    # 1. On first tile (k == 0): initialize l to 0
    # 2. On ALL tiles: l += sum(exp(tile - m))


# %%
x = jax.random.uniform(jax.random.key(10), (N,), minval=-5.0, maxval=-1.0)

# Pass 1: find global max
m = pl.pallas_call(
    tiled_max_kernel,
    grid=(tiles_k,),
    in_specs=[pl.BlockSpec((bn,), lambda k: (k,))],
    out_specs=pl.BlockSpec(memory_space=pl.ANY),
    out_shape=jax.ShapeDtypeStruct((), jnp.float32),
    interpret=True,
)(x)

# Pass 2: compute sum(exp(x - m)) — needs m from pass 1
l = pl.pallas_call(
    tiled_sumexp_kernel,
    grid=(tiles_k,),
    in_specs=[
        pl.BlockSpec((bn,), lambda k: (k,)),  # x: tiled
        pl.BlockSpec(memory_space=pl.ANY),      # m: scalar, no blocking
    ],
    out_specs=pl.BlockSpec(memory_space=pl.ANY),
    out_shape=jax.ShapeDtypeStruct((), jnp.float32),
    interpret=True,
)(x, m)

# Pass 3: normalize (just JAX, no kernel needed)
result = jnp.exp(x - m) / l

expected_m = jnp.max(x)
expected_l = jnp.sum(jnp.exp(x - expected_m))
expected = softmax_spec(x)
m_ok = jnp.allclose(m, expected_m, atol=1e-3)
l_ok = jnp.allclose(l, expected_l, atol=1e-3)
s_ok = jnp.allclose(result, expected, atol=1e-3)
if m_ok and l_ok and s_ok:
    print(f"PASSED ✓  (m={float(m):.3f}, l={float(l):.3f})")
else:
    if not m_ok:
        print(f"FAILED ✗  m={float(m):.3f} (expected {float(expected_m):.3f})")
    if not l_ok:
        print(f"FAILED ✗  l={float(l):.3f} (expected {float(expected_l):.3f})")
    if not s_ok:
        print(f"FAILED ✗  softmax max error: {float(jnp.max(jnp.abs(result - expected))):.6f}")

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
N = 512
bn = 128
tiles_k = N // bn

# --- Reference ---
def softmax_spec(x):
    """x: (N,) → (N,)"""
    return jax.nn.softmax(x)


# --- Kernel: online softmax stats ---
def online_softmax_kernel(x_ref, m_ref, l_ref):
    """Single-pass softmax stats: m = global max, l = sum(exp(x - m)).

    x_ref: (bn,) — one tile of x
    m_ref: ()    — running global max (scalar)
    l_ref: ()    — running sum of exponentials (scalar)

    Grid: (tiles_k,) — iterates over tiles of x

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
x = jax.random.normal(jax.random.key(20), (N,))

m, l = pl.pallas_call(
    online_softmax_kernel,
    grid=(tiles_k,),
    in_specs=[pl.BlockSpec((bn,), lambda k: (k,))],
    out_specs=(
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
    ),
    out_shape=(
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.float32),
    ),
    interpret=True,
)(x)

result = jnp.exp(x - m) / l

expected = softmax_spec(x)
if jnp.allclose(result, expected, atol=1e-3):
    print(f"PASSED ✓  (m={float(m):.3f}, l={float(l):.3f})")
else:
    print(f"FAILED ✗  max error: {float(jnp.max(jnp.abs(result - expected))):.6f}")
    print(f"  m={float(m):.3f} (expected {float(jnp.max(x)):.3f})")
    print(f"  l={float(l):.3f} (expected {float(jnp.sum(jnp.exp(x - jnp.max(x)))):.3f})")

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
T = 128      # sequence length
H = 64       # head dimension
bq = 32      # Q block size
bk = 32      # KV block size
tiles_kv = T // bk

Q = jax.random.normal(jax.random.key(30), (T, H))
K = jax.random.normal(jax.random.key(31), (T, H))
V = jax.random.normal(jax.random.key(32), (T, H))

# --- Reference: attention for just the first Q block ---
def attention_one_block_spec(Q, K, V):
    """Attention output for first bq rows only."""
    q_block = Q[:bq]                                          # (bq, H)
    S = jnp.einsum('qh,kh->qk', q_block, K) / jnp.sqrt(H).astype(Q.dtype)  # (bq, T)
    P = jax.nn.softmax(S, axis=-1)                             # (bq, T)
    return jnp.einsum('qk,kh->qh', P, V)                      # (bq, H)


# --- Kernel: tiled attention for one Q block ---
def tiled_attention_one_block_kernel(
    q_ref,      # (bq, H) — the Q block (same for all KV iterations)
    k_ref,      # (bk, H) — one KV block
    v_ref,      # (bk, H) — one KV block
    o_ref,      # (bq, H) — output
    acc_ref,    # (bq, H) — scratch: running output accumulator
    m_ref,      # (bq,)   — scratch: running row max
    l_ref,      # (bq,)   — scratch: running row sum_exp
):
    """Process one KV block for a single Q block using online softmax.

    Grid: (tiles_kv,) — iterates over KV blocks
    """
    kv = pl.program_id(0)
    # YOUR CODE HERE
    # 1. On first KV block: init acc=0, m=-inf, l=0
    # 2. Scores: s = einsum('qh,kh->qk', q, k) / sqrt(H)  → (bq, bk)
    # 3. Compute row-wise max of scores: m_tile        → (bq,)
    # 4. Update running max: m_new = max(m, m_tile)    → (bq,)
    # 5. Correction factor: corr = exp(m - m_new)      → (bq,)
    # 6. Rescale accumulator: acc *= corr[:, None]
    # 7. Compute P_block = exp(s - m_new[:, None])     → (bq, bk)
    # 8. Update l: l = l * corr + P_block.sum(axis=-1)
    # 9. Accumulate: acc += einsum('qk,kh->qh', P_block, v)
    # 10. Update m = m_new
    # 11. On LAST KV block: o = acc / l[:, None]


# %%
expected = attention_one_block_spec(Q, K, V)

actual = pl.pallas_call(
    tiled_attention_one_block_kernel,
    grid=(tiles_kv,),
    in_specs=[
        pl.BlockSpec((bq, H), lambda kv: (0, 0)),       # Q: always first block
        pl.BlockSpec((bk, H), lambda kv: (kv, 0)),      # K: iterate over blocks
        pl.BlockSpec((bk, H), lambda kv: (kv, 0)),      # V: iterate over blocks
    ],
    out_specs=pl.BlockSpec((bq, H), lambda kv: (0, 0)),
    out_shape=jax.ShapeDtypeStruct((bq, H), jnp.float32),
    scratch_shapes=[
        pltpu.VMEM((bq, H), jnp.float32),    # acc
        pltpu.VMEM((bq,), jnp.float32),       # m
        pltpu.VMEM((bq,), jnp.float32),       # l
    ],
    interpret=True,
)(Q, K, V)

if jnp.allclose(actual, expected, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual.shape})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected (first 2 rows):\n{expected[:2, :8]}")
    print(f"  Got      (first 2 rows):\n{actual[:2, :8]}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Initialization and scores</summary>
#
# ```python
# @pl.when(kv == 0)
# def _init():
#     acc_ref[...] = jnp.zeros((bq, H), dtype=jnp.float32)
#     m_ref[...] = jnp.full((bq,), -jnp.inf, dtype=jnp.float32)
#     l_ref[...] = jnp.zeros((bq,), dtype=jnp.float32)
#
# q = q_ref[...]
# k = k_ref[...]
# v = v_ref[...]
# s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H).astype(jnp.float32)  # (bq, bk)
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — Online softmax update with output correction</summary>
#
# ```python
# m_tile = jnp.max(s, axis=-1)                    # (bq,)
# m_new = jnp.maximum(m_ref[...], m_tile)          # (bq,)
# corr = jnp.exp(m_ref[...] - m_new)               # (bq,)
#
# acc_ref[...] = acc_ref[...] * corr[:, None]       # rescale old output
# p = jnp.exp(s - m_new[:, None])                   # (bq, bk)
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
#         acc_ref[...] = jnp.zeros((bq, H), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq,), dtype=jnp.float32)
#
#     q = q_ref[...]
#     k = k_ref[...]
#     v = v_ref[...]
#     s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H).astype(jnp.float32)
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
#     @pl.when(kv == tiles_kv - 1)
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
# **Each Q block is completely independent — they don't share state.**
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
# Here's the beautiful part: **the kernel body is identical to Puzzle 4.**
# Copy it verbatim. The variable `i` is unused inside the kernel — the
# BlockSpecs handle routing Q and O by `i`, so the kernel just sees "my Q
# block" and "my KV block" exactly like before.
#
# All the new complexity lives in the `pallas_call` setup — the 2D grid
# and the BlockSpec index maps. The kernel doesn't change at all.
#
# **This is flash attention.** Same exact outputs as Puzzle 1, but O(T)
# memory instead of O(T²). The score matrix never exists in full — each
# `(bq, bk)` tile is computed, used, and discarded.


# %%
T = 128
H = 64
bq = 32
bk = 32
tiles_q = T // bq
tiles_kv = T // bk

Q = jax.random.normal(jax.random.key(40), (T, H))
K = jax.random.normal(jax.random.key(41), (T, H))
V = jax.random.normal(jax.random.key(42), (T, H))

# --- Reference ---
def flash_attention_spec(Q, K, V):
    """Full attention: softmax(QK^T / sqrt(H)) @ V"""
    H = Q.shape[-1]
    S = jnp.einsum('th,sh->ts', Q, K) / jnp.sqrt(H).astype(Q.dtype)
    P = jax.nn.softmax(S, axis=-1)
    return jnp.einsum('ts,sh->th', P, V)


# --- Kernel ---
def flash_attention_kernel(
    q_ref,      # (bq, H)
    k_ref,      # (bk, H)
    v_ref,      # (bk, H)
    o_ref,      # (bq, H) — output
    acc_ref,    # (bq, H) — scratch: accumulator
    m_ref,      # (bq,)   — scratch: running max
    l_ref,      # (bq,)   — scratch: running sum_exp
):
    """Flash attention kernel.

    Grid: (tiles_q, tiles_kv)
      - program_id(0) = i: which Q block
      - program_id(1) = kv: which KV block (reduction dimension)
    """
    i = pl.program_id(0)    # unused — BlockSpecs handle Q/O routing by i
    kv = pl.program_id(1)
    # YOUR CODE HERE
    # Copy your Puzzle 4 kernel body — it works unchanged!


# %%
expected = flash_attention_spec(Q, K, V)
actual = pl.pallas_call(
    flash_attention_kernel,
    grid=(tiles_q, tiles_kv),
    in_specs=[
        pl.BlockSpec((bq, H), lambda i, kv: (i, 0)),    # Q: route by i
        pl.BlockSpec((bk, H), lambda i, kv: (kv, 0)),   # K: route by kv
        pl.BlockSpec((bk, H), lambda i, kv: (kv, 0)),   # V: route by kv
    ],
    out_specs=pl.BlockSpec((bq, H), lambda i, kv: (i, 0)),
    out_shape=jax.ShapeDtypeStruct((T, H), jnp.float32),
    scratch_shapes=[
        pltpu.VMEM((bq, H), jnp.float32),   # acc
        pltpu.VMEM((bq,), jnp.float32),      # m
        pltpu.VMEM((bq,), jnp.float32),      # l
    ],
    interpret=True,
)(Q, K, V)

if jnp.allclose(actual, expected, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual.shape})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected (first 2 rows):\n{expected[:2, :8]}")
    print(f"  Got      (first 2 rows):\n{actual[:2, :8]}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Really, just copy Puzzle 4</summary>
#
# The kernel body is *literally* the same as Puzzle 4. Copy-paste it.
# `i` is unused — the BlockSpecs route Q/O by `i` so the kernel never
# needs to know which Q block it's processing.
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
#         acc_ref[...] = jnp.zeros((bq, H), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq,), dtype=jnp.float32)
#
#     q = q_ref[...]
#     k = k_ref[...]
#     v = v_ref[...]
#     s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H).astype(jnp.float32)
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
#     @pl.when(kv == tiles_kv - 1)
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
T = 128
H = 64
bq = 32
bk = 32
tiles_q = T // bq
tiles_kv = T // bk

Q = jax.random.normal(jax.random.key(50), (T, H))
K = jax.random.normal(jax.random.key(51), (T, H))
V = jax.random.normal(jax.random.key(52), (T, H))

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

    Grid: (tiles_q, tiles_kv)

    Block (i, kv) falls into one of two cases:
    - SKIP: i * bq < kv * bk → entirely above diagonal, do nothing
    - COMPUTE: otherwise → apply causal mask and do flash attention
      (the causal mask is all-True for blocks fully below the diagonal,
       so you don't need to special-case FULL vs PARTIAL)
    """
    i = pl.program_id(0)
    kv = pl.program_id(1)
    # YOUR CODE HERE
    # 1. Init on kv == 0 (same as Puzzle 5)
    # 2. Determine if this block should be skipped: i * bq < kv * bk
    # 3. @pl.when(should_compute): compute scores, apply causal mask,
    #    do online softmax update
    # 4. Normalize on last kv block


# %%
expected = causal_attention_spec(Q, K, V)
actual = pl.pallas_call(
    causal_flash_kernel,
    grid=(tiles_q, tiles_kv),
    in_specs=[
        pl.BlockSpec((bq, H), lambda i, kv: (i, 0)),
        pl.BlockSpec((bk, H), lambda i, kv: (kv, 0)),
        pl.BlockSpec((bk, H), lambda i, kv: (kv, 0)),
    ],
    out_specs=pl.BlockSpec((bq, H), lambda i, kv: (i, 0)),
    out_shape=jax.ShapeDtypeStruct((T, H), jnp.float32),
    scratch_shapes=[
        pltpu.VMEM((bq, H), jnp.float32),
        pltpu.VMEM((bq,), jnp.float32),
        pltpu.VMEM((bq,), jnp.float32),
    ],
    interpret=True,
)(Q, K, V)

if jnp.allclose(actual, expected, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual.shape})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected (first 2 rows):\n{expected[:2, :8]}")
    print(f"  Got      (first 2 rows):\n{actual[:2, :8]}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Block classification</summary>
#
# ```python
# should_compute = (i * bq >= kv * bk)    # not above diagonal
# is_full = ((i + 1) * bq > (kv + 1) * bk)  # entirely below diagonal
# ```
# When `should_compute` is False, the block is fully masked — skip it.
# When `is_full` is True, no per-element mask needed.
# </details>
#
# <details><summary>Hint 2 of 3 — Applying the causal mask</summary>
#
# For partial (diagonal) blocks, create the mask with global indices:
# ```python
# q_idx = i * bq + jnp.arange(bq)[:, None]
# kv_idx = kv * bk + jnp.arange(bk)[None, :]
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
#         acc_ref[...] = jnp.zeros((bq, H), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq,), dtype=jnp.float32)
#
#     should_compute = (i * bq >= kv * bk)
#
#     @pl.when(should_compute)
#     def _compute():
#         q = q_ref[...]
#         k = k_ref[...]
#         v = v_ref[...]
#         s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H).astype(jnp.float32)
#
#         # Apply causal mask for partial blocks
#         q_idx = i * bq + jnp.arange(bq)[:, None]
#         kv_idx = kv * bk + jnp.arange(bk)[None, :]
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
#     @pl.when(kv == tiles_kv - 1)
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
# ![block_mask (precomputed)](https://raw.githubusercontent.com/vorushin/pallas_puzzles/master/images/splash-attention-puzzle7.drawio.svg)
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
T = 128
H = 64
bq = 32
bk = 32
tiles_q = T // bq
tiles_kv = T // bk

Q = jax.random.normal(jax.random.key(60), (T, H))
K = jax.random.normal(jax.random.key(61), (T, H))
V = jax.random.normal(jax.random.key(62), (T, H))

# --- Build causal block_mask ---
# 0 = SKIP, 1 = PARTIAL (diagonal), 2 = FULL (below diagonal)
block_mask = jnp.zeros((tiles_q, tiles_kv), dtype=jnp.int32)
for qi in range(tiles_q):
    for kvi in range(tiles_kv):
        if qi * bq < kvi * bk:
            block_mask = block_mask.at[qi, kvi].set(0)    # SKIP
        elif (qi + 1) * bq > (kvi + 1) * bk:
            block_mask = block_mask.at[qi, kvi].set(2)    # FULL
        else:
            block_mask = block_mask.at[qi, kvi].set(1)    # PARTIAL

print("block_mask:")
print(block_mask)

# --- Build partial mask blocks ---
# For each diagonal block, precompute the per-element causal mask.
# We store them as a list indexed by the diagonal block number.
num_partial = int(min(tiles_q, tiles_kv))
partial_masks = jnp.zeros((num_partial, bq, bk), dtype=jnp.bool_)
partial_idx = 0
for qi in range(tiles_q):
    for kvi in range(tiles_kv):
        if int(block_mask[qi, kvi]) == 1:
            q_idx = qi * bq + jnp.arange(bq)[:, None]
            kv_idx = kvi * bk + jnp.arange(bk)[None, :]
            partial_masks = partial_masks.at[partial_idx].set(q_idx >= kv_idx)
            partial_idx += 1

print(f"\npartial_masks shape: {partial_masks.shape} ({partial_idx} partial blocks)")


# --- Reference ---
def block_sparse_attention_spec(Q, K, V, block_mask, partial_masks):
    """Attention with block-sparse mask (reference implementation)."""
    H = Q.shape[-1]
    T = Q.shape[0]
    S = jnp.einsum('th,sh->ts', Q, K) / jnp.sqrt(H).astype(Q.dtype)
    # Reconstruct full mask from block_mask + partial_masks
    full_mask = jnp.zeros((T, T), dtype=jnp.bool_)
    pidx = 0
    for qi in range(tiles_q):
        for kvi in range(tiles_kv):
            bm = int(block_mask[qi, kvi])
            r0, r1 = qi * bq, (qi + 1) * bq
            c0, c1 = kvi * bk, (kvi + 1) * bk
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
    block_mask_ref,        # (tiles_kv,) — mask row for this Q block
    partial_masks_ref,     # (num_partial, bq, bk) — all partial masks
    o_ref,
    acc_ref, m_ref, l_ref,
):
    """Flash attention with block-sparse mask from block_mask array.

    Grid: (tiles_q, tiles_kv)

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
expected = block_sparse_attention_spec(Q, K, V, block_mask, partial_masks)

# For partial block indexing, we need to know which partial mask index
# corresponds to each diagonal block. For causal: block (i,i) is the
# i-th partial block.
# We pass block_mask row-by-row via BlockSpec.
actual = pl.pallas_call(
    block_sparse_flash_kernel,
    grid=(tiles_q, tiles_kv),
    in_specs=[
        pl.BlockSpec((bq, H), lambda i, kv: (i, 0)),              # Q
        pl.BlockSpec((bk, H), lambda i, kv: (kv, 0)),             # K
        pl.BlockSpec((bk, H), lambda i, kv: (kv, 0)),             # V
        pl.BlockSpec((tiles_kv,), lambda i, kv: (i,)),            # block_mask row
        pl.BlockSpec(memory_space=pl.ANY),                          # partial_masks (full)
    ],
    out_specs=pl.BlockSpec((bq, H), lambda i, kv: (i, 0)),
    out_shape=jax.ShapeDtypeStruct((T, H), jnp.float32),
    scratch_shapes=[
        pltpu.VMEM((bq, H), jnp.float32),
        pltpu.VMEM((bq,), jnp.float32),
        pltpu.VMEM((bq,), jnp.float32),
    ],
    interpret=True,
)(Q, K, V, block_mask, partial_masks)

if jnp.allclose(actual, expected, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual.shape})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected (first 2 rows):\n{expected[:2, :8]}")
    print(f"  Got      (first 2 rows):\n{actual[:2, :8]}")

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
# mask = partial_masks_ref[i]  # (bq, bk) boolean mask
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
#         acc_ref[...] = jnp.zeros((bq, H), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq,), dtype=jnp.float32)
#
#     block_type = block_mask_ref[kv]
#     should_compute = block_type > 0
#
#     @pl.when(should_compute)
#     def _compute():
#         q = q_ref[...]
#         k = k_ref[...]
#         v = v_ref[...]
#         s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H).astype(jnp.float32)
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
#     @pl.when(kv == tiles_kv - 1)
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
# ![Regular vs compacted grid](https://raw.githubusercontent.com/vorushin/pallas_puzzles/master/images/splash-attention-puzzle8.drawio.svg)
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
# are loaded into scalar memory (SMEM, separate from VMEM where tile data lives)
# and accessible to both index maps
# and the kernel body. The index maps use `data_next` to route K,V loads
# to the correct blocks.
#
# This is the production pattern from `jax.experimental.pallas.ops.tpu.splash_attention`.


# %%
T = 128
H = 64
bq = 32
bk = 32
tiles_q = T // bq
tiles_kv = T // bk

Q = jax.random.normal(jax.random.key(70), (T, H))
K = jax.random.normal(jax.random.key(71), (T, H))
V = jax.random.normal(jax.random.key(72), (T, H))

# --- Build causal block_mask (same as Puzzle 7) ---
block_mask = jnp.zeros((tiles_q, tiles_kv), dtype=jnp.int32)
for qi in range(tiles_q):
    for kvi in range(tiles_kv):
        if qi * bq < kvi * bk:
            block_mask = block_mask.at[qi, kvi].set(0)
        elif (qi + 1) * bq > (kvi + 1) * bk:
            block_mask = block_mask.at[qi, kvi].set(2)
        else:
            block_mask = block_mask.at[qi, kvi].set(1)

# --- Build compacted iteration maps ---
# For each Q block, list the non-skip KV blocks in order
grid_width = 0
for qi in range(tiles_q):
    count = int(jnp.sum(block_mask[qi] > 0))
    grid_width = max(grid_width, count)

# data_next[i, step] = which KV block to process at step `step`
# mask_next[i, step] = block type at that step
data_next = jnp.zeros((tiles_q, grid_width), dtype=jnp.int32)
mask_next = jnp.zeros((tiles_q, grid_width), dtype=jnp.int32)

for qi in range(tiles_q):
    step = 0
    for kvi in range(tiles_kv):
        if int(block_mask[qi, kvi]) > 0:
            data_next = data_next.at[qi, step].set(kvi)
            mask_next = mask_next.at[qi, step].set(int(block_mask[qi, kvi]))
            step += 1

print(f"block_mask:\n{block_mask}")
print(f"\ndata_next (kv indices per step):\n{data_next}")
print(f"\nmask_next (block types per step):\n{mask_next}")
print(f"\ngrid_width: {grid_width} (max non-skip blocks per Q block)")

# --- Build partial masks ---
num_partial = int(min(tiles_q, tiles_kv))
partial_masks = jnp.zeros((num_partial, bq, bk), dtype=jnp.bool_)
pidx = 0
for qi in range(tiles_q):
    for kvi in range(tiles_kv):
        if int(block_mask[qi, kvi]) == 1:
            q_idx = qi * bq + jnp.arange(bq)[:, None]
            kv_idx = kvi * bk + jnp.arange(bk)[None, :]
            partial_masks = partial_masks.at[pidx].set(q_idx >= kv_idx)
            pidx += 1


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
    data_next_ref,       # (tiles_q, grid_width) — scalar prefetch
    mask_next_ref,       # (tiles_q, grid_width) — scalar prefetch
    q_ref,               # (bq, H) — Q block
    k_ref,               # (bk, H) — KV block (routed by data_next)
    v_ref,               # (bk, H) — KV block (routed by data_next)
    partial_masks_ref,   # (num_partial, bq, bk) — all partial masks
    o_ref,               # (bq, H) — output
    acc_ref,             # (bq, H) — scratch
    m_ref,               # (bq,) — scratch
    l_ref,               # (bq,) — scratch
):
    """Splash attention kernel with compacted block-sparse iteration.

    Grid: (tiles_q, grid_width)
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
    # 4. Normalize on step == grid_width - 1


# %%
expected = splash_attention_spec(Q, K, V, data_next, mask_next, partial_masks)

actual = pl.pallas_call(
    splash_attention_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,   # data_next and mask_next
        in_specs=[
            pl.BlockSpec((bq, H), q_index_map),              # Q
            pl.BlockSpec((bk, H), kv_index_map),             # K (routed!)
            pl.BlockSpec((bk, H), kv_index_map),             # V (routed!)
            pl.BlockSpec(memory_space=pl.ANY),                 # partial_masks (full)
        ],
        out_specs=pl.BlockSpec((bq, H), o_index_map),
        scratch_shapes=[
            pltpu.VMEM((bq, H), jnp.float32),   # acc
            pltpu.VMEM((bq,), jnp.float32),      # m
            pltpu.VMEM((bq,), jnp.float32),      # l
        ],
        grid=(tiles_q, grid_width),
    ),
    interpret=True,
)(data_next, mask_next, Q, K, V, partial_masks)

if jnp.allclose(actual, expected, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual.shape})")
    print(f"\n🎉 Congratulations! You've built splash attention from scratch!")
    print(f"   Grid: ({tiles_q}, {grid_width}) instead of ({tiles_q}, {tiles_kv})")
    print(f"   Skipped {tiles_q * tiles_kv - tiles_q * grid_width} "
          f"of {tiles_q * tiles_kv} total block pairs")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected (first 2 rows):\n{expected[:2, :8]}")
    print(f"  Got      (first 2 rows):\n{actual[:2, :8]}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — The kernel is similar to Puzzle 7</summary>
#
# The main differences from Puzzle 7:
# - K,V are already loaded for the correct block (index maps handle routing)
# - Read block type from `mask_next_ref[i, step]` instead of `block_mask_ref[kv]`
# - Init/normalize on `step == 0` and `step == grid_width - 1`
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
#         acc_ref[...] = jnp.zeros((bq, H), dtype=jnp.float32)
#         m_ref[...] = jnp.full((bq,), -jnp.inf, dtype=jnp.float32)
#         l_ref[...] = jnp.zeros((bq,), dtype=jnp.float32)
#
#     block_type = mask_next_ref[i, step]
#     should_compute = block_type > 0
#
#     @pl.when(should_compute)
#     def _compute():
#         q = q_ref[...]
#         k = k_ref[...]
#         v = v_ref[...]
#         s = jnp.einsum('qh,kh->qk', q, k) / jnp.sqrt(H).astype(jnp.float32)
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
#     @pl.when(step == grid_width - 1)
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
