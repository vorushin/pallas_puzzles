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
# <a href="https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/grouped_matmul.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # Pallas Puzzles: Grouped Matmul
#
# **Progressive puzzles** building toward a production **grouped matmul**
# kernel — the core operation behind **Mixture-of-Experts** (MoE) dispatch
# on TPU. You'll implement scalar prefetch, group metadata, masked stores,
# and full grouped matmul with variable-size groups.
#
# Every puzzle runs on **CPU** via `interpret=True` — no TPU needed.
#
# **Prerequisites**: Complete **basics.py** first (Pallas foundations and
# tiled matmul patterns).
#
# **Key Pallas docs**: https://docs.jax.dev/en/latest/pallas/index.html

# %% [markdown]
# ## Setup

# %%
# !pip install -q jax jaxtyping

# %%
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
print(f"JAX {jax.__version__}")

# %% [markdown]
# ---
# # MoE & Grouped Matmul
#
# Before writing any Pallas code, let's understand **why** grouped matmul
# exists.
#
# ### What is Mixture-of-Experts?
#
# A standard transformer FFN applies the same weight matrix to every token.
# **Mixture-of-Experts** (MoE) replaces this single FFN with `E` parallel
# **expert** FFNs. A small **router** network selects the top-k experts
# for each token (typically k=1 or k=2). The result: model capacity scales
# with the number of experts, but compute stays constant per token — each
# token only activates k out of E experts.
#
# ### Router and top-k gating
#
# The router is a linear layer producing logits over experts. Top-k
# selection + softmax gives gating weights. After routing, tokens are
# grouped by their assigned expert.
#
# ### The grouped matmul problem
#
# After routing, we have G groups of tokens (one per expert) with variable
# sizes. The naive approach — a Python loop of G separate matmuls — has
# terrible hardware utilization because each matmul has different
# shapes, causing JAX to compile lots of programs to execute them. Earlier
# JAX implementations worked around the ragged shapes of tensors by capping
# them to a certain limit and discarding tokens above the limit, but this
# approach performs worse on the kinds of tasks that are very sensitive to
# individual tokens (e.g., coding).
#
# **Grouped matmul** solves this: concatenate all tokens into a single
# `lhs (M, K)`, stack expert weights into `rhs (G, K, N)`, and run one
# kernel that handles all G matmuls. The kernel uses metadata to route
# each tile to the correct expert. Different numbers of tokens per expert
# are handled by splitting the inputs into fixed tiles — the JAX compiler
# sees only the constant tile sizes, and the logic of which tiles to
# process fully and which to process multiple times with different expert
# weights is handled by the kernel.
#
# Here is the part of the famous Block Sparse Matrix Multiplication diagram
# from [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841):
#
# # <img src="https://raw.githubusercontent.com/vorushin/pallas_puzzles/master/images/megablocks_paper_block_sparse_mm.png" width="500">
#
# ### Reference materials
#
# - [Sparse computations on TPU with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html) —
#   official JAX tutorial on using scalar prefetch for block-sparse kernels,
#   including a grouped matmul example.
# - [`jax.experimental.pallas.ops.tpu.megablox`](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/megablox) —
#   production MegaBlox grouped matmul kernels in the JAX repo, used by
#   MaxText and other large-scale MoE training frameworks.
# - [Tokamax `ragged_dot`](https://github.com/openxla/tokamax) —
#   cross-platform (GPU + TPU) kernel library built on Pallas, with an
#   optimized ragged dot implementation for MoE workloads.
#
# ### Data shapes for this notebook
#
# - `lhs (M, K)` — concatenated token representations
# - `rhs (G, K, N)` — stacked expert weights
# - `group_sizes (G,)` — number of tokens per expert
# - `out (M, N)` — concatenated outputs
#
# ### The naive loop (our reference spec)
#
# This is what the kernel must match — but in a single fused operation:

# %%
def grouped_matmul_spec(lhs, rhs, group_sizes):
    """Reference: G separate matmuls in a loop."""
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])
    out = jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=jnp.float32)
    for g in range(len(group_sizes)):
        s, e = int(offsets[g]), int(offsets[g + 1])
        out = out.at[s:e].set(lhs[s:e] @ rhs[g])
    return out

# %% [markdown]
# ---
# # Part I: Scalar Prefetch & Group Metadata (Puzzles 1–2)

# %% [markdown]
# ---
# ## Puzzle 1: Scalar Prefetch — Permuted Batched Matmul
#
# **Goal**: Implement a **permuted batched matmul** where the mapping from
# output group to rhs group is determined at runtime by a permutation array.
#
# ### Theory
#
# In grouped matmul, the tile-to-group mapping is computed at runtime (from
# `group_sizes`). Standard `BlockSpec` index maps only see grid indices —
# they can't access runtime arrays.
#
# **Scalar prefetch** solves this. With `PrefetchScalarGridSpec`:
# - Small arrays are loaded into **SMEM** (scalar memory — separate from
#   **VMEM**, the vector memory where tile data lives) before the kernel
# - Index maps receive these SMEM refs as extra arguments
# - The kernel also receives them as leading arguments
#
# ```python
# PrefetchScalarGridSpec(
#     num_scalar_prefetch=1,  # first 1 arg is scalar-prefetched
#     in_specs=[...],
#     out_specs=...,
#     grid=(...),
# )
# ```
#
# Index map signature becomes: `lambda grid_idx0, ..., *prefetch_refs: (...)`
#
# The kernel signature becomes: `kernel(prefetch_ref0, ..., in_ref0, ..., out_ref, *scratch)`
#
# **Reminder from basics.py Puzzle 10**: Using `None` in `BlockSpec`
# block_shape squeezes that dimension — the ref won't have that dim.
# `BlockSpec((None, M, K), lambda b: (b, 0, 0))` gives the kernel a
# ref of shape `(M, K)`, not `(1, M, K)`.

# %%
G = 4
M, K, N = 64, 64, 64

# --- Reference ---
def permuted_matmul_spec(lhs, rhs, perm):
    """lhs: (G, M, K), rhs: (G, K, N), perm: (G,) -> (G, M, N)
    out[i] = lhs[i] @ rhs[perm[i]]
    """
    return jnp.stack([lhs[i] @ rhs[perm[i]] for i in range(G)])

# --- Kernel skeleton ---
def permuted_matmul_kernel(perm_ref, lhs_ref, rhs_ref, o_ref):
    # perm_ref: scalar-prefetched permutation array (in SMEM)
    # lhs_ref: (M, K) — current group's lhs (batch dim squeezed by None)
    # rhs_ref: (K, N) — permuted group's rhs (loaded via index map)
    # o_ref: (M, N) — output tile
    # YOUR CODE HERE


# --- Index maps ---
def lhs_index_map(g, perm_ref):
    return (g, 0, 0)

def rhs_index_map(g, perm_ref):
    # Use the scalar-prefetched perm to look up which rhs group to load
    return (perm_ref[g], 0, 0)

def out_index_map(g, perm_ref):
    return (g, 0, 0)

# --- Tests ---
lhs = jax.random.normal(jax.random.key(14), (G, M, K))
rhs = jax.random.normal(jax.random.key(15), (G, K, N))
perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)

expected = permuted_matmul_spec(lhs, rhs, perm)

actual = pl.pallas_call(
    permuted_matmul_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,  # first arg (perm) is prefetched to SMEM
        in_specs=[
            pl.BlockSpec((None, M, K), lhs_index_map),
            pl.BlockSpec((None, K, N), rhs_index_map),
        ],
        out_specs=pl.BlockSpec((None, M, N), out_index_map),
        grid=(G,),
    ),
    out_shape=jax.ShapeDtypeStruct((G, M, N), jnp.float32),
    interpret=True,
)(perm, lhs, rhs)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual - expected)))
    print(f"FAILED ✗  max error: {max_err:.6f}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# The index maps handle the permutation using `perm_ref[g]`. By the time the kernel runs, `rhs_ref` already points to the correct permuted group. So the kernel body is identical to basics.py Puzzle 10 — just a matmul.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# o_ref[...] = lhs_ref[...] @ rhs_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 2: Group Metadata — Tile-to-Group Mapping
#
# **Goal**: Implement the metadata functions that map fixed-size tiles to
# variable-size groups. This is **pure JAX** — not a kernel puzzle.
#
# ### Theory
#
# In grouped matmul, `lhs` has shape `(M, K)` where rows are divided into
# `G` groups of variable sizes. We need to figure out which **tiles**
# belong to which **groups**.
#
# Given `group_sizes = [300, 212, 512]` with `bm = 128`:
#
# ![Groups and tiles](https://raw.githubusercontent.com/vorushin/pallas_puzzles/master/images/grouped-matmul-puzzle2.drawio.svg)
#
# Tile at row 256 straddles the group boundary at row 300. It gets visited
# **twice**: once for group 0 (rows 256-299 are valid) and once for group 1
# (rows 300-383 are valid). The kernel uses a **mask** to only store the
# valid rows for each visit.
#
# **Rule of thumb**: `num_tiles = tiles_m + (number of non-aligned group
# boundaries)`. Aligned boundaries don't cause extra visits.
#
# **Output arrays**:
# - `group_offsets`: `[0, 300, 512, 1024]` — cumsum with leading 0
# - `group_ids`: maps each grid index to a group id
# - `m_tile_ids`: maps each grid index to which m-tile to process
# - `num_tiles`: total number of grid iterations needed
#
# The arrays can be longer than `num_tiles` (padded with the last group).

# %% [markdown]
# ### Step 2a: Group Offsets
#
# **Goal**: Compute CSR-style prefix sum of group sizes.
#
# ```
# group_sizes = [300, 212, 512]
# group_offsets = [0, 300, 512, 1024]
#                  ^    ^    ^     ^
#                  g0   g1   g2   end
# ```

# %%
def compute_group_offsets(group_sizes):
    """[0, cumsum(group_sizes)] — maps group id to start row."""
    # YOUR CODE HERE
    # Concatenate a leading zero with the cumulative sum of group_sizes

# --- Tests ---
assert jnp.array_equal(
    compute_group_offsets(jnp.array([300, 212, 512], dtype=jnp.int32)),
    jnp.array([0, 300, 512, 1024], dtype=jnp.int32))
assert jnp.array_equal(
    compute_group_offsets(jnp.array([256, 256, 256, 256], dtype=jnp.int32)),
    jnp.array([0, 256, 512, 768, 1024], dtype=jnp.int32))
print("Step 2a — compute_group_offsets: PASSED ✓")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# `jnp.cumsum` gives cumulative sums. You need to prepend a zero to get
# the leading offset.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# return jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(group_sizes)])
# ```
# </details>

# %% [markdown]
# ### Step 2b: Tiles per Group
#
# **Goal**: Compute how many tile visits each group needs.
#
# The key insight: group boundaries don't align with tile boundaries.
# Round group starts **down** to tile boundaries, round group ends **up**,
# then compute how many tiles that covers. Handle zero-size groups
# (they need 0 tiles).
#
# ```
# group_sizes = [300, 212, 512], bm = 128
#
# Group 0: rows 0–299   → tiles 0,1,2   (3 tiles: 0→256 covers 0-255, plus tile 2 covers 256-383)
# Group 1: rows 300–511 → tiles 2,3     (2 tiles: tile 2 for 300-383, tile 3 for 384-511)
# Group 2: rows 512–1023 → tiles 4,5,6,7 (4 tiles)
# ```

# %%
def compute_group_tiles(group_sizes, group_offsets, bm):
    """Number of tile visits per group (boundary tiles counted by both neighbors).

    Args:
        group_sizes: (G,) int32
        group_offsets: (G+1,) int32 from compute_group_offsets
        bm: tile size
    Returns:
        (G,) int32
    """
    # YOUR CODE HERE
    # 1. Extract group starts and ends from offsets
    # 2. Round starts DOWN and ends UP to tile boundaries
    # 3. Handle zero-size groups
    # 4. Convert rounded range sizes to tile counts

# --- Tests ---
assert jnp.array_equal(
    compute_group_tiles(jnp.array([256, 256, 256, 256], dtype=jnp.int32),
                        jnp.array([0, 256, 512, 768, 1024], dtype=jnp.int32), 128),
    jnp.array([2, 2, 2, 2]))
assert jnp.array_equal(
    compute_group_tiles(jnp.array([300, 212, 512], dtype=jnp.int32),
                        jnp.array([0, 300, 512, 1024], dtype=jnp.int32), 128),
    jnp.array([3, 2, 4]))
assert jnp.array_equal(
    compute_group_tiles(jnp.array([512, 0, 512], dtype=jnp.int32),
                        jnp.array([0, 512, 512, 1024], dtype=jnp.int32), 128),
    jnp.array([4, 0, 4]))
assert jnp.array_equal(
    compute_group_tiles(jnp.array([300, 0, 724], dtype=jnp.int32),
                        jnp.array([0, 300, 300, 1024], dtype=jnp.int32), 128),
    jnp.array([3, 0, 6]))
print("Step 2b — compute_group_tiles: PASSED ✓")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Approach</summary>
#
# Extract group starts and ends from offsets. Round starts down to tile
# boundaries (`start // bm * bm`), round ends up (`(end + bm - 1) // bm * bm`).
# The number of tiles is the rounded range divided by `bm`.
# </details>
#
# <details><summary>Hint 2 of 3 — Edge case</summary>
#
# Zero-size groups (where `group_sizes[g] == 0`) need special handling.
# Their rounded range would be nonzero because start == end but rounding
# can push them apart. Use `jnp.where(group_sizes == 0, 0, ...)`.
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# group_starts = group_offsets[:-1]
# group_ends = group_offsets[1:]
# rounded_starts = (group_starts // bm * bm).astype(jnp.int32)
# rounded_ends = ((group_ends + bm - 1) // bm * bm).astype(jnp.int32)
# rounded_sizes = jnp.where(group_sizes == 0, 0, rounded_ends - rounded_starts)
# return rounded_sizes // bm
# ```
# </details>

# %% [markdown]
# ### Step 2c: Group IDs
#
# **Goal**: Create a flat array mapping each grid index to its group id.
#
# ```
# group_tiles = [3, 2, 4]  →  group_ids = [0,0,0, 1,1, 2,2,2,2]
# ```
#
# Use `jnp.repeat` with `total_repeat_length`.

# %%
def compute_group_ids(group_tiles, num_groups, max_len):
    """Flat array mapping grid index to group id."""
    # YOUR CODE HERE
    # Repeat each group index by the number of tiles it has

# --- Tests ---
assert compute_group_ids(jnp.array([3, 2, 4]), 3, 10)[:9].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
assert compute_group_ids(jnp.array([2, 2, 2, 2]), 4, 8).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
print("Step 2c — compute_group_ids: PASSED ✓")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# `jnp.repeat` can repeat each element a different number of times.
# Repeat `[0, 1, 2]` by `group_tiles` counts. Use `total_repeat_length`
# to fix the output size (required for JIT compatibility).
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# return jnp.repeat(
#     jnp.arange(num_groups, dtype=jnp.int32),
#     group_tiles,
#     total_repeat_length=max_len,
# )
# ```
# </details>

# %% [markdown]
# ### Step 2d: Tile Visits
#
# **Goal**: Count how many times each tile is visited.
#
# Every tile is visited at least once. When a group boundary falls in the
# **middle** of a tile (not aligned to `bm`), that tile gets an extra visit.
#
# ```
# group_offsets = [0, 300, 512, 1024],  bm = 128
# Group 1 starts at row 300 → inside tile 2 → extra visit
# tile_visits = [1, 1, 2, 1, 1, 1, 1, 1]
# ```
#
# Use `jnp.histogram` to count how many non-aligned boundaries land in each tile.

# %%
def compute_tile_visits(group_sizes, group_offsets, tiles_m, bm):
    """Visit count per tile (1 + extra for each mid-tile group boundary).

    Args:
        group_sizes: (G,) int32
        group_offsets: (G+1,) int32
        tiles_m: M // bm
        bm: tile size
    Returns:
        (tiles_m,) int32
    """
    # YOUR CODE HERE
    # 1. Find group start positions (offsets[:-1] = all but the trailing end)
    # 2. Identify which starts are non-aligned (start % bm != 0)
    #    AND belong to non-empty groups
    # 3. For non-aligned starts, compute which tile they land in (start // bm)
    # 4. Count how many non-aligned boundaries per tile (jnp.histogram)
    # 5. Result = 1 + extra_visits_per_tile

# --- Tests ---
assert compute_tile_visits(
    jnp.array([256, 256, 256, 256], dtype=jnp.int32),
    jnp.array([0, 256, 512, 768, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
assert compute_tile_visits(
    jnp.array([300, 212, 512], dtype=jnp.int32),
    jnp.array([0, 300, 512, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 2, 1, 1, 1, 1, 1]
assert compute_tile_visits(
    jnp.array([512, 0, 512], dtype=jnp.int32),
    jnp.array([0, 512, 512, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
assert compute_tile_visits(
    jnp.array([300, 0, 724], dtype=jnp.int32),
    jnp.array([0, 300, 300, 1024], dtype=jnp.int32), 8, 128
).tolist() == [1, 1, 2, 1, 1, 1, 1, 1]
assert compute_tile_visits(
    jnp.array([300, 212, 512], dtype=jnp.int32),
    jnp.array([0, 300, 512, 1024], dtype=jnp.int32), 8, 128
).dtype == jnp.int32, "compute_tile_visits must return int32, not float32"
print("Step 2d — compute_tile_visits: PASSED ✓")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Approach</summary>
#
# Every tile starts with 1 visit. Extra visits come from non-aligned group
# boundaries that land in the middle of a tile. Find those boundaries,
# figure out which tile they're in, and count extras per tile.
# </details>
#
# <details><summary>Hint 2 of 3 — Key trick</summary>
#
# For boundaries that are aligned or belong to empty groups, map them to
# a dummy tile index (`tiles_m + 1`) so they fall outside the histogram
# range. Then `jnp.histogram(..., bins=tiles_m, range=(0, tiles_m))`
# counts only the real extra visits.
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# group_starts = group_offsets[:-1]
# aligned_or_empty = ((group_starts % bm) == 0) | (group_sizes == 0)
# partial_tile_ids = jnp.where(aligned_or_empty, tiles_m + 1, group_starts // bm)
# extra_visits = jnp.histogram(
#     partial_tile_ids, bins=tiles_m, range=(0, tiles_m)
# )[0]
# return (extra_visits + 1).astype(jnp.int32)
# ```
# </details>

# %% [markdown]
# ### Step 2e: M-tile IDs
#
# **Goal**: Create a flat array mapping each grid index to its m-tile id.
# Tiles visited twice (boundary tiles) appear twice.
#
# ```
# tile_visits = [1,1,2,1,1,1,1,1]  →  m_tile_ids = [0,1,2,2,3,4,5,6,7]
# ```
#
# Use `jnp.repeat` with `total_repeat_length`.

# %%
def compute_m_tile_ids(tile_visits, tiles_m, max_len):
    """Flat array mapping grid index to m-tile id."""
    # YOUR CODE HERE
    # Repeat each tile index by the number of visits it has

# --- Tests ---
assert compute_m_tile_ids(jnp.array([1,1,2,1,1,1,1,1]), 8, 10)[:9].tolist() == [0,1,2,2,3,4,5,6,7]
assert compute_m_tile_ids(jnp.array([1,1,1,1,1,1,1,1]), 8, 8).tolist() == [0,1,2,3,4,5,6,7]
print("Step 2e — compute_m_tile_ids: PASSED ✓")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# Same pattern as Step 2c, but repeat tile indices by their visit counts
# instead of group indices by tile counts.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# return jnp.repeat(
#     jnp.arange(tiles_m, dtype=jnp.int32),
#     tile_visits,
#     total_repeat_length=max_len,
# )
# ```
# </details>

# %% [markdown]
# ### Step 2f: Combined `make_group_metadata`

# %%
# --- Reference implementation for testing ---
def make_group_metadata_reference(group_sizes, m, bm):
    """Simple reference implementation — O(m) but correct."""
    num_groups = len(group_sizes)
    group_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])

    row_to_group = jnp.zeros(m, dtype=jnp.int32)
    for g in range(num_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        row_to_group = row_to_group.at[start:end].set(g)

    tiles_m = m // bm
    group_ids_list = []
    m_tile_ids_list = []

    for t in range(tiles_m):
        tile_start = t * bm
        tile_end = (t + 1) * bm
        groups_in_tile = jnp.unique(row_to_group[tile_start:tile_end])
        for g in groups_in_tile:
            group_ids_list.append(int(g))
            m_tile_ids_list.append(t)

    num_tiles = len(group_ids_list)

    max_len = tiles_m + num_groups - 1
    group_ids = jnp.zeros(max_len, dtype=jnp.int32)
    m_tile_ids = jnp.zeros(max_len, dtype=jnp.int32)
    group_ids = group_ids.at[:num_tiles].set(jnp.array(group_ids_list, dtype=jnp.int32))
    m_tile_ids = m_tile_ids.at[:num_tiles].set(jnp.array(m_tile_ids_list, dtype=jnp.int32))
    if num_tiles < max_len:
        group_ids = group_ids.at[num_tiles:].set(group_ids_list[-1])
        m_tile_ids = m_tile_ids.at[num_tiles:].set(m_tile_ids_list[-1])

    return (group_offsets.astype(jnp.int32), group_ids, m_tile_ids), num_tiles


def make_group_metadata_yours(group_sizes, m, bm):
    """Vectorized group metadata — chains steps 2a-2e.

    Args:
        group_sizes: jnp.array of shape (num_groups,), dtype int32
        m: total number of rows
        bm: tile size for m dimension

    Returns:
        (group_offsets, group_ids, m_tile_ids), num_tiles
    """
    num_groups = group_sizes.shape[0]
    tiles_m = m // bm
    max_len = tiles_m + num_groups - 1

    # YOUR CODE HERE — chain steps 2a-2e, then compute num_tiles
    # Replace this raise with your implementation:
    raise NotImplementedError("Chain compute_group_offsets -> ... -> compute_m_tile_ids")

    return (group_offsets, group_ids, m_tile_ids), num_tiles


# --- Integration tests ---
def check_metadata(name, group_sizes, m, bm):
    ref, ref_nt = make_group_metadata_reference(group_sizes, m, bm)
    yours, your_nt = make_group_metadata_yours(group_sizes, m, bm)
    ok = (ref_nt == your_nt
          and bool(jnp.array_equal(ref[0], yours[0]))
          and bool(jnp.array_equal(ref[1][:ref_nt], yours[1][:your_nt]))
          and bool(jnp.array_equal(ref[2][:ref_nt], yours[2][:your_nt])))
    status = "PASSED ✓" if ok else "FAILED ✗"
    print(f"  {name}: {status}  (num_tiles: ref={ref_nt}, yours={your_nt})")
    if not ok:
        print(f"    group_ids ref:   {ref[1][:ref_nt].tolist()}")
        print(f"    group_ids yours: {yours[1][:your_nt].tolist()}")
        print(f"    m_tile_ids ref:   {ref[2][:ref_nt].tolist()}")
        print(f"    m_tile_ids yours: {yours[2][:your_nt].tolist()}")

print("=== Integration tests ===")
check_metadata("Aligned groups",
               jnp.array([256, 256, 256, 256], dtype=jnp.int32), 1024, 128)
check_metadata("Unaligned groups",
               jnp.array([300, 212, 512], dtype=jnp.int32), 1024, 128)
check_metadata("Zero-size group (aligned)",
               jnp.array([512, 0, 512], dtype=jnp.int32), 1024, 128)
check_metadata("Zero-size group (non-aligned)",
               jnp.array([300, 0, 724], dtype=jnp.int32), 1024, 128)

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# Call the functions in order: offsets → group_tiles → group_ids,
# and in parallel offsets → tile_visits → m_tile_ids.
# `num_tiles` is the total number of tile visits across all groups.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# group_offsets = compute_group_offsets(group_sizes)
# group_tiles = compute_group_tiles(group_sizes, group_offsets, bm)
# group_ids = compute_group_ids(group_tiles, num_groups, max_len)
# tile_visits = compute_tile_visits(group_sizes, group_offsets, tiles_m, bm)
# m_tile_ids = compute_m_tile_ids(tile_visits, tiles_m, max_len)
# num_tiles = int(group_tiles.sum())
# ```
# </details>

# %% [markdown]
# ---
# # Part II: Grouped Matmul Kernels (Puzzles 3–6)

# %% [markdown]
# ### Provided utilities
#
# These are the building blocks from Part I. The production
# `make_group_metadata` is provided so you can focus on kernel logic.

# %%
def make_group_metadata(group_sizes, m, bm):
    """Compute tile-to-group mapping for grouped matmul.

    Returns:
        (group_offsets, group_ids, m_tile_ids), num_tiles
    """
    num_groups = group_sizes.shape[0]
    tiles_m = m // bm

    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_ends = ((group_ends + bm - 1) // bm * bm).astype(jnp.int32)
    rounded_starts = (group_starts // bm * bm).astype(jnp.int32)
    rounded_sizes = rounded_ends - rounded_starts
    rounded_sizes = jnp.where(group_sizes == 0, 0, rounded_sizes)
    group_tiles = rounded_sizes // bm

    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    partial_mask = ((group_offsets[:-1] % bm) == 0) | (group_sizes == 0)
    partial_tile_ids = jnp.where(partial_mask, tiles_m + 1, group_offsets[:-1] // bm)
    tile_visits = (
        jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m))[0] + 1
    )
    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    num_tiles = int(group_tiles.sum())
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def get_store_mask(grid_id, group_offsets, group_ids, m_tile_ids, bm, bn):
    """Build a (bm, bn) boolean mask for rows belonging to the current group."""
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * bm
    iota = jax.lax.broadcasted_iota(jnp.int32, (bm, bn), 0) + m_id
    return (iota >= group_start) & (iota < group_end)


# --- Shared index maps for grouped matmul ---
def lhs_imap(n_i, grid_id, k_i, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], k_i)

def rhs_imap(n_i, grid_id, k_i, group_meta_ref, group_offset_ref):
    _, group_ids, _ = group_meta_ref
    return (group_ids[grid_id], k_i, n_i)

def out_imap(n_i, grid_id, k_i, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], n_i)


# %% [markdown]
# ---
# ## Puzzle 3: Masked Store with Group Boundaries
#
# **Goal**: Write a kernel that copies input rows to output, but **masks**
# writes based on group boundaries. Only rows belonging to the current
# group are written; other rows retain their previous value (zero).
#
# ### Theory
#
# When a tile straddles a group boundary, some rows belong to group `g`
# and others to group `g+1`. The kernel must only store the rows that
# belong to the **current group** being processed.
#
# The mask is built from:
# - `group_offsets[group_id]` — start row of current group
# - `group_offsets[group_id + 1]` — end row of current group
# - `m_tile_ids[grid_id] * bm` — first row of current tile
#
# For a 2D mask `(bm, N)`, use `jax.lax.broadcasted_iota(dtype, shape, dim)`
# — it creates an array where values along `dim` are `0, 1, 2, ...` and
# all other dimensions are broadcast. Think of it as a multi-dimensional
# `jnp.arange`:
# ```python
# broadcasted_iota(int32, (4, 3), 0) -> [[0,0,0], [1,1,1], [2,2,2], [3,3,3]]
# broadcasted_iota(int32, (4, 3), 1) -> [[0,1,2], [0,1,2], [0,1,2], [0,1,2]]
# ```

# %%
M = 1024
N = 64
bm = 128
G = 3

group_sizes = jnp.array([300, 212, 512], dtype=jnp.int32)

(group_offsets, group_ids, m_tile_ids), num_tiles = \
    make_group_metadata(group_sizes, M, bm)

# --- Reference ---
def masked_copy_spec(x, group_offsets, group_ids, m_tile_ids):
    """Copy x to output, but only rows within their assigned group."""
    out = jnp.zeros_like(x)
    for grid_id in range(num_tiles):
        g = int(group_ids[grid_id])
        tile_id = int(m_tile_ids[grid_id])
        g_start = int(group_offsets[g])
        g_end = int(group_offsets[g + 1])
        t_start = tile_id * bm
        t_end = t_start + bm
        for row in range(t_start, t_end):
            if g_start <= row < g_end:
                out = out.at[row].set(x[row])
    return out

# --- Kernel skeleton ---
def masked_copy_kernel(group_offsets_ref, group_ids_ref, m_tile_ids_ref,
                       x_ref, o_ref):
    # group_offsets_ref, group_ids_ref, m_tile_ids_ref: metadata in SMEM
    # x_ref: (bm, N) — tile of input
    # o_ref: (bm, N) — tile of output
    grid_id = pl.program_id(0)
    # YOUR CODE HERE
    # 1. Look up which group and tile this grid iteration processes
    # 2. Get the group's row boundaries
    # 3. Build a 2D boolean mask for rows inside this group
    # 4. Masked store: only write rows belonging to this group

# --- Tests ---
x = jax.random.normal(jax.random.key(27), (M, N))
expected = masked_copy_spec(x, group_offsets, group_ids, m_tile_ids)

actual = pl.pallas_call(
    masked_copy_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        in_specs=[pl.BlockSpec((bm, N), lambda i, go, gi, mt: (mt[i], 0))],
        out_specs=pl.BlockSpec((bm, N), lambda i, go, gi, mt: (mt[i], 0)),
        grid=(num_tiles,),
    ),
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    interpret=True,
)(group_offsets, group_ids, m_tile_ids, x)

if jnp.allclose(actual, expected, atol=1e-5):
    print(f"PASSED ✓  (shape={actual.shape})")
else:
    nan_count = int(jnp.isnan(actual).sum())
    if nan_count > 0:
        nan_rows = jnp.where(jnp.isnan(actual).any(axis=1))[0]
        print(f"FAILED ✗  {nan_count} NaN values in output (rows: {nan_rows.tolist()[:8]}...)")
        print(f"  Common cause: indexing group_offsets_ref with grid_id instead of group_id")
    else:
        diff = jnp.abs(actual - expected)
        worst_row = int(jnp.argmax(diff.max(axis=1)))
        max_err = float(diff[worst_row].max())
        print(f"FAILED ✗  max error: {max_err:.6f} at row {worst_row}")
        g_boundaries = group_offsets.tolist()
        print(f"  Group boundaries at rows: {g_boundaries}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Key pattern</summary>
#
# ```python
# group_id = group_ids_ref[grid_id]
# m_tile = m_tile_ids_ref[grid_id]
# group_start = group_offsets_ref[group_id]
# group_end = group_offsets_ref[group_id + 1]
# tile_start = m_tile * bm
#
# # Build a (bm, N) mask where row_index in [group_start, group_end)
# # Tip: jax.lax.broadcasted_iota(dtype, shape, dimension)
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# group_id = group_ids_ref[grid_id]
# m_tile = m_tile_ids_ref[grid_id]
# group_start = group_offsets_ref[group_id]
# group_end = group_offsets_ref[group_id + 1]
# tile_start = m_tile * bm
#
# row_ids = tile_start + jax.lax.broadcasted_iota(jnp.int32, (bm, N), 0)
# mask = (row_ids >= group_start) & (row_ids < group_end)
#
# o_ref[...] = jnp.where(mask, x_ref[...], o_ref[...])
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 4: Configure Your Own Scalar-Prefetch `pallas_call`
#
# **Goal**: Given the working kernel from Puzzle 3, write the **entire**
# `pl.pallas_call` invocation from scratch.
#
# ### Theory
#
# You need to understand:
# - **`num_scalar_prefetch=3`**: the first 3 call args (`group_offsets`,
#   `group_ids`, `m_tile_ids`) are prefetched to SMEM. They appear as
#   leading refs in the kernel signature.
# - **Index map signature**: each index map gets the grid index first,
#   then all prefetch refs. E.g.,
#   `lambda i, go, gi, mt: (mt[i], 0)` — `i` is the grid index,
#   `go/gi/mt` are the three scalar-prefetched refs.
# - **Grid size**: `(num_tiles,)` — iterates over tile visits, not
#   just `tiles_m`.
# - **Call argument order**: scalar-prefetched args come first, then
#   regular inputs:
#   `(group_offsets, group_ids, m_tile_ids, x)`.

# %%
M, N = 1024, 64
bm = 128
G = 3

group_sizes = jnp.array([300, 212, 512], dtype=jnp.int32)
(group_offsets, group_ids, m_tile_ids), num_tiles = \
    make_group_metadata(group_sizes, M, bm)

# The kernel is provided (solved from Puzzle 3):
def masked_copy_kernel_solved(group_offsets_ref, group_ids_ref, m_tile_ids_ref,
                               x_ref, o_ref):
    """Copy rows from x to output, masked by group boundaries."""
    grid_id = pl.program_id(0)
    group_id = group_ids_ref[grid_id]
    m_tile = m_tile_ids_ref[grid_id]
    group_start = group_offsets_ref[group_id]
    group_end = group_offsets_ref[group_id + 1]
    tile_start = m_tile * bm

    row_ids = tile_start + jax.lax.broadcasted_iota(jnp.int32, (bm, N), 0)
    mask = (row_ids >= group_start) & (row_ids < group_end)
    o_ref[...] = jnp.where(mask, x_ref[...], o_ref[...])

# Reference spec
def masked_copy_spec4(x, group_offsets, group_ids, m_tile_ids):
    out = jnp.zeros_like(x)
    for grid_id in range(num_tiles):
        g = int(group_ids[grid_id])
        tile_id = int(m_tile_ids[grid_id])
        g_start = int(group_offsets[g])
        g_end = int(group_offsets[g + 1])
        t_start = tile_id * bm
        t_end = t_start + bm
        for row in range(t_start, t_end):
            if g_start <= row < g_end:
                out = out.at[row].set(x[row])
    return out

# --- Tests ---
x = jax.random.normal(jax.random.key(26), (M, N))
expected = masked_copy_spec4(x, group_offsets, group_ids, m_tile_ids)

# YOUR TASK: Write the complete pl.pallas_call invocation.
# Replace `None` with your working code.
#
# You need:
# - PrefetchScalarGridSpec with num_scalar_prefetch=3
# - in_specs: BlockSpec that uses m_tile_ids to route tiles
# - out_specs: BlockSpec that uses m_tile_ids to route tiles
# - grid=(num_tiles,)
# - Call args: (group_offsets, group_ids, m_tile_ids, x) — scalar prefetch first!
actual = None  # Replace with pl.pallas_call(...)(...) invocation

# --- Tests ---
if actual is not None and jnp.allclose(actual, expected, atol=1e-5):
    print(f"PASSED ✓  (shape={actual.shape})")
else:
    print("FAILED ✗  (fill in the cell above)")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Structure</summary>
#
# ```python
# actual = pl.pallas_call(
#     masked_copy_kernel_solved,
#     grid_spec=pltpu.PrefetchScalarGridSpec(
#         num_scalar_prefetch=3,
#         in_specs=[pl.BlockSpec((bm, N), lambda i, go, gi, mt: (mt[i], 0))],
#         out_specs=pl.BlockSpec((bm, N), lambda i, go, gi, mt: (mt[i], 0)),
#         grid=(num_tiles,),
#     ),
#     out_shape=...,
#     interpret=True,
# )(...)  # scalar-prefetched args first, then regular inputs
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# actual = pl.pallas_call(
#     masked_copy_kernel_solved,
#     grid_spec=pltpu.PrefetchScalarGridSpec(
#         num_scalar_prefetch=3,
#         in_specs=[pl.BlockSpec((bm, N), lambda i, go, gi, mt: (mt[i], 0))],
#         out_specs=pl.BlockSpec((bm, N), lambda i, go, gi, mt: (mt[i], 0)),
#         grid=(num_tiles,),
#     ),
#     out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
#     interpret=True,
# )(group_offsets, group_ids, m_tile_ids, x)
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 5: Simple Grouped Matmul — Equal, Tile-Aligned Groups
#
# **Goal**: Implement grouped matmul for the simplest case: all groups have
# equal size and group sizes are divisible by the tile size.
#
# ### Theory
#
# This is the "easy mode" grouped matmul. With equal, tile-aligned groups:
# - No partial tiles (every tile belongs to exactly one group)
# - `group_ids` is a simple repeat: `[0,0,1,1,2,2,3,3]`
# - `m_tile_ids` = `[0,1,2,3,4,5,6,7]` (just sequential)
# - No masking needed on stores
#
# **Grid**: `(tiles_n, num_tiles, tiles_k)`
# - `tiles_n`: N dimension (parallel — independent output columns)
# - `num_tiles`: M tiles across all groups (may revisit same tile)
# - `tiles_k`: K reduction dimension (accumulates)
#
# N is outermost because all N tiles are independent — this matters for
# pipelining later.
#
# **Index maps** (provided — study them!):
# - `lhs_imap`: `(n_i, grid_id, k_i) -> (m_tile_ids[grid_id], k_i)`
# - `rhs_imap`: `(n_i, grid_id, k_i) -> (group_ids[grid_id], k_i, n_i)`
# - `out_imap`: `(n_i, grid_id, k_i) -> (m_tile_ids[grid_id], n_i)`
#
# The `group_ids` lookup in `rhs_imap` routes each tile to the correct
# group's weight matrix.
#
# **`num_scalar_prefetch=2`**: metadata packed as a tuple
# `(group_offsets, group_ids, m_tile_ids)` in slot 1, sharding offset
# `[0]` in slot 2. The sharding offset is used for multi-device setups;
# for single-device it's always `[0]`.
#
# **Recall from basics.py Puzzle 8**: the zero/accumulate/store pattern:
# ```python
# @pl.when(k_i == 0)           # ZERO on first K tile
# def _(): acc[...] = zeros
#
# acc[...] += a @ b             # ACCUMULATE on every tile
#
# @pl.when(k_i == tiles_k - 1) # STORE on last K tile
# def _(): out[...] = acc[...]
# ```

# %%
G = 4
M, K, N = 512, 256, 128
bm, bk, bn = 128, 128, 128

group_sizes = jnp.array([M // G] * G, dtype=jnp.int32)
tiles_k = K // bk
tiles_n = N // bn

(group_offsets, group_ids, m_tile_ids), num_tiles = \
    make_group_metadata(group_sizes, M, bm)

# --- Reference ---
def simple_gmm_spec(lhs, rhs, group_sizes):
    """lhs: (M, K), rhs: (G, K, N), group_sizes: (G,) -> (M, N)"""
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])
    out = jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=jnp.float32)
    for g in range(len(group_sizes)):
        s, e = int(offsets[g]), int(offsets[g + 1])
        out = out.at[s:e].set(lhs[s:e] @ rhs[g])
    return out

# --- Kernel skeleton ---
def simple_gmm_kernel(group_metadata_ref, group_offset_ref,
                      lhs_ref, rhs_ref, o_ref, acc_ref):
    # group_metadata_ref: (group_offsets, group_ids, m_tile_ids) in SMEM
    # group_offset_ref: unused here (for sharding), always [0]
    # lhs_ref: (bm, bk) — tile of lhs
    # rhs_ref: (bk, bn) — tile of rhs (group dim squeezed by None)
    # o_ref: (bm, bn) — output tile
    # acc_ref: (bm, bn) — scratch accumulator (VMEM on TPU)
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    # YOUR CODE HERE
    # 1. Zero accumulator on first K tile
    # 2. Accumulate tile matmul
    # 3. Store result on last K tile

# --- Tests ---
lhs = jax.random.normal(jax.random.key(30), (M, K))
rhs = jax.random.normal(jax.random.key(31), (G, K, N))
expected = simple_gmm_spec(lhs, rhs, group_sizes)

group_metadata = (group_offsets, group_ids, m_tile_ids)
group_offset = jnp.array([0], dtype=jnp.int32)

actual = pl.pallas_call(
    simple_gmm_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((bm, bk), lhs_imap),
            pl.BlockSpec((None, bk, bn), rhs_imap),
        ],
        out_specs=pl.BlockSpec((bm, bn), out_imap),
        grid=(tiles_n, num_tiles, tiles_k),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    interpret=True,
)(group_metadata, group_offset, lhs, rhs)

if jnp.allclose(actual, expected, atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual - expected)))
    print(f"FAILED ✗  max error: {max_err:.6f}")
    print(f"  Expected[:2,:4]:\n{expected[:2,:4]}")
    print(f"  Actual[:2,:4]:\n{actual[:2,:4]}")

# %% [markdown]
# > **AHA moment**: The kernel body you just wrote is identical to basics.py
# > Puzzle 8. All the complexity of grouped matmul — variable group sizes,
# > tile routing, boundary handling — lives in the *metadata* and *index
# > maps*. The kernel itself is oblivious to groups.

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# You don't need `group_metadata_ref` or `group_offset_ref` in the kernel body — the index maps already used them to route the right tiles. Focus on just `lhs_ref`, `rhs_ref`, `o_ref`, and `acc_ref`. With `None` in the rhs BlockSpec, the group dimension is squeezed — `rhs_ref` is just `(bk, bn)`.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)
#
# acc_ref[...] += lhs_ref[...] @ rhs_ref[...]
#
# @pl.when(k_i == tiles_k - 1)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 6: Full Ragged Dot — Variable Group Sizes
#
# **Goal**: Handle **variable group sizes** where tiles can straddle group
# boundaries. This is the real grouped matmul.
#
# ### Theory
#
# The only difference from Puzzle 5: when groups are unequal, a tile may
# be visited **multiple times** (once per group it straddles). On each visit,
# the kernel must **mask** the store so only rows belonging to the current
# group are written.
#
# `make_group_metadata` handles all the complexity — the `group_ids` and
# `m_tile_ids` arrays already encode the repeated visits. The kernel just
# needs to add the mask at store time using `get_store_mask` from Puzzle 3:
#
# ```python
# mask = get_store_mask(grid_id, group_offsets, group_ids, m_tile_ids, bm, bn)
# o_ref[...] = jnp.where(mask, acc[...], o_ref[...])
# ```
#
# Why `o_ref[...]` in the else branch? Boundary tiles are visited twice —
# the first visit writes some rows, and the second visit must preserve
# those rows while writing others.
#
# ```
# Tile at row 256, bm=128:
# +------------------------+
# | rows 256-299: group 0  | <- Visit 1: mask=True for rows 256-299
# | rows 300-383: group 1  | <- Visit 2: mask=True for rows 300-383
# +------------------------+
# ```

# %%
G = 3
M, K, N = 1024, 256, 128
bm, bk, bn = 128, 128, 128

group_sizes = jnp.array([300, 212, 512], dtype=jnp.int32)
tiles_k = K // bk
tiles_n = N // bn

(group_offsets, group_ids, m_tile_ids), num_tiles = \
    make_group_metadata(group_sizes, M, bm)

print(f"M={M}, G={G}, group_sizes={group_sizes.tolist()}")
print(f"num_tiles={num_tiles} (vs {M//bm} base tiles)")
print(f"group_ids[:num_tiles]={group_ids[:num_tiles].tolist()}")
print(f"m_tile_ids[:num_tiles]={m_tile_ids[:num_tiles].tolist()}")

# --- Reference ---
def ragged_dot_spec(lhs, rhs, group_sizes):
    """Same as jax.lax.ragged_dot but explicit for clarity."""
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])
    out = jnp.zeros((lhs.shape[0], rhs.shape[2]), dtype=jnp.float32)
    for g in range(len(group_sizes)):
        s, e = int(offsets[g]), int(offsets[g + 1])
        if s < e:
            out = out.at[s:e].set(lhs[s:e] @ rhs[g])
    return out

# --- Kernel skeleton ---
def ragged_dot_kernel(group_metadata_ref, group_offset_ref,
                      lhs_ref, rhs_ref, o_ref, acc_ref):
    group_offsets, group_ids, m_tile_ids = group_metadata_ref
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    # YOUR CODE HERE
    # Same as Puzzle 5, but on the last K tile, apply a masked store
    # so only rows belonging to the current group are written.
    # Use get_store_mask(grid_id, group_offsets, group_ids, m_tile_ids, bm, bn)

# --- Tests ---
lhs = jax.random.normal(jax.random.key(40), (M, K))
rhs = jax.random.normal(jax.random.key(41), (G, K, N))
expected = ragged_dot_spec(lhs, rhs, group_sizes)

group_metadata = (group_offsets, group_ids, m_tile_ids)
group_offset = jnp.array([0], dtype=jnp.int32)

actual = pl.pallas_call(
    ragged_dot_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((bm, bk), lhs_imap),
            pl.BlockSpec((None, bk, bn), rhs_imap),
        ],
        out_specs=pl.BlockSpec((bm, bn), out_imap),
        grid=(tiles_n, num_tiles, tiles_k),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    interpret=True,
)(group_metadata, group_offset, lhs, rhs)

total_rows = int(group_sizes.sum())
if jnp.allclose(actual[:total_rows], expected[:total_rows], atol=1e-2, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual.shape})")
    print(f"  Verified {total_rows} active rows")
else:
    max_err = float(jnp.max(jnp.abs(actual[:total_rows] - expected[:total_rows])))
    print(f"FAILED ✗  max error: {max_err:.6f}")

# Test with equal groups too (masked kernel is a strict generalization)
equal_sizes = jnp.array([256, 256, 512], dtype=jnp.int32)
(go2, gi2, mt2), nt2 = make_group_metadata(equal_sizes, M, bm)
expected2 = ragged_dot_spec(lhs, rhs, equal_sizes)
gm2 = (go2, gi2, mt2)

actual2 = pl.pallas_call(
    ragged_dot_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((bm, bk), lhs_imap),
            pl.BlockSpec((None, bk, bn), rhs_imap),
        ],
        out_specs=pl.BlockSpec((bm, bn), out_imap),
        grid=(tiles_n, nt2, tiles_k),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    interpret=True,
)(gm2, group_offset, lhs, rhs)

if jnp.allclose(actual2, expected2, atol=1e-2, rtol=1e-2):
    print(f"  Equal groups also PASSED ✓  (strict generalization)")
else:
    max_err2 = float(jnp.max(jnp.abs(actual2 - expected2)))
    print(f"  Equal groups FAILED ✗  max error: {max_err2:.6f}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — The masked store block</summary>
#
# ```python
# # Steps 1-2 are the same as Puzzle 5 (zero + accumulate)
#
# @pl.when(k_i == tiles_k - 1)
# def _store():
#     mask = get_store_mask(grid_id, group_offsets, group_ids,
#                           m_tile_ids, bm, bn)
#     acc = acc_ref[...]
#     o_ref[...] = jnp.where(mask, acc, o_ref[...].astype(acc.dtype))
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)
#
# acc_ref[...] += lhs_ref[...] @ rhs_ref[...]
#
# @pl.when(k_i == tiles_k - 1)
# def _store():
#     mask = get_store_mask(grid_id, group_offsets, group_ids,
#                           m_tile_ids, bm, bn)
#     acc = acc_ref[...]
#     o_ref[...] = jnp.where(mask, acc, o_ref[...].astype(acc.dtype))
# ```
# </details>

# %% [markdown]
# ---
# # Part III: Putting It All Together (Puzzle 7)

# %% [markdown]
# ---
# ## Puzzle 7: Configure Your Own Grouped Matmul
#
# **Goal**: Capstone puzzle — write the complete `pl.pallas_call(...)` invocation
# for grouped matmul. The kernel from Puzzle 6 is provided.
#
# ### Theory
#
# You need to assemble all the pieces:
# - `PrefetchScalarGridSpec` with `num_scalar_prefetch=2`
# - `grid=(tiles_n, num_tiles, tiles_k)` — 3D grid
# - `in_specs`: `lhs` BlockSpec with `m_tile_ids` routing, `rhs` BlockSpec
#   with `group_ids` routing and `None` dimension squeeze
# - `out_specs`: BlockSpec with `m_tile_ids` routing
# - `scratch_shapes`: VMEM accumulator `(bm, bn)`
# - `out_shape`: full output shape `(M, N)`
# - Correct argument order: `(group_metadata, group_offset, lhs, rhs)`
#
# The index maps are the same shared `lhs_imap`, `rhs_imap`, `out_imap`
# from the utilities section. Study them — they show exactly how grid
# indices and scalar-prefetched metadata combine to route tiles.

# %%
G = 3
M, K, N = 1024, 256, 128
bm, bk, bn = 128, 128, 128
tiles_k = K // bk
tiles_n = N // bn

# The kernel is provided (solved from Puzzle 6):
def gmm_kernel_solved(group_metadata_ref, group_offset_ref,
                      lhs_ref, rhs_ref, o_ref, acc_ref):
    group_offsets, group_ids, m_tile_ids = group_metadata_ref
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    @pl.when(k_i == 0)
    def _zero():
        acc_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)

    acc_ref[...] += lhs_ref[...] @ rhs_ref[...]

    @pl.when(k_i == tiles_k - 1)
    def _store():
        mask = get_store_mask(grid_id, group_offsets, group_ids,
                              m_tile_ids, bm, bn)
        acc = acc_ref[...]
        o_ref[...] = jnp.where(mask, acc, o_ref[...].astype(acc.dtype))

# --- Test with variable group sizes ---
group_sizes = jnp.array([300, 212, 512], dtype=jnp.int32)
(group_offsets, group_ids, m_tile_ids), num_tiles = \
    make_group_metadata(group_sizes, M, bm)

lhs = jax.random.normal(jax.random.key(50), (M, K))
rhs = jax.random.normal(jax.random.key(51), (G, K, N))
expected = grouped_matmul_spec(lhs, rhs, group_sizes)

group_metadata = (group_offsets, group_ids, m_tile_ids)
group_offset = jnp.array([0], dtype=jnp.int32)

# YOUR TASK: Write the complete pl.pallas_call invocation.
# Replace `None` with your working code.
actual = None  # Replace with pl.pallas_call(...)(...) invocation

# --- Tests ---
total_rows = int(group_sizes.sum())
if actual is not None and jnp.allclose(actual[:total_rows], expected[:total_rows], atol=1e-2, rtol=1e-2):
    print(f"Variable groups: PASSED ✓  (shape={actual.shape})")
else:
    print("Variable groups: FAILED ✗  (fill in the cell above)")

# Test with equal groups
equal_sizes = jnp.array([256, 512, 256], dtype=jnp.int32)
(go2, gi2, mt2), nt2 = make_group_metadata(equal_sizes, M, bm)
expected2 = grouped_matmul_spec(lhs, rhs, equal_sizes)
gm2 = (go2, gi2, mt2)

# Re-run with equal groups (copy your pallas_call, updating metadata + grid)
actual2 = None  # Replace with pl.pallas_call(...)(...) invocation

if actual2 is not None and jnp.allclose(actual2, expected2, atol=1e-2, rtol=1e-2):
    print(f"Equal groups:    PASSED ✓")
else:
    print("Equal groups:    FAILED ✗")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Grid and scalar prefetch</summary>
#
# ```python
# grid_spec=pltpu.PrefetchScalarGridSpec(
#     num_scalar_prefetch=2,
#     grid=(tiles_n, num_tiles, tiles_k),
#     ...
# )
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — in_specs and out_specs</summary>
#
# ```python
# in_specs=[
#     pl.BlockSpec((bm, bk), lhs_imap),       # lhs: route by m_tile_ids
#     pl.BlockSpec((None, bk, bn), rhs_imap),  # rhs: route by group_ids, squeeze group dim
# ],
# out_specs=pl.BlockSpec((bm, bn), out_imap),  # out: route by m_tile_ids
# scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# actual = pl.pallas_call(
#     gmm_kernel_solved,
#     grid_spec=pltpu.PrefetchScalarGridSpec(
#         num_scalar_prefetch=2,
#         in_specs=[
#             pl.BlockSpec((bm, bk), lhs_imap),
#             pl.BlockSpec((None, bk, bn), rhs_imap),
#         ],
#         out_specs=pl.BlockSpec((bm, bn), out_imap),
#         grid=(tiles_n, num_tiles, tiles_k),
#         scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
#     ),
#     out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
#     interpret=True,
# )(group_metadata, group_offset, lhs, rhs)
# ```
#
# For the equal-groups test, replace `group_metadata` with `gm2`,
# `num_tiles` with `nt2` in the grid.
# </details>

# %% [markdown]
# ---
# # What's Next
#
# This notebook covered the forward-pass grouped matmul — the core of MoE
# dispatch. Production MoE kernels add several more layers:
#
# 1. **Backward pass (tgmm)**: Transpose grouped matmul computes gradients
#    w.r.t. expert weights. Accumulation is over M tiles (group rows) instead
#    of K tiles. Uses prologue/epilogue detection for group transitions.
#
# 2. **Software pipelining (`emit_pipeline`)**: Overlaps async DMA with
#    compute on TPU. Double-buffered loading hides HBM latency.
#    `dimension_semantics` tells the compiler which axes can be reordered.
#
# 3. **Production implementations**: The kernel pattern from this notebook
#    maps directly to the [tokamax](https://github.com/jax-ml/jax-triton)
#    `gmm` implementation (TPU) and
#    [MegaBlocks](https://github.com/databricks/megablocks) (GPU).
