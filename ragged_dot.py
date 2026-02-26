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
# <a href="https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/ragged_dot.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # Pallas Puzzles: Ragged Dot
#
# **9 progressive puzzles** building toward a production **ragged_dot** kernel
# for Mixture-of-Experts. You'll implement scalar prefetch, group metadata,
# masked stores, and full grouped matmul — the core of MoE dispatch on TPU.
#
# Every puzzle runs on **CPU** via `interpret=True` — no TPU needed.
#
# **Prerequisites**: Complete **basics.py** first (Pallas foundations and
# tiled matmul patterns).
#
# **Key Pallas docs**: https://docs.jax.dev/en/latest/pallas/index.html
#
# | Part | Puzzles | Focus |
# |------|---------|-------|
# | III — Scalar Prefetch | 1–5 | Runtime index maps, group metadata, masking |
# | IV — Ragged Dot | 6–9 | Grouped matmul, tgmm, pipelining |

# %% [markdown]
# ## Setup
#
# Click › to collapse this section, then click ▶ to get everything ready.

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
# # Part III: Scalar Prefetch & Group Metadata (Puzzles 1–5)

# %% [markdown]
# ---
# ## Puzzle 1: Scalar Prefetch — Permuted Batched Matmul
#
# **Goal**: Implement a **permuted batched matmul** where the mapping from
# output group → rhs group is determined at runtime by a permutation array.
#
# ### Theory
#
# In ragged_dot, the tile-to-group mapping is computed at runtime (from
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
# We skip teaching plain `GridSpec` as a separate concept — the simpler
# `grid=` kwarg to `pallas_call` (used in basics.py Puzzles 1–10) handles the basic
# case. `PrefetchScalarGridSpec` is introduced now because it's what
# production kernels use.
#
# **Note**: From this point on, puzzles use `grid_spec=` instead of
# `grid=` for `PrefetchScalarGridSpec`.

# %%
G = 4
M, K, N = 64, 64, 64

# --- Reference ---
def permuted_matmul_spec(lhs, rhs, perm):
    """lhs: (G, M, K), rhs: (G, K, N), perm: (G,) → (G, M, N)
    out[i] = lhs[i] @ rhs[perm[i]]
    """
    return jnp.stack([lhs[i] @ rhs[perm[i]] for i in range(G)])

# --- Kernel skeleton ---
def permuted_matmul_kernel(perm_ref, lhs_ref, rhs_ref, o_ref):
    # perm_ref: scalar-prefetched permutation array (in SMEM)
    # lhs_ref: (M, K) — current group's lhs
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

# %%
lhs = jax.random.normal(jax.random.key(14), (G, M, K))
rhs = jax.random.normal(jax.random.key(15), (G, K, N))
perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)  # permutation

expected = permuted_matmul_spec(lhs, rhs, perm)

actual = pl.pallas_call(
    permuted_matmul_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
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
# The index maps handle the permutation using `perm_ref[g]`. By the time the kernel runs, `rhs_ref` already points to the correct permuted group. So the kernel body is identical to basics.py Puzzle 9 — just `lhs_ref[...] @ rhs_ref[...]`.
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
# ## Puzzle 2: Group Metadata — CSR-style Tile Mapping
#
# **Goal**: Implement the `make_group_metadata` function that computes
# the tile-to-group mapping for ragged_dot. This is **pure JAX** — not a
# kernel puzzle.
#
# ### Theory
#
# In ragged_dot, `lhs` has shape `(M, K)` where rows are divided into `G`
# groups of variable sizes. We need to figure out which **tiles** belong to
# which **groups**.
#
# Given `group_sizes = [300, 212, 512]` with `bm = 128`:
#
# ![Groups and tiles](https://raw.githubusercontent.com/vorushin/pallas_puzzles/master/images/ragged-dot-puzzle2.drawio.svg)
#
# Tile at row 256 straddles the group boundary. It gets visited **twice**:
# once for group 0 (rows 256-299 are valid) and once for group 1 (rows
# 300-383 are valid). The kernel uses a **mask** to only store the valid
# rows for each visit.
#
# **Rule of thumb**: `num_tiles = tiles_m + (number of non-aligned group
# boundaries)`. Aligned boundaries don't cause extra visits.
#
# **Output arrays**:
# - `group_offsets`: `[0, 300, 512, 1024]` — cumsum with leading 0
# - `group_ids`: maps each grid index → group id
# - `m_tile_ids`: maps each grid index → which m-tile to process
# - `num_tiles`: total number of grid iterations needed
#
# The arrays can be longer than `num_tiles` (padded with the last group).

# %%
def make_group_metadata_reference(group_sizes, m, bm):
    """Simple reference implementation — O(m) but correct."""
    num_groups = len(group_sizes)
    group_offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])

    # Assign each row to a group
    row_to_group = jnp.zeros(m, dtype=jnp.int32)
    for g in range(num_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        row_to_group = row_to_group.at[start:end].set(g)

    # Assign each tile to group(s)
    tiles_m = m // bm
    group_ids_list = []
    m_tile_ids_list = []

    for t in range(tiles_m):
        tile_start = t * bm
        tile_end = (t + 1) * bm
        # Which groups touch this tile?
        groups_in_tile = jnp.unique(row_to_group[tile_start:tile_end])
        for g in groups_in_tile:
            group_ids_list.append(int(g))
            m_tile_ids_list.append(t)

    num_tiles = len(group_ids_list)

    # Pad to max possible length
    max_len = tiles_m + num_groups - 1
    group_ids = jnp.zeros(max_len, dtype=jnp.int32)
    m_tile_ids = jnp.zeros(max_len, dtype=jnp.int32)
    group_ids = group_ids.at[:num_tiles].set(jnp.array(group_ids_list, dtype=jnp.int32))
    m_tile_ids = m_tile_ids.at[:num_tiles].set(jnp.array(m_tile_ids_list, dtype=jnp.int32))
    # Pad remainder with last values
    if num_tiles < max_len:
        group_ids = group_ids.at[num_tiles:].set(group_ids_list[-1])
        m_tile_ids = m_tile_ids.at[num_tiles:].set(m_tile_ids_list[-1])

    return (group_offsets.astype(jnp.int32), group_ids, m_tile_ids), num_tiles


# %% [markdown]
# ### Your implementation — decomposed into 5 testable steps
#
# We break `make_group_metadata` into independent functions,
# each tested before combining them.
#
# ### Step 2a: Group Offsets
#
# **Goal**: Compute CSR-style prefix sum `[0, cumsum(group_sizes)]`.
#
# ```
# group_sizes = [300, 212, 512]
# group_offsets = [0, 300, 512, 1024]
#                  ^    ^    ^     ^
#                  g0   g1   g2   end
# ```

# %%
def compute_group_offsets(group_sizes):
    """[0, cumsum(group_sizes)] — maps group id → start row.

    Args:
        group_sizes: (G,) int32
    Returns:
        (G+1,) int32
    """
    # YOUR CODE HERE


# %%
assert jnp.array_equal(
    compute_group_offsets(jnp.array([256, 256, 256, 256], dtype=jnp.int32)),
    jnp.array([0, 256, 512, 768, 1024], dtype=jnp.int32))
assert jnp.array_equal(
    compute_group_offsets(jnp.array([300, 212, 512], dtype=jnp.int32)),
    jnp.array([0, 300, 512, 1024], dtype=jnp.int32))
assert jnp.array_equal(
    compute_group_offsets(jnp.array([512, 0, 512], dtype=jnp.int32)),
    jnp.array([0, 512, 512, 1024], dtype=jnp.int32))
print("Step 2a — compute_group_offsets: PASSED ✓")

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
#
# ```python
# return jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(group_sizes)])
# ```
# </details>

# %% [markdown]
# ### Step 2b: Tiles per Group

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


# %%
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
# <details><summary>Hint — Full solution</summary>
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
# ```
# group_tiles = [3, 2, 4]  →  group_ids = [0,0,0, 1,1, 2,2,2,2]
# ```

# %%
def compute_group_ids(group_tiles, num_groups, max_len):
    """Flat array mapping grid index → group id.

    Args:
        group_tiles: (G,) int32 from compute_group_tiles
        num_groups: G
        max_len: output array length (padded)
    Returns:
        (max_len,) int32
    """
    # YOUR CODE HERE


# %%
assert compute_group_ids(jnp.array([2, 2, 2, 2]), 4, 11)[:8].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
assert compute_group_ids(jnp.array([3, 2, 4]), 3, 10)[:9].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
print("Step 2c — compute_group_ids: PASSED ✓")

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
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
# ```
# group_offsets = [0, 300, 512, 1024],  bm = 128
# Group 1 starts at row 300 → inside tile 2 → extra visit
# tile_visits = [1, 1, 2, 1, 1, 1, 1, 1]
# ```

# %%
def compute_tile_visits(group_sizes, group_offsets, tiles_m, bm):
    """Visit count per tile (1 + extra for each mid-tile group boundary).

    Every tile is visited at least once. When a group boundary falls
    in the MIDDLE of a tile (not aligned to bm), that tile gets an
    extra visit. We need to count how many non-aligned boundaries
    land in each tile.

    Args:
        group_sizes: (G,) int32
        group_offsets: (G+1,) int32
        tiles_m: M // bm
        bm: tile size
    Returns:
        (tiles_m,) int32
    """
    # YOUR CODE HERE
    # 1. Find group start positions (from offsets, skip the leading 0)
    # 2. Identify which starts are non-aligned (start % bm != 0)
    #    AND belong to non-empty groups
    # 3. For non-aligned starts, compute which tile they land in (start // bm)
    # 4. Count how many non-aligned boundaries per tile (jnp.histogram)
    # 5. Result = 1 + extra_visits_per_tile


# %%
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
# <details><summary>Hint — Full solution</summary>
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
# ```
# tile_visits = [1,1,2,1,1,1,1,1]  →  m_tile_ids = [0,1,2,2,3,4,5,6,7]
# ```

# %%
def compute_m_tile_ids(tile_visits, tiles_m, max_len):
    """Flat array mapping grid index → m-tile id.

    Args:
        tile_visits: (tiles_m,) int32 from compute_tile_visits
        tiles_m: M // bm
        max_len: output array length (padded)
    Returns:
        (max_len,) int32
    """
    # YOUR CODE HERE


# %%
assert compute_m_tile_ids(jnp.array([1,1,1,1,1,1,1,1]), 8, 11)[:8].tolist() == [0,1,2,3,4,5,6,7]
assert compute_m_tile_ids(jnp.array([1,1,2,1,1,1,1,1]), 8, 10)[:9].tolist() == [0,1,2,2,3,4,5,6,7]
print("Step 2e — compute_m_tile_ids: PASSED ✓")

# %% [markdown]
# <details><summary>Hint — Full solution</summary>
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
#
# **Goal**: Chain the 5 steps above into the complete function.

# %%
def make_group_metadata_yours(group_sizes, m, bm):
    """Vectorized group metadata — chains steps 2a–2e.

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

    # YOUR CODE HERE — chain steps 2a–2e, then compute num_tiles
    # Replace this raise with your implementation:
    raise NotImplementedError("Chain compute_group_offsets → ... → compute_m_tile_ids")

    return (group_offsets, group_ids, m_tile_ids), num_tiles


# %%
# Integration tests — compare against reference
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
# <details><summary>Hint — Full solution</summary>
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
# ## Puzzle 3: Configure Your Own Scalar-Prefetch `pallas_call`
#
# **Goal**: Given a working kernel, write the **entire** `pl.pallas_call`
# invocation from scratch, including `PrefetchScalarGridSpec` with
# `num_scalar_prefetch=3`, `in_specs` with runtime index maps, `out_specs`,
# and `grid`.
#
# ### Theory
#
# This is the most challenging configuration exercise. You need to
# understand:
# - **Which args are scalar-prefetched**: metadata arrays that index maps
#   need at runtime. They appear as leading args in the call and leading
#   refs in the kernel.
# - **How index maps receive prefetch refs**: each index map gets the grid
#   indices first, then all prefetch refs. E.g.,
#   `lambda i, go, gi, mt: (mt[i], 0)` — `i` is the grid index, `go/gi/mt`
#   are the three scalar-prefetched refs.
# - **How grid size comes from num_tiles**: the grid iterates over
#   `num_tiles` (from `make_group_metadata`), not over `tiles_m`.
# - **Call argument order**: scalar-prefetched args come first, then
#   regular inputs.

# %%
M, N = 1024, 64
bm = 128
G = 3

group_sizes = jnp.array([300, 212, 512], dtype=jnp.int32)
(group_offsets, group_ids, m_tile_ids), num_tiles = \
    make_group_metadata_reference(group_sizes, M, bm)

# The kernel is provided (solved):
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
def masked_copy_spec(x, group_offsets, group_ids, m_tile_ids):
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


# %%
x = jax.random.normal(jax.random.key(26), (M, N))
expected = masked_copy_spec(x, group_offsets, group_ids, m_tile_ids)

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

# %%
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
# ## Puzzle 4: Masked Store with Group Boundaries
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
# - `group_offsets[group_id]` → start row of current group
# - `group_offsets[group_id + 1]` → end row of current group
# - `m_tile_ids[grid_id] * bm` → first row of current tile
#
# ```python
# row_indices = tile_start + jnp.arange(bm)
# mask = (row_indices >= group_start) & (row_indices < group_end)
# ```
#
# For a 2D mask (bm, N), use `jax.lax.broadcasted_iota(dtype, shape, dim)`
# — it creates an array where values along `dim` are `0, 1, 2, ...` and
# all other dimensions are broadcast. Think of it as a multi-dimensional
# `jnp.arange`:
# ```python
# broadcasted_iota(int32, (4, 3), 0) → [[0,0,0], [1,1,1], [2,2,2], [3,3,3]]
# broadcasted_iota(int32, (4, 3), 1) → [[0,1,2], [0,1,2], [0,1,2], [0,1,2]]
# ```
#
# This is exactly the `get_store_mask` pattern used in ragged_dot.

# %%
M = 1024
N = 64
bm = 128
G = 3

group_sizes = jnp.array([300, 212, 512], dtype=jnp.int32)

(group_offsets, group_ids, m_tile_ids), num_tiles = \
    make_group_metadata_reference(group_sizes, M, bm)

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


# %%
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
# ## Puzzle 5: Softmax Kernel
#
# **Goal**: Write a softmax kernel **and** configure the `pallas_call`
# yourself. Each kernel invocation processes one row-block of the full
# matrix.
#
# ### Theory
#
# Softmax is a non-matmul kernel that combines **reduction** (max, sum)
# with **elementwise** ops (exp, divide). For each row:
#
# ```
# softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
# ```
#
# Since each row fits entirely within one tile (no column-tiling needed),
# the kernel is simpler than matmul — no `@pl.when` guards, no scratch
# accumulator, just straight computation. The grid only tiles along rows.
#
# **This puzzle is also a "configure your own pallas_call" exercise** —
# you need to write both the kernel body AND the `grid`, `in_specs`,
# `out_specs`.
#
# In production (FlashAttention), the column dimension is also tiled using
# an **online softmax** algorithm that maintains running max and sum across
# column tiles. The core pattern of max → subtract → exp → normalize is
# the same.

# %%
ROWS, COLS = 256, 128
bm = 64

# --- Reference ---
def softmax_spec(x):
    """x: (ROWS, COLS) → row-wise softmax"""
    return jax.nn.softmax(x, axis=1)

# --- Kernel skeleton ---
def softmax_kernel(x_ref, o_ref):
    # x_ref: (bm, COLS) — one row block (full width)
    # o_ref: (bm, COLS) — output
    # YOUR CODE HERE
    # 1. Compute row max for numerical stability
    # 2. Subtract max, exponentiate
    # 3. Divide by row sum


# %%
x = jax.random.normal(jax.random.key(28), (ROWS, COLS))

# YOUR TASK: Write the kernel above AND define the config below.
softmax_grid = ...       # TODO: how many row blocks?
softmax_in_specs = ...   # TODO: list with one BlockSpec
softmax_out_specs = ...  # TODO: BlockSpec for output

expected = softmax_spec(x)
actual = pl.pallas_call(
    softmax_kernel,
    grid=softmax_grid,
    in_specs=softmax_in_specs,
    out_specs=softmax_out_specs,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    interpret=True,
)(x)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected:\n{expected[:4]}")
    print(f"  Got:\n{actual[:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Kernel</summary>
#
# ```python
# x = x_ref[...]
# row_max = x.max(axis=1, keepdims=True)
# exp_x = jnp.exp(x - row_max)
# row_sum = exp_x.sum(axis=1, keepdims=True)
# o_ref[...] = exp_x / row_sum
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — Config</summary>
#
# ```python
# softmax_grid = (ROWS // bm,)  # 256 // 64 = 4 row blocks
# softmax_in_specs = [pl.BlockSpec((bm, COLS), lambda i: (i, 0))]
# softmax_out_specs = pl.BlockSpec((bm, COLS), lambda i: (i, 0))
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# def softmax_kernel(x_ref, o_ref):
#     x = x_ref[...]
#     row_max = x.max(axis=1, keepdims=True)
#     exp_x = jnp.exp(x - row_max)
#     row_sum = exp_x.sum(axis=1, keepdims=True)
#     o_ref[...] = exp_x / row_sum
#
# softmax_grid = (ROWS // bm,)
# softmax_in_specs = [pl.BlockSpec((bm, COLS), lambda i: (i, 0))]
# softmax_out_specs = pl.BlockSpec((bm, COLS), lambda i: (i, 0))
# ```
# </details>

# %% [markdown]
# ---
# # Part IV: Ragged Dot (Puzzles 6–9)

# %% [markdown]
# ### Provided utilities
#
# These are the building blocks from basics.py and Part III. They're provided
# here so you can focus on the kernel logic.

# %%
def make_group_metadata(group_sizes, m, bm, *, visit_empty_groups=False):
    """Compute tile-to-group mapping for ragged_dot.

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

    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    partial_mask = ((group_offsets[:-1] % bm) == 0) | (group_sizes == 0)
    if visit_empty_groups:
        partial_mask = jnp.where(group_sizes == 0, 0, partial_mask)
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


# --- Shared index maps for ragged_dot ---
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
# ## Puzzle 6: Simple Grouped Matmul — Equal Groups, Tile-Aligned
#
# **Goal**: Implement grouped matmul for the simplest case: all groups have
# equal size and group sizes are divisible by the tile size.
#
# ### Theory
#
# This is the "easy mode" ragged_dot. With equal, tile-aligned groups:
# - No partial tiles (every tile belongs to exactly one group)
# - `group_ids` is a simple repeat: `[0,0,1,1,2,2,3,3]`
# - `m_tile_ids` = `[0,1,2,3,4,5,6,7]` (just sequential)
# - No masking needed on stores
#
# **Grid**: `(tiles_n, num_tiles, tiles_k)`
# - `tiles_n`: N dimension (parallel — independent output columns)
# - `num_tiles`: M tiles across all groups (may revisit same output)
# - `tiles_k`: K reduction dimension (accumulates)
#
# **Kernel structure** (same as tokamax `gmm`):
# 1. Get `grid_id = program_id(1)`, `k_i = program_id(2)`
# 2. Zero accumulator when `k_i == 0`
# 3. Accumulate `lhs_tile @ rhs_tile`
# 4. Store on last K tile
#
# **Index maps** (provided — study them!):
# - `lhs_imap`: `(n_i, grid_id, k_i) → (m_tile_ids[grid_id], k_i)`
# - `rhs_imap`: `(n_i, grid_id, k_i) → (group_ids[grid_id], k_i, n_i)`
# - `out_imap`: `(n_i, grid_id, k_i) → (m_tile_ids[grid_id], n_i)`
#
# The `group_ids` lookup in `rhs_imap` is what routes each tile to
# the correct group's weight matrix!

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
    """lhs: (M, K), rhs: (G, K, N), group_sizes: (G,) → (M, N)"""
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
    # group_offset_ref: unused here (for sharding)
    # lhs_ref: (bm, bk) — tile of lhs
    # rhs_ref: (bk, bn) — tile of rhs (group dim squeezed by None)
    # o_ref: (bm, bn) — output tile
    # acc_ref: (bm, bn) — scratch accumulator
    grid_id = pl.program_id(1)
    k_i = pl.program_id(2)

    # YOUR CODE HERE
    # 1. Zero accumulator on first K tile
    # 2. Accumulate tile matmul
    # 3. Store result on last K tile


# %%
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
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# The kernel body is identical to basics.py Puzzle 8: zero / accumulate / store with `@pl.when`. The index maps (already provided) handle all the group-to-tile routing via `group_ids` and `m_tile_ids`. With `None` in the rhs BlockSpec, the group dimension is squeezed — `rhs_ref` is just `(bk, bn)`.
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
# ## Puzzle 7: Full Ragged Dot — Unequal Groups
#
# **Goal**: Handle **variable group sizes** where tiles can straddle group
# boundaries. This is the real ragged_dot.
#
# ### Theory
#
# The only difference from Puzzle 6: when groups are unequal, a tile may
# be visited **multiple times** (once per group it straddles). On each visit,
# the kernel must **mask** the store so only rows belonging to the current
# group are written.
#
# `make_group_metadata` handles all the complexity — the `group_ids` and
# `m_tile_ids` arrays already encode the repeated visits. The kernel just
# needs to add the mask at store time:
#
# ```python
# mask = get_store_mask(grid_id, group_offsets, group_ids, m_tile_ids, bm, bn)
# o_ref[...] = jnp.where(mask, acc[...], o_ref[...])
# ```
#
# ```
# Tile at row 256, bm=128:
# ┌────────────────────────┐
# │ rows 256-299: group 0  │ ← Visit 1: mask=True for rows 256-299
# │ rows 300-383: group 1  │ ← Visit 2: mask=True for rows 300-383
# └────────────────────────┘
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
    # Same as Puzzle 6, but on the last K tile, apply a masked store
    # so only rows belonging to the current group are written.


# %%
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

# %% [markdown]
# <details><summary>Hint 1 of 2 — The masked store block</summary>
#
# ```python
# # Steps 1-2 are the same as Puzzle 6 (zero + accumulate)
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
# ## Puzzle 8: Transpose Grouped Matmul (tgmm)
#
# **Goal**: Implement the **backward-pass** kernel: `tgmm` computes the
# gradient w.r.t. the RHS weight matrices.
#
# ### Theory
#
# In the backward pass of ragged_dot, we need:
# - `dlhs = dout @ rhs[g].T` (gradient w.r.t. lhs — another gmm)
# - `drhs[g] = lhs[g_rows].T @ dout[g_rows]` (gradient w.r.t. rhs — this is tgmm)
#
# **tgmm** computes `lhs.T @ rhs` accumulated per group:
# - `lhs`: `(M, K)` (original lhs, or the transposed `(K, M)` passed as `(M, K)`)
# - `rhs`: `(M, N)` (dout)
# - `out`: `(G, K, N)` — one output per group
#
# **Key difference from gmm**: In gmm, multiple K tiles contribute to the
# same output tile (accumulate over K). In tgmm, multiple **M tiles** from
# the same group contribute to the same output tile (accumulate over group
# rows). This requires a different accumulation pattern:
#
# - **Prologue** (entering new group): zero the accumulator
# - **Body**: accumulate `lhs_tile.T @ rhs_tile`, masked by group boundaries
# - **Epilogue** (leaving group): store accumulator to output
#
# Group transitions detected by comparing consecutive group_ids.
#
# ```
# Grid iteration:  0   1   2   3   4   5   6   7   8   9
# group_ids:      [0,  0,  0,  1,  1,  2,  2,  2,  2,  2]
#                  P       E  P    E  P               E
#                  P = prologue (zero), E = epilogue (store)
# ```
#
# **New concepts in this puzzle:**
#
# - **`visit_empty_groups=True`**: If a group has zero rows, we still
#   need one grid iteration for it — so the kernel can zero and store
#   an empty accumulator. Without this, the output for that group
#   would contain garbage.
#
# - **`pl.num_programs(axis)`**: Returns the total number of grid
#   iterations along an axis (like `gridDim` in CUDA). Used here to
#   detect the very last iteration for the final epilogue.
#
# - **Grid axis order is `(tiles_n, tiles_k, num_tiles)`** — note
#   that `num_tiles` is now on axis 2 (not axis 1 like in gmm).
#   This is because the M-tile iteration is the "reduction" axis
#   in tgmm (we accumulate across M tiles), so it must be the
#   innermost `"arbitrary"` dimension for correct pipelining.

# %%
G = 3
M, K, N = 1024, 128, 128
bm, bk, bn = 128, 128, 128

group_sizes = jnp.array([300, 340, 384], dtype=jnp.int32)
tiles_k = K // bk
tiles_n = N // bn

(group_offsets, group_ids, m_tile_ids), num_tiles = \
    make_group_metadata(group_sizes, M, bm, visit_empty_groups=True)

print(f"group_sizes={group_sizes.tolist()}, num_tiles={num_tiles}")
print(f"group_ids={group_ids[:num_tiles].tolist()}")

# --- Reference ---
def tgmm_spec(lhs_t, rhs, group_sizes):
    """lhs_t: (K, M), rhs: (M, N) → (G, K, N)
    Computes lhs_t[:, g_start:g_end] @ rhs[g_start:g_end, :] per group.
    """
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])
    G = len(group_sizes)
    K, N = lhs_t.shape[0], rhs.shape[1]
    out = jnp.zeros((G, K, N), dtype=jnp.float32)
    for g in range(G):
        s, e = int(offsets[g]), int(offsets[g + 1])
        if s < e:
            out = out.at[g].set(lhs_t[:, s:e] @ rhs[s:e, :])
    return out

# --- Kernel skeleton ---
def tgmm_kernel(group_metadata_ref, group_offset_ref,
                lhs_ref, rhs_ref, o_ref, acc_ref):
    # lhs_ref: (bm, bk) — tile of lhs (M, K)
    # rhs_ref: (bm, bn) — tile of rhs
    # o_ref: (bk, bn) — output tile for one group (None dim squeezed)
    # acc_ref: (bk, bn) — scratch accumulator
    group_offsets, group_ids, m_tile_ids = group_metadata_ref
    grid_id = pl.program_id(2)  # tgmm grid: (tiles_n, tiles_k, num_tiles)

    # YOUR CODE HERE
    # 1. Detect group transitions: when does a new group start? end?
    # 2. Zero accumulator at the start of each group
    # 3. Accumulate masked lhs.T @ rhs
    # 4. Store accumulator at the end of each group


# --- Index maps for tgmm ---
def tgmm_lhs_imap(n_i, k_i, grid_id, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], k_i)

def tgmm_rhs_imap(n_i, k_i, grid_id, group_meta_ref, group_offset_ref):
    _, _, m_tile_ids = group_meta_ref
    return (m_tile_ids[grid_id], n_i)

def tgmm_out_imap(n_i, k_i, grid_id, group_meta_ref, group_offset_ref):
    _, group_ids, _ = group_meta_ref
    return (group_ids[grid_id], k_i, n_i)


# %%
lhs_t = jax.random.normal(jax.random.key(50), (K, M))
rhs = jax.random.normal(jax.random.key(51), (M, N))
expected = tgmm_spec(lhs_t, rhs, group_sizes)

# tgmm works on (M, K) internally — transpose lhs
lhs = lhs_t.T  # (M, K)

group_metadata = (group_offsets, group_ids, m_tile_ids)
group_offset = jnp.array([0], dtype=jnp.int32)

actual = pl.pallas_call(
    tgmm_kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            pl.BlockSpec((bm, bk), tgmm_lhs_imap),
            pl.BlockSpec((bm, bn), tgmm_rhs_imap),
        ],
        out_specs=pl.BlockSpec((None, bk, bn), tgmm_out_imap),
        grid=(tiles_n, tiles_k, num_tiles),
        scratch_shapes=[pltpu.VMEM((bk, bn), jnp.float32)],
    ),
    out_shape=jax.ShapeDtypeStruct((G, K, N), jnp.float32),
    interpret=True,
)(group_metadata, group_offset, lhs, rhs)

if jnp.allclose(actual, expected, atol=1e-1, rtol=1e-2):
    print(f"PASSED ✓  (shape={actual.shape})")
else:
    max_err = float(jnp.max(jnp.abs(actual - expected)))
    print(f"FAILED ✗  max error: {max_err:.6f}")
    print(f"  Expected[0,:2,:4]:\n{expected[0,:2,:4]}")
    print(f"  Actual[0,:2,:4]:\n{actual[0,:2,:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 3 — Prologue/epilogue detection</summary>
#
# ```python
# group = group_ids[grid_id]
# prev_group = group_ids[jnp.where(grid_id > 0, grid_id - 1, 0)]
# is_prologue = (grid_id == 0) | (group != prev_group)
#
# is_end = grid_id == (pl.num_programs(2) - 1)
# next_group = group_ids[jnp.where(is_end, grid_id, grid_id + 1)]
# is_epilogue = is_end | (group != next_group)
# ```
# </details>
#
# <details><summary>Hint 2 of 3 — Masked accumulation</summary>
#
# ```python
# @pl.when(is_prologue)
# def _zero():
#     acc_ref[...] = jnp.zeros((bk, bn), dtype=jnp.float32)
#
# mask_lhs = get_store_mask(grid_id, group_offsets, group_ids,
#                            m_tile_ids, bm, bk)
# mask_rhs = get_store_mask(grid_id, group_offsets, group_ids,
#                            m_tile_ids, bm, bn)
# lhs_masked = jnp.where(mask_lhs, lhs_ref[...], 0)
# rhs_masked = jnp.where(mask_rhs, rhs_ref[...], 0)
# acc_ref[...] += lhs_masked.T @ rhs_masked
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# group = group_ids[grid_id]
# prev_group = group_ids[jnp.where(grid_id > 0, grid_id - 1, 0)]
# is_prologue = (grid_id == 0) | (group != prev_group)
#
# is_end = grid_id == (pl.num_programs(2) - 1)
# next_group = group_ids[jnp.where(is_end, grid_id, grid_id + 1)]
# is_epilogue = is_end | (group != next_group)
#
# group_size = group_offsets[group + 1] - group_offsets[group]
# nonzero_gs = group_size > 0
#
# @pl.when(is_prologue)
# def _zero():
#     acc_ref[...] = jnp.zeros((bk, bn), dtype=jnp.float32)
#
# @pl.when(nonzero_gs)
# def _compute():
#     mask_lhs = get_store_mask(grid_id, group_offsets, group_ids,
#                                m_tile_ids, bm, bk)
#     mask_rhs = get_store_mask(grid_id, group_offsets, group_ids,
#                                m_tile_ids, bm, bn)
#     lhs_masked = jnp.where(mask_lhs, lhs_ref[...], 0)
#     rhs_masked = jnp.where(mask_rhs, rhs_ref[...], 0)
#     acc_ref[...] += lhs_masked.T @ rhs_masked
#
# @pl.when(is_epilogue)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 9: Understanding `emit_pipeline` — Annotated Walkthrough
#
# This is a **reading exercise**, not a coding puzzle. We walk through the
# tokamax `custom_buffered_pallas_call` to understand how production kernels
# use software pipelining for async DMA on TPU.
#
# ### Why pipelining?
#
# On TPU, data lives in **HBM** (32 GB, high bandwidth but high latency).
# Computation happens in **VMEM** (small, fast SRAM). Without pipelining:
#
# ```
# Time:  [DMA load] [compute] [DMA load] [compute] ...
#        ^^^idle^^^            ^^^idle^^^
# ```
#
# With double-buffered pipelining:
#
# ```
# DMA:      [load 0] [load 1] [load 2] [load 3] ...
# Compute:           [comp 0] [comp 1] [comp 2] ...
# ```
#
# The DMA engine and compute engine run in parallel, hiding memory latency.
#
# ### The `emit_pipeline` wrapper
#
# `pltpu.emit_pipeline` transforms a simple kernel into a pipelined one:
#
# ```python
# pltpu.emit_pipeline(
#     kernel_fn,          # Your original kernel
#     grid=grid,          # Iteration space
#     in_specs=in_specs,  # How to tile inputs
#     out_specs=out_specs, # How to tile outputs
#     dimension_semantics=("parallel", "arbitrary", "arbitrary"),
# )
# ```
#
# **`dimension_semantics`** tells the compiler about loop dependencies:
# - `"parallel"`: iterations are independent → can be reordered freely
# - `"arbitrary"`: iterations may have dependencies → must execute in order
#
# For ragged_dot: `(tiles_n, num_tiles, tiles_k)`:
# - `tiles_n` is `"parallel"` — different output columns are independent
# - `num_tiles` is `"arbitrary"` — tiles may share output locations
# - `tiles_k` is `"arbitrary"` — accumulation across K must be ordered

# %%
# Annotated version of the tokamax custom_buffered_pallas_call
# (Read and understand — no code to write)

import dataclasses

def annotated_custom_buffered_pallas_call(kernel, out_shape, grid_spec,
                                          compiler_params,
                                          input_buffer_count=None, **kw):
    """Wraps a kernel with emit_pipeline for async DMA pipelining.

    The outer pallas_call sees all data in HBM. Inside, emit_pipeline
    creates a software-pipelined loop that overlaps DMA with compute.
    """
    num_scalar_prefetch = grid_spec.num_scalar_prefetch

    def pipeline(*args_refs):
        # === Phase 1: Unpack grid and SMEM refs ===
        smem_refs = args_refs[1 : num_scalar_prefetch + 1]

        # === Phase 2: Bind SMEM refs to index maps ===
        def _augment_blockspec(bs):
            index_map_ = lambda *idxs: bs.index_map(*idxs, *smem_refs)
            return pl.BlockSpec(bs.block_shape, index_map_)

        in_specs = jax.tree.map(_augment_blockspec, grid_spec.in_specs)
        out_specs = jax.tree.map(_augment_blockspec, grid_spec.out_specs)

        # === Phase 3: Separate input/output/scratch refs ===
        input_output_refs = args_refs[num_scalar_prefetch + 1:]

        # === Phase 4: Emit the pipeline! ===
        pltpu.emit_pipeline(
            lambda *args: kernel(*smem_refs, *args),
            grid=grid_spec.grid,
            in_specs=in_specs,
            out_specs=out_specs,
            dimension_semantics=compiler_params.dimension_semantics,
        )(*input_output_refs)

    # The OUTER pallas_call has NO grid — single invocation.
    return pl.pallas_call(
        pipeline,
        out_shape,
        compiler_params=dataclasses.replace(compiler_params, dimension_semantics=()),
        in_specs=(
            jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.SMEM),
                        tuple(range(num_scalar_prefetch + 1))),
            jax.tree.map(lambda _: pl.BlockSpec(memory_space=pl.ANY),
                        tuple(grid_spec.in_specs)),
        ),
        out_specs=jax.tree.map(lambda _: pl.BlockSpec(memory_space=pl.ANY),
                               grid_spec.out_specs),
        **kw,
    )

print("emit_pipeline annotated walkthrough loaded.")
print("Study the code above — on real TPU, this is what makes the kernel fast!")

# %% [markdown]
# ### Comprehension questions
#
# Answer these by reading the annotated code above:
#
# 1. **Why does the outer `pallas_call` have no grid?** What happens
#    inside `emit_pipeline` that replaces the grid?
#
# 2. **What does `_augment_blockspec` do?** Why can't we pass the
#    original `grid_spec.in_specs` directly to `emit_pipeline`?
#
# 3. **Why is `tiles_n` labeled `"parallel"` but `num_tiles` and
#    `tiles_k` are `"arbitrary"`?** What would go wrong if we marked
#    `tiles_k` as `"parallel"`?
#
# 4. **Double buffering requires 2× the VMEM.** Why is this trade-off
#    worth it on TPU?
#
# <details><summary>Answers</summary>
#
# 1. The outer `pallas_call` runs once. Inside, `emit_pipeline` creates
#    its own software-pipelined loop over the grid, overlapping DMA with
#    compute. The grid iteration is "inlined" into the pipeline.
#
# 2. `_augment_blockspec` rebinds the index maps to include the SMEM
#    refs. The original index maps expect `(grid_idx, *smem_refs)`,
#    but `emit_pipeline` only passes grid indices. The wrapper curries
#    in the SMEM refs.
#
# 3. `"parallel"` means iterations are independent and can be reordered
#    or executed concurrently. `tiles_k` has accumulation dependencies —
#    K tile 2 adds to the same accumulator as K tile 1. Marking it
#    `"parallel"` would allow reordering, producing wrong results.
#    `num_tiles` has masked-store dependencies (boundary tiles are
#    visited by multiple groups sequentially).
#
# 4. TPU HBM latency is high (~100s of cycles). Without pipelining,
#    the MXU sits idle during every DMA load. With double buffering,
#    the MXU computes on buffer A while DMA fills buffer B. The 2×
#    VMEM cost is small compared to the throughput gain (often 2-3×
#    higher MFU).
# </details>

