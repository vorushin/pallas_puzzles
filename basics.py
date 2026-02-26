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
# <a href="https://colab.research.google.com/github/vorushin/pallas_puzzles/blob/master/basics.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # Pallas Puzzles: Basics
#
# **Pallas** is JAX's kernel language for writing custom operations that run on
# TPU (and GPU). Think of it as "NumPy inside a tile" — you write a function
# that operates on small blocks of data, and `pallas_call` maps that function
# across a grid of tiles covering the full arrays.
#
# This notebook contains **11 progressive puzzles** that
# build your Pallas intuition from scratch, culminating in tiled matmul
# with fusion. Every puzzle runs on **CPU**
# via `interpret=True` — no TPU needed. Fill in the kernel skeletons and
# run the test cells.
#
# **Prerequisites**: solid JAX/NumPy. No prior Pallas required.
#
# **Key Pallas docs**: https://docs.jax.dev/en/latest/pallas/index.html
#
# | Part | Puzzles | Focus |
# |------|---------|-------|
# | I — Foundations | 1–7 | Refs, grids, BlockSpec, `@pl.when` |
# | II — Matmul Patterns | 8–11 | Scratch, accumulation, fusion |

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
# # Part I: Foundations (Puzzles 1–7)

# %% [markdown]
# ---
# ## Puzzle 1: Hello Pallas — Constant Add
#
# **Goal**: Write a kernel that adds 10 to every element.
#
# ### Theory
#
# A Pallas kernel is a Python function that receives **Ref** objects — typed
# pointers to blocks of memory. You read from a Ref with `ref[...]` (loads the
# entire block) and write with `ref[...] = value`. The `[...]` (Ellipsis)
# means "all elements" — it's the standard way to read or write an entire Ref
# in Pallas. You can also use slicing like `ref[0:4]`, but full `ref[...]`
# reads/writes are by far the most common pattern.
#
# `pallas_call` invokes your kernel once for each point in a **grid**. With an
# empty grid `()`, the kernel runs exactly once and sees the full arrays.
#
# ```
#   x_ref  →  [ read ]
#                 ↓
#            x + 10.0
#                 ↓
#   o_ref  ←  [ write ]
# ```

# %%
N = 32

# --- Reference (spec) ---
def add10_spec(x):
    """x: (N,) → x + 10"""
    return x + 10.0

# --- Kernel skeleton ---
def add10_kernel(x_ref, o_ref):
    # x_ref: Ref to input block (shape (N,))
    # o_ref: Ref to output block (shape (N,))
    # YOUR CODE HERE


# %%
x = jax.random.uniform(jax.random.key(0), (N,))

expected = add10_spec(x)
actual = pl.pallas_call(
    add10_kernel,
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
# <details><summary>Hint</summary>
#
# Read the entire input with `x_ref[...]`, add 10, write to `o_ref[...] = ...`
# </details>

# %% [markdown]
# ---
# ## Puzzle 2: Tiled Vector Add
#
# **Goal**: Add two vectors using a 1D grid with block tiling.
#
# ### Theory
#
# In Puzzle 1 we used `grid=()` — an empty grid — so the kernel ran once
# and saw the entire array. That's fine for tiny inputs, but real TPU
# kernels work on arrays with millions of elements. We need to split
# them into **blocks** (also called tiles) and process one block per
# kernel invocation.
#
# That's what the **grid** is for. It's a tuple that says "how many times
# to invoke the kernel, and along which dimensions":
#
# ```
# grid=()      → 1 invocation  (Puzzle 1)
# grid=(4,)    → 4 invocations, numbered i=0,1,2,3
# grid=(2, 3)  → 6 invocations, numbered (i,j) for i=0,1 and j=0,1,2
# ```
#
# With `grid=(4,)` and a 256-element vector, we get 4 invocations that
# each process a 64-element block:
#
# ```
# Array (256 elements):
# [████████ ████████ ████████ ████████]
#  block 0   block 1   block 2   block 3
#  i=0       i=1       i=2       i=3
# ```
#
# But the grid alone only says *how many* invocations — it doesn't say
# which slice of the array each invocation sees. That's the job of
# **BlockSpec**, which pairs a block shape with an **index map**:
#
# ```python
# BlockSpec(block_shape, index_map)
# ```
#
# The index map is a function from grid indices → block position. For
# the simplest case, `lambda i: (i,)` means "invocation `i` gets block `i`":
#
# ```
# grid=(4,)  +  BlockSpec((64,), lambda i: (i,))
#
# i=0 → index_map(0) = (0,) → array[0:64]
# i=1 → index_map(1) = (1,) → array[64:128]
# i=2 → index_map(2) = (2,) → array[128:192]
# i=3 → index_map(3) = (3,) → array[192:256]
# ```
#
# Inside the kernel, `pl.program_id(axis)` returns the current grid index.
# But with `BlockSpec`, the Refs already point to the right block — so
# you often don't need `program_id` at all for element-wise ops!
# The kernel body stays identical whether you have 4 blocks or 400.

# %%
N = 256   # vector length
bm = 64   # tile (block) size — each kernel invocation processes bm elements

# --- Reference ---
def vadd_spec(x, y):
    """x, y: (N,) → x + y"""
    return x + y

# --- Kernel skeleton ---
def vadd_kernel(x_ref, y_ref, o_ref):
    # Each invocation sees a (bm,) slice thanks to BlockSpec
    # YOUR CODE HERE


# %%
x = jax.random.uniform(jax.random.key(1), (N,))
y = jax.random.uniform(jax.random.key(2), (N,))

expected = vadd_spec(x, y)
actual = pl.pallas_call(
    vadd_kernel,
    grid=(N // bm,),              # 256 // 64 = 4 invocations
    in_specs=[
        pl.BlockSpec((bm,), lambda i: (i,)),  # x: invocation i → block i
        pl.BlockSpec((bm,), lambda i: (i,)),  # y: invocation i → block i
    ],
    out_specs=pl.BlockSpec((bm,), lambda i: (i,)),  # out: invocation i → block i
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    interpret=True,
)(x, y)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected:\n{expected[:4]}")
    print(f"  Got:\n{actual[:4]}")

# %% [markdown]
# <details><summary>Hint</summary>
#
# The BlockSpecs handle all the slicing. Your kernel just needs:
# `o_ref[...] = x_ref[...] + y_ref[...]`
# </details>

# %% [markdown]
# ---
# ## Puzzle 3: Reversed Block Add — Index Map Manipulation
#
# **Goal**: Add `x` to a **block-reversed** version of `y` by changing
# only the index map. The kernel body is identical to Puzzle 2!
#
# ### Theory
#
# The index map in a `BlockSpec` controls **which block** each grid
# invocation sees. So far every index map was `lambda i: (i,)` — grid
# invocation `i` sees block `i` (sequential order). But the map can
# be any function: `lambda i: (3 - i,)` would read blocks in reverse.
#
# ```
# y = [  y₀  |  y₁  |  y₂  |  y₃  ]      4 blocks, bm=64
#
# Normal index map     λi: (i,)         → y₀  y₁  y₂  y₃
# Reversed index map   λi: (3-i,)       → y₃  y₂  y₁  y₀
#
# x:          [  x₀  ][  x₁  ][  x₂  ][  x₃  ]
# y reversed: [  y₃  ][  y₂  ][  y₁  ][  y₀  ]
# result:     [x₀+y₃ ][x₁+y₂ ][x₂+y₁ ][x₃+y₀]
# ```
#
# This is the key insight behind all advanced Pallas kernels: the index
# map decides what data the kernel sees, while the kernel body stays
# simple and generic.

# %%
N = 256   # vector length (same as Puzzle 2)
bm = 64   # tile size
num_blocks = N // bm   # 4 blocks total

# --- Reference ---
def vadd_rev_spec(x, y):
    """x, y: (N,) → x + block_reverse(y)"""
    y_rev = y.reshape(num_blocks, bm)[::-1].reshape(N)
    return x + y_rev

# Kernel is provided (same body as Puzzle 2):
def vadd_rev_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


# %%
x = jax.random.uniform(jax.random.key(100), (N,))
y = jax.random.uniform(jax.random.key(101), (N,))

# YOUR TASK: Fix the y BlockSpec so it reads blocks in reversed order.
# Only the y index map needs to change — x and out are correct.
expected = vadd_rev_spec(x, y)
actual = pl.pallas_call(
    vadd_rev_kernel,
    grid=(num_blocks,),
    in_specs=[
        pl.BlockSpec((bm,), lambda i: (i,)),              # x: block i (correct)
        pl.BlockSpec((bm,), lambda i: (i,)),              # y: block i — FIX THIS
    ],
    out_specs=pl.BlockSpec((bm,), lambda i: (i,)),
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    interpret=True,
)(x, y)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected:\n{expected[:4]}")
    print(f"  Got:\n{actual[:4]}")

# %% [markdown]
# <details><summary>Hint</summary>
#
# The y index map should map grid index `i` to the reversed block position.
# With 4 blocks, `i=0 → block 3`, `i=1 → block 2`, etc.:
# ```python
# pl.BlockSpec((bm,), lambda i: (num_blocks - 1 - i,))
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 4: 2D Element-wise with 2D Grid
#
# **Goal**: Multiply every element of a 2D matrix by 2, using a 2D grid of
# blocks.
#
# ### Theory
#
# Grids can be multi-dimensional. A `grid=(4, 4)` creates 16 invocations,
# each indexed by `(i, j)`. Use `pl.program_id(0)` for `i` and
# `pl.program_id(1)` for `j`.
#
# BlockSpecs for 2D: `BlockSpec((bm, bn), lambda i, j: (i, j))`
# means "tile `(i,j)` is the block at rows `[i*bm:(i+1)*bm]`,
# cols `[j*bn:(j+1)*bn]`".
#
# **Key insight**: The kernel body is identical to Puzzle 2 — just
# `o_ref[...] = f(x_ref[...])`. The BlockSpec handles all the 2D
# indexing. This is the power of Pallas's tiling abstraction: the
# kernel doesn't care whether the grid is 1D, 2D, or 3D.
#
# ```
# Matrix (128×128):
# ┌────┬────┬────┬────┐
# │0,0 │0,1 │0,2 │0,3 │  ← row blocks
# ├────┼────┼────┼────┤
# │1,0 │1,1 │1,2 │1,3 │
# ├────┼────┼────┼────┤
# │2,0 │2,1 │2,2 │2,3 │
# ├────┼────┼────┼────┤
# │3,0 │3,1 │3,2 │3,3 │
# └────┴────┴────┴────┘
#        32×32 each
# ```

# %%
M, N = 128, 128
bm, bn = 32, 32

# --- Reference ---
def mul2d_spec(x):
    """x: (M, N) → x * 2"""
    return x * 2.0

# --- Kernel skeleton ---
def mul2d_kernel(x_ref, o_ref):
    # YOUR CODE HERE


# %%
x = jax.random.uniform(jax.random.key(3), (M, N))

expected = mul2d_spec(x)
actual = pl.pallas_call(
    mul2d_kernel,
    grid=(M // bm, N // bn),
    in_specs=[pl.BlockSpec((bm, bn), lambda i, j: (i, j))],
    out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
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
# <details><summary>Hint</summary>
#
# Same as Puzzle 2 — `o_ref[...] = x_ref[...] * 2.0`. The 2D BlockSpec
# handles the tiling.
# </details>

# %% [markdown]
# ---
# ## Puzzle 5: Outer Product (Broadcasting Inside Kernels)
#
# **Goal**: Compute the outer product `a[:, None] * b[None, :]` for two
# vectors, producing a 2D matrix.
#
# ### Theory
#
# Inputs and output can have **different shapes**. Here:
# - `a`: shape `(M,)` → BlockSpec tiles along dim 0
# - `b`: shape `(N,)` → BlockSpec tiles along dim 0 (it's 1D)
# - `out`: shape `(M, N)` → BlockSpec tiles along both dims
#
# The index maps must line up correctly:
# - For `a`: grid `(i, j)` → tile `(i,)` (only depends on row)
# - For `b`: grid `(i, j)` → tile `(j,)` (only depends on col)
# - For `out`: grid `(i, j)` → tile `(i, j)`
#
# ```
#                       b (N=64)
#                  b₀ (j=0)    b₁ (j=1)
#
#  a (M=128)  i=0   a₀ × b₀     a₀ × b₁
#             i=1   a₁ × b₀     a₁ × b₁
#             i=2   a₂ × b₀     a₂ × b₁
#             i=3   a₃ × b₀     a₃ × b₁
#
#                    output (128×64)
#
# Each tile (i,j): a_ref shape (bm,), b_ref shape (bn,)
#   -> broadcast to (bm, bn) via [:, None] * [None, :]
# ```
#
# Inside the kernel, `a_ref` has shape `(bm,)` and `b_ref` has shape `(bn,)`.
# You need to broadcast them: `a_ref[...][:, None] * b_ref[...][None, :]`
# produces shape `(bm, bn)`.

# %%
M, N = 128, 64
bm, bn = 32, 32

# --- Reference ---
def outer_spec(a, b):
    """a: (M,), b: (N,) → (M, N)"""
    return a[:, None] * b[None, :]

# --- Kernel skeleton ---
def outer_kernel(a_ref, b_ref, o_ref):
    # a_ref: (bm,) — a slice of vector a
    # b_ref: (bn,) — a slice of vector b
    # o_ref: (bm, bn) — output tile
    # YOUR CODE HERE


# %%
a = jax.random.uniform(jax.random.key(4), (M,))
b = jax.random.uniform(jax.random.key(5), (N,))

expected = outer_spec(a, b)
actual = pl.pallas_call(
    outer_kernel,
    grid=(M // bm, N // bn),
    in_specs=[
        pl.BlockSpec((bm,), lambda i, j: (i,)),
        pl.BlockSpec((bn,), lambda i, j: (j,)),
    ],
    out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    interpret=True,
)(a, b)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected:\n{expected[:4]}")
    print(f"  Got:\n{actual[:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# You need to broadcast `a_ref[...]` (shape `(bm,)`) and `b_ref[...]` (shape `(bn,)`) to produce shape `(bm, bn)`. Use NumPy-style broadcasting: add a new axis with `[:, None]` and `[None, :]`.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# o_ref[...] = a_ref[...][:, None] * b_ref[...][None, :]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 6: Configure Your Own `pallas_call` — Vector Add
#
# **Goal**: Given a working kernel, fill in the `grid`, `in_specs`, and
# `out_specs` arguments yourself.
#
# ### Theory
#
# So far we've given you the `pallas_call` setup and you only wrote the
# kernel body. Now it's your turn to configure the call. You need:
#
# 1. **`grid`**: a tuple specifying how many tiles in each dimension.
#    For a 1D vector of length `N` with tile size `bm`: `grid = (N // bm,)`.
#
# 2. **`in_specs`**: a list of `BlockSpec`, one per input. Each says what
#    shape the kernel sees and how grid indices map to tile positions.
#
# 3. **`out_specs`**: a single `BlockSpec` for the output.
#
# The kernel below is the solved version from Puzzle 2. Your task is to
# wire up the tiling so it processes `N`-element vectors in blocks of
# `bm`.

# %%
N = 256
bm = 64

def vadd_spec6(x, y):
    return x + y

# Kernel is provided (solved):
def vadd_kernel_solved(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


# %%
x = jax.random.uniform(jax.random.key(10), (N,))
y = jax.random.uniform(jax.random.key(11), (N,))

# YOUR TASK: Define grid, in_specs, out_specs to tile the computation
# into bm-sized blocks. The kernel processes one block per invocation.
vadd_grid = ...       # TODO: how many tiles? (should be a tuple)
vadd_in_specs = ...   # TODO: list of BlockSpec, one per input
vadd_out_specs = ...  # TODO: BlockSpec for output

expected = vadd_spec6(x, y)
actual = pl.pallas_call(
    vadd_kernel_solved,
    grid=vadd_grid,
    in_specs=vadd_in_specs,
    out_specs=vadd_out_specs,
    out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    interpret=True,
)(x, y)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected:\n{expected[:4]}")
    print(f"  Got:\n{actual[:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — What to fill in</summary>
#
# ```python
# vadd_grid = (N // bm,)  # 256 // 64 = 4 tiles
# vadd_in_specs = [
#     pl.BlockSpec((bm,), lambda i: (i,)),  # one per input
#     pl.BlockSpec((bm,), lambda i: (i,)),
# ]
# vadd_out_specs = pl.BlockSpec((bm,), lambda i: (i,))
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# vadd_grid = (N // bm,)
# vadd_in_specs = [
#     pl.BlockSpec((bm,), lambda i: (i,)),
#     pl.BlockSpec((bm,), lambda i: (i,)),
# ]
# vadd_out_specs = pl.BlockSpec((bm,), lambda i: (i,))
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 7: Reduction — Row Sum with `@pl.when`
#
# **Goal**: Sum each row of a matrix. The K dimension is tiled, so the
# kernel must **accumulate** partial sums across multiple invocations.
#
# ### Theory
#
# Matmul and many other operations have a **reduction dimension** (K) that
# gets summed over. In Pallas, we tile K and iterate:
#
# 1. Each grid point `(i, k)` processes row-block `i`, K-block `k`
# 2. On the first K-block (`k == 0`): **zero** the output
# 3. On every K-block: **accumulate** the partial sum
#
# ```
# x: (ROWS, COLS)
#     ┌──────┬──────┬──────┬──────┐
# r=0 │ k=0  │ k=1  │ k=2  │ k=3  │  → sum → out[0:bm]
#     ├──────┼──────┼──────┼──────┤
# r=1 │ k=0  │ k=1  │ k=2  │ k=3  │  → sum → out[bm:2*bm]
#     └──────┴──────┴──────┴──────┘
# ```
#
# **`@pl.when(condition)`** is Pallas's conditional execution primitive.
# It compiles to **predicated execution** on TPU — no branch divergence
# penalty. Use it to guard operations that should only run on certain
# grid iterations:
#
# ```python
# @pl.when(k_i == 0)           # only runs when k_i is 0
# def _():
#     acc[...] = jnp.zeros(...)
# ```
#
# This is the key pattern for all reduction kernels: conditionally zero
# the accumulator on the first tile, accumulate on every tile, and
# (for matmul) conditionally store on the last tile.

# %%
ROWS, COLS = 16, 256
bm, bk = 16, 64
tiles_k = COLS // bk

# --- Reference ---
def rowsum_spec(x):
    """x: (ROWS, COLS) → (ROWS,)"""
    return x.sum(axis=1)

# --- Kernel skeleton ---
def rowsum_kernel(x_ref, o_ref):
    # x_ref: (bm, bk) — one tile of x
    # o_ref: (bm,) — accumulator for this row block
    # Grid: (ROWS // bm, COLS // bk) — iterates (row_block, k_block)
    k_i = pl.program_id(1)
    # YOUR CODE HERE
    # 1. On first k tile (k_i == 0), initialize the output
    # 2. Add this tile's contribution to the running sum


# %%
x = jax.random.uniform(jax.random.key(6), (ROWS, COLS))
expected = rowsum_spec(x)
actual = pl.pallas_call(
    rowsum_kernel,
    grid=(ROWS // bm, tiles_k),
    in_specs=[pl.BlockSpec((bm, bk), lambda i, k: (i, k))],
    out_specs=pl.BlockSpec((bm,), lambda i, k: (i,)),
    out_shape=jax.ShapeDtypeStruct((ROWS,), jnp.float32),
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
# <details><summary>Hint 1 of 3 — Approach</summary>
#
# Use `@pl.when(k_i == 0)` to conditionally zero the output on the first K tile. On every tile, accumulate the partial row sum with `o_ref[...] += x_ref[...].sum(axis=1)`.
# </details>
#
# <details><summary>Hint 2 of 3 — Pattern skeleton</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     o_ref[...] = jnp.zeros((bm,), dtype=jnp.float32)
#
# o_ref[...] += ...  # partial row sum of x_ref
# ```
# </details>
#
# <details><summary>Hint 3 of 3 — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     o_ref[...] = jnp.zeros((bm,), dtype=jnp.float32)
#
# o_ref[...] += x_ref[...].sum(axis=1)
# ```
# </details>

# %% [markdown]
# ---
# # Part II: Matmul Patterns (Puzzles 8–11)

# %% [markdown]
# ---
# ## Puzzle 8: Tiled Matmul with Scratch Accumulator
#
# **Goal**: Implement tiled matrix multiplication `C = A @ B` using a scratch
# buffer for accumulation across K tiles.
#
# ### Theory
#
# This is the bread-and-butter of Pallas. Tiled matmul has a **3D grid**:
# `(tiles_m, tiles_n, tiles_k)`. For each `(m, n)` output tile, we iterate
# over K tiles (K for "Kontracting" dimension) and accumulate
# `A_tile @ B_tile`.
#
# ```
#   A (128×256)              B (256×128)           C (128×128)
#   ┌──────────┬──────────┐  ┌──────────┬────────┐  ┌────────┬────────┐
#   │   A0,0   │   A0,1   │  │   B0,0   │  B0,1  │  │  C0,0  │  C0,1  │
#   │  64×128  │  64×128  │  │  128×64  │ 128×64 │  │  64×64 │  64×64 │
#   ├──────────┼──────────┤  ├──────────┼────────┤  ├────────┼────────┤
#   │   A1,0   │   A1,1   │  │   B1,0   │  B1,1  │  │  C1,0  │  C1,1  │
#   │  64×128  │  64×128  │  │  128×64  │ 128×64 │  │  64×64 │  64×64 │
#   └──────────┴──────────┘  └──────────┴────────┘  └────────┴────────┘
#
# To compute C[0,0], sweep k over the K dimension:
#
#   k=0: acc  = A0,0 @ B0,0   (64×128) @ (128×64) → (64×64)
#   k=1: acc += A0,1 @ B1,0   (64×128) @ (128×64) → (64×64)
#         └─→ store acc → C[0,0]
# ```
#
# We use **scratch memory** (`scratch_shapes`) for the accumulator.
# Scratch is allocated in **VMEM** — TPU's fast on-chip SRAM (like shared
# memory on GPU). Why not just accumulate directly in `o_ref`? Two reasons:
# 1. **Performance**: `o_ref` points to HBM. Reading and writing it on
#    every K iteration means round-trips to slow off-chip memory.
#    Scratch stays in fast VMEM throughout all K iterations.
# 2. **Correctness**: The output BlockSpec maps `(m, n, k) → (m, n)` —
#    multiple K iterations target the same output tile. Without a local
#    accumulator, K iteration 1 would overwrite K iteration 0's result.
#
# Specify scratch with `pltpu.VMEM(shape, dtype)`.
#
# **`out_shape`** tells `pallas_call` the shape and dtype of the output
# array to allocate. It's a `jax.ShapeDtypeStruct` — just metadata, no
# actual data:
# ```python
# out_shape = jax.ShapeDtypeStruct((M, N), jnp.float32)
# ```
# (In earlier puzzles, `out_shape` matched the input shape. Here the
# output shape `(M, N)` differs from either input, so it must be explicit.)
#
# Inside a kernel, use `a @ b` (or equivalently `jax.lax.dot(a, b)`) for
# the matrix multiply. Both map to the TPU's MXU (Matrix Multiplier Unit).
#
# The production-ready pattern uses `@pl.when` guards:
# ```python
# @pl.when(k_i == 0)           # ZERO on first K tile
# def _(): acc[...] = zeros
#
# acc[...] += a @ b             # ACCUMULATE on every tile
#
# @pl.when(k_i == tiles_k - 1) # STORE on last K tile
# def _(): out[...] = acc[...]
# ```
#
# On TPU hardware, `@pl.when` compiles to predicated execution — no branch
# divergence penalty. This zero/accumulate/store pattern is used in every
# production Pallas kernel.

# %%
M, K, N = 128, 256, 128
bm, bk, bn = 64, 128, 64
tiles_m = M // bm
tiles_n = N // bn
tiles_k = K // bk

# --- Reference ---
def matmul_spec(a, b):
    """a: (M, K), b: (K, N) → (M, N)"""
    return a @ b

# --- Kernel skeleton ---
def matmul_kernel(a_ref, b_ref, o_ref, acc_ref):
    # a_ref: (bm, bk) — tile of A
    # b_ref: (bk, bn) — tile of B
    # o_ref: (bm, bn) — output tile
    # acc_ref: (bm, bn) — scratch accumulator (VMEM on TPU)
    k_i = pl.program_id(2)
    # YOUR CODE HERE
    # 1. Zero acc_ref when k_i == 0
    # 2. Accumulate: acc_ref[...] += a_ref[...] @ b_ref[...]
    # 3. Store acc_ref → o_ref when k_i == tiles_k - 1


# %%
a = jax.random.normal(jax.random.key(7), (M, K))
b = jax.random.normal(jax.random.key(8), (K, N))

expected = matmul_spec(a, b)
actual = pl.pallas_call(
    matmul_kernel,
    grid=(tiles_m, tiles_n, tiles_k),
    in_specs=[
        pl.BlockSpec((bm, bk), lambda m, n, k: (m, k)),
        pl.BlockSpec((bk, bn), lambda m, n, k: (k, n)),
    ],
    out_specs=pl.BlockSpec((bm, bn), lambda m, n, k: (m, n)),
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
    interpret=True,
)(a, b)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected:\n{expected[:4]}")
    print(f"  Got:\n{actual[:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Pattern skeleton</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)
#
# acc_ref[...] += ...  # A_tile @ B_tile
#
# @pl.when(k_i == tiles_k - 1)
# def _store():
#     o_ref[...] = acc_ref[...]
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
# acc_ref[...] += a_ref[...] @ b_ref[...]
#
# @pl.when(k_i == tiles_k - 1)
# def _store():
#     o_ref[...] = acc_ref[...]
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 9: Configure Your Own Matmul `pallas_call`
#
# **Goal**: Given a working matmul kernel, fill in **all** the `pallas_call`
# arguments: `grid`, `in_specs`, `out_specs`, `out_shape`, and
# `scratch_shapes`.
#
# ### Theory
#
# This is the next step in learning to configure `pallas_call` yourself.
# Unlike Puzzle 6 (1D vector add), matmul has a **3D grid** and requires
# scratch memory. You need to understand how `BlockSpec` index maps route
# tiles in a 3D grid:
#
# - `A` tile `(m, k)` is at `A[m*bm:(m+1)*bm, k*bk:(k+1)*bk]`
#   → index map: `lambda m, n, k: (m, k)`
# - `B` tile `(k, n)` is at `B[k*bk:(k+1)*bk, n*bn:(n+1)*bn]`
#   → index map: `lambda m, n, k: (k, n)`
# - `C` tile `(m, n)` is at `C[m*bm:(m+1)*bm, n*bn:(n+1)*bn]`
#   → index map: `lambda m, n, k: (m, n)` (no K dependency!)
#
# Don't forget `out_shape` (the full output shape, not the tile shape)
# and `scratch_shapes` (the VMEM accumulator from Puzzle 8).

# %%
M, K, N = 128, 256, 128
bm, bk, bn = 64, 128, 64
tiles_m = M // bm
tiles_n = N // bn
tiles_k = K // bk

def matmul_spec9(a, b):
    return a @ b

# Kernel is provided (solved — same pattern as Puzzle 8):
def matmul_kernel_solved(a_ref, b_ref, o_ref, acc_ref):
    k_i = pl.program_id(2)
    @pl.when(k_i == 0)
    def _zero():
        acc_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)
    acc_ref[...] += a_ref[...] @ b_ref[...]
    @pl.when(k_i == tiles_k - 1)
    def _store():
        o_ref[...] = acc_ref[...]


# %%
a = jax.random.normal(jax.random.key(20), (M, K))
b = jax.random.normal(jax.random.key(21), (K, N))

expected = matmul_spec9(a, b)

# YOUR TASK: Replace ALL arguments with correct values.
actual = pl.pallas_call(
    matmul_kernel_solved,
    grid=(),                   # FIX THIS — 3D grid (tiles_m, tiles_n, tiles_k)
    in_specs=None,             # FIX THIS — BlockSpec for A and B
    out_specs=None,            # FIX THIS — BlockSpec for C
    out_shape=None,            # FIX THIS — output ShapeDtypeStruct
    scratch_shapes=(),         # FIX THIS — VMEM scratch for accumulator
    interpret=True,
)(a, b)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected:\n{expected[:4]}")
    print(f"  Got:\n{actual[:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — What to fill in</summary>
#
# You need five things:
# - `grid = (tiles_m, tiles_n, tiles_k)`
# - `in_specs` with two BlockSpecs: A maps `(m,n,k)→(m,k)`, B maps `(m,n,k)→(k,n)`
# - `out_specs` maps `(m,n,k)→(m,n)`
# - `out_shape = jax.ShapeDtypeStruct((M, N), jnp.float32)`
# - `scratch_shapes = [pltpu.VMEM((bm, bn), jnp.float32)]`
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# actual = pl.pallas_call(
#     matmul_kernel_solved,
#     grid=(tiles_m, tiles_n, tiles_k),
#     in_specs=[
#         pl.BlockSpec((bm, bk), lambda m, n, k: (m, k)),
#         pl.BlockSpec((bk, bn), lambda m, n, k: (k, n)),
#     ],
#     out_specs=pl.BlockSpec((bm, bn), lambda m, n, k: (m, n)),
#     out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
#     scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
#     interpret=True,
# )(a, b)
# ```
# </details>

# %% [markdown]
# ---
# ## Puzzle 10: Batched Matmul — Batch Dimension on RHS
#
# **Goal**: Compute `out[g] = lhs[g] @ rhs[g]` for `G` independent batches.
# The RHS has a leading batch dimension.
#
# ### Theory
#
# In batched matmul, the RHS is `(G, K, N)` — a stack of `G` weight
# matrices. Each batch element `g` has its own `(K, N)` matrix.
#
# (We use `G` for the batch size here because in the ragged_dot notebook,
# this same structure will represent **groups** in ragged_dot.)
#
# The grid adds a **batch dimension**: `grid = (G,)`.
# Each iteration's BlockSpec selects one batch element at a time.
#
# ```
#  lhs (G,M,K)          rhs (G,K,N)          out (G,M,N)
#  ┌───────────┐        ┌──────────┐         ┌──────────┐
#  │ g=0  M×K  │──┐     │ g=0 K×N  │──┐      │ g=0 M×N  │
#  ├───────────┤  │     ├──────────┤  │      ├──────────┤
#  │ g=1  M×K  │  │     │ g=1 K×N  │  │      │ g=1 M×N  │
#  ├───────────┤  │     ├──────────┤  │      ├──────────┤
#  │ g=2  M×K  │  │     │ g=2 K×N  │  │      │ g=2 M×N  │
#  ├───────────┤  │     ├──────────┤  │      ├──────────┤
#  │ g=3  M×K  │  │     │ g=3 K×N  │  │      │ g=3 M×N  │
#  └───────────┘  │     └──────────┘  │      └──────────┘
#                 │                   │
#  Grid iter g=0: └──→ lhs_ref(M,K) @ rhs_ref(K,N) ──→ o_ref(M,N)
#                      ▲ batch dim squeezed by None
# ```
#
# **`None` vs integer in block_shape**: Using `None` means "load the entire
# axis and **squeeze** that dimension". The ref will NOT have that dim.
# Using an integer (e.g. `1`) means "load 1 element" — the ref keeps that
# dim with size 1.
#
# For a batch dim, `None` is convenient — the kernel sees simple 2D
# shapes like `(M, K)` instead of `(1, M, K)`:
# ```
# BlockSpec((None, M, K), lambda g: (g, 0, 0))
#            ^^^^
#            squeezed — ref shape is (M, K), not (1, M, K)
# ```
#
# This is the precursor to ragged_dot, where different row-ranges of a
# single LHS matrix are multiplied by different group weight matrices.

# %%
G, M, K, N = 4, 64, 128, 64

# --- Reference ---
def batched_matmul_spec(lhs, rhs):
    """lhs: (G, M, K), rhs: (G, K, N) → (G, M, N)"""
    return jnp.einsum('gmk,gkn->gmn', lhs, rhs)

# --- Kernel skeleton ---
def batched_matmul_kernel(lhs_ref, rhs_ref, o_ref):
    # With None in block_shape, the batch dim is squeezed:
    # lhs_ref: (M, K) — one group's lhs (batch dim squeezed)
    # rhs_ref: (K, N) — one group's rhs (batch dim squeezed)
    # o_ref: (M, N) — one group's output (batch dim squeezed)
    # YOUR CODE HERE


# %%
lhs = jax.random.normal(jax.random.key(12), (G, M, K))
rhs = jax.random.normal(jax.random.key(13), (G, K, N))

expected = batched_matmul_spec(lhs, rhs)
actual = pl.pallas_call(
    batched_matmul_kernel,
    grid=(G,),
    in_specs=[
        pl.BlockSpec((None, M, K), lambda g: (g, 0, 0)),
        pl.BlockSpec((None, K, N), lambda g: (g, 0, 0)),
    ],
    out_specs=pl.BlockSpec((None, M, N), lambda g: (g, 0, 0)),
    # Full output is (G, M, N) even though each kernel invocation writes (M, N)
    out_shape=jax.ShapeDtypeStruct((G, M, N), jnp.float32),
    interpret=True,
)(lhs, rhs)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected:\n{expected[:4]}")
    print(f"  Got:\n{actual[:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# With `None` in BlockSpec, the batch dimension is **squeezed** — the refs have shape `(M, K)` and `(K, N)` directly (no leading dim). So the kernel just needs a single matmul.
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
# ## Puzzle 11: Fused Matmul + ReLU
#
# **Goal**: Compute `ReLU(A @ B)` in a single fused kernel — matmul and
# activation in one pass, no intermediate materialization.
#
# ### Theory
#
# On TPU, fusing operations into the kernel avoids an extra HBM round-trip.
# Without fusion: matmul writes `C` to HBM, then a separate kernel reads
# `C` back and applies ReLU. With fusion: ReLU is applied inside the kernel
# before the final store, saving one full read+write of the output matrix.
#
# The pattern is the same zero/accumulate/store from Puzzle 8, but the
# **store** step applies the activation before writing:
#
# ```python
# @pl.when(k_i == tiles_k - 1)
# def _store():
#     o_ref[...] = jnp.maximum(acc_ref[...], 0)  # fused ReLU!
# ```
#
# This fusion pattern generalizes to any elementwise activation (GELU,
# SiLU, etc.) and is used in production MoE kernels.

# %%
M, K, N = 128, 256, 128
bm, bk, bn = 64, 128, 64
tiles_m = M // bm
tiles_n = N // bn
tiles_k = K // bk

# --- Reference ---
def fused_relu_spec(a, b):
    """a: (M, K), b: (K, N) → ReLU(a @ b)"""
    return jnp.maximum(a @ b, 0)

# --- Kernel skeleton ---
def fused_relu_kernel(a_ref, b_ref, o_ref, acc_ref):
    k_i = pl.program_id(2)
    # YOUR CODE HERE
    # Same zero/accumulate/store as Puzzle 8, but apply ReLU before storing


# %%
a = jax.random.normal(jax.random.key(22), (M, K))
b = jax.random.normal(jax.random.key(23), (K, N))

expected = fused_relu_spec(a, b)
actual = pl.pallas_call(
    fused_relu_kernel,
    grid=(tiles_m, tiles_n, tiles_k),
    in_specs=[
        pl.BlockSpec((bm, bk), lambda m, n, k: (m, k)),
        pl.BlockSpec((bk, bn), lambda m, n, k: (k, n)),
    ],
    out_specs=pl.BlockSpec((bm, bn), lambda m, n, k: (m, n)),
    out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
    scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
    interpret=True,
)(a, b)

if jnp.allclose(actual, expected, atol=1e-3):
    print(f"PASSED ✓  (shape={actual.shape}, dtype={actual.dtype})")
else:
    diff = jnp.abs(actual - expected)
    print(f"FAILED ✗  max error: {float(jnp.max(diff)):.6f}")
    print(f"  Expected:\n{expected[:4]}")
    print(f"  Got:\n{actual[:4]}")

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# Copy the Puzzle 8 solution, but change the store step to apply `jnp.maximum(..., 0)` before writing to `o_ref`.
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm, bn), dtype=jnp.float32)
#
# acc_ref[...] += a_ref[...] @ b_ref[...]
#
# @pl.when(k_i == tiles_k - 1)
# def _store():
#     o_ref[...] = jnp.maximum(acc_ref[...], 0)
# ```
# </details>

