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
# run the check cells.
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

# %%
# !pip install -q jax jaxtyping

# %%
import functools
import jax
import jax.numpy as jnp
from jax import lax
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
# ┌────────────────────────┐
# │  x_ref  →  [ read ]   │
# │                ↓       │
# │           x + 10.0     │
# │                ↓       │
# │  o_ref  ←  [ write ]  │
# └────────────────────────┘
# ```

# %%
N1 = 32

# --- Reference (spec) ---
def add10_spec(x):
    """x: (N1,) → x + 10"""
    return x + 10.0

# --- Kernel skeleton ---
def add10_kernel(x_ref, o_ref):
    # x_ref: Ref to input block (shape (N1,))
    # o_ref: Ref to output block (shape (N1,))
    pass  # YOUR CODE HERE


# %%
x1 = jax.random.uniform(jax.random.key(0), (N1,))
check(add10_kernel, add10_spec, (x1,))

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
N2 = 256   # vector length
bm2 = 64   # tile (block) size — each kernel invocation processes bm2 elements

# --- Reference ---
def vadd_spec(x, y):
    """x, y: (N2,) → x + y"""
    return x + y

# --- Kernel skeleton ---
def vadd_kernel(x_ref, y_ref, o_ref):
    # Each invocation sees a (bm2,) slice thanks to BlockSpec
    pass  # YOUR CODE HERE


# %%
x2 = jax.random.uniform(jax.random.key(1), (N2,))
y2 = jax.random.uniform(jax.random.key(2), (N2,))

check(vadd_kernel, vadd_spec, (x2, y2),
      grid=(N2 // bm2,),              # 256 // 64 = 4 invocations
      in_specs=[
          pl.BlockSpec((bm2,), lambda i: (i,)),  # x: invocation i → block i
          pl.BlockSpec((bm2,), lambda i: (i,)),  # y: invocation i → block i
      ],
      out_specs=pl.BlockSpec((bm2,), lambda i: (i,)))  # out: invocation i → block i

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
N3 = 256   # vector length (same as Puzzle 2)
bm3 = 64   # tile size
num_blocks_3 = N3 // bm3   # 4 blocks total

# --- Reference ---
def vadd_rev_spec(x, y):
    """x, y: (N3,) → x + block_reverse(y)"""
    y_rev = y.reshape(num_blocks_3, bm3)[::-1].reshape(N3)
    return x + y_rev

# Kernel is provided (same body as Puzzle 2):
def vadd_rev_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


# %%
x3 = jax.random.uniform(jax.random.key(100), (N3,))
y3 = jax.random.uniform(jax.random.key(101), (N3,))

# YOUR TASK: Fix the y BlockSpec so it reads blocks in reversed order.
# Only the y index map needs to change — x and out are correct.
check(vadd_rev_kernel, vadd_rev_spec, (x3, y3),
      grid=(num_blocks_3,),
      in_specs=[
          pl.BlockSpec((bm3,), lambda i: (i,)),              # x: block i (correct)
          pl.BlockSpec((bm3,), lambda i: (i,)),              # y: block i — FIX THIS
      ],
      out_specs=pl.BlockSpec((bm3,), lambda i: (i,)))

# %% [markdown]
# <details><summary>Hint</summary>
#
# The y index map should map grid index `i` to the reversed block position.
# With 4 blocks, `i=0 → block 3`, `i=1 → block 2`, etc.:
# ```python
# pl.BlockSpec((bm3,), lambda i: (num_blocks_3 - 1 - i,))
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
M4, N4 = 128, 128
bm4, bn4 = 32, 32

# --- Reference ---
def mul2d_spec(x):
    """x: (M4, N4) → x * 2"""
    return x * 2.0

# --- Kernel skeleton ---
def mul2d_kernel(x_ref, o_ref):
    pass  # YOUR CODE HERE


# %%
x4 = jax.random.uniform(jax.random.key(3), (M4, N4))
check(mul2d_kernel, mul2d_spec, (x4,),
      grid=(M4 // bm4, N4 // bn4),
      in_specs=[pl.BlockSpec((bm4, bn4), lambda i, j: (i, j))],
      out_specs=pl.BlockSpec((bm4, bn4), lambda i, j: (i, j)))

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
#              b (N=64)
#            [b₀ ][b₁ ]
#             j=0  j=1
#           ┌─────┬─────┐
#   a  a₀   │ a₀  │ a₀  │
#  (M  i=0  │ ×b₀ │ ×b₁ │
#  =        ├─────┼─────┤
#  128 a₁   │ a₁  │ a₁  │
#  )   i=1  │ ×b₀ │ ×b₁ │
#           ├─────┼─────┤
#       a₂  │ a₂  │ a₂  │
#       i=2 │ ×b₀ │ ×b₁ │
#           ├─────┼─────┤
#       a₃  │ a₃  │ a₃  │
#       i=3 │ ×b₀ │ ×b₁ │
#           └─────┴─────┘
#          output (128×64)
#
# Each tile (i,j): a_ref=(bm,) and b_ref=(bn,)
#   → broadcast to (bm, bn) via [:, None] * [None, :]
# ```
#
# Inside the kernel, `a_ref` has shape `(bm,)` and `b_ref` has shape `(bn,)`.
# You need to broadcast them: `a_ref[...][:, None] * b_ref[...][None, :]`
# produces shape `(bm, bn)`.

# %%
M5, N5 = 128, 64
bm5, bn5 = 32, 32

# --- Reference ---
def outer_spec(a, b):
    """a: (M5,), b: (N5,) → (M5, N5)"""
    return a[:, None] * b[None, :]

# --- Kernel skeleton ---
def outer_kernel(a_ref, b_ref, o_ref):
    # a_ref: (bm5,) — a slice of vector a
    # b_ref: (bn5,) — a slice of vector b
    # o_ref: (bm5, bn5) — output tile
    pass  # YOUR CODE HERE


# %%
a5 = jax.random.uniform(jax.random.key(4), (M5,))
b5 = jax.random.uniform(jax.random.key(5), (N5,))

check(outer_kernel, outer_spec, (a5, b5),
      grid=(M5 // bm5, N5 // bn5),
      in_specs=[
          pl.BlockSpec((bm5,), lambda i, j: (i,)),
          pl.BlockSpec((bn5,), lambda i, j: (j,)),
      ],
      out_specs=pl.BlockSpec((bm5, bn5), lambda i, j: (i, j)),
      out_shape=jax.ShapeDtypeStruct((M5, N5), jnp.float32))

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# You need to broadcast `a_ref[...]` (shape `(bm5,)`) and `b_ref[...]` (shape `(bn5,)`) to produce shape `(bm5, bn5)`. Use NumPy-style broadcasting: add a new axis with `[:, None]` and `[None, :]`.
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
# wire up the tiling so it processes `N6`-element vectors in blocks of
# `bm6`.

# %%
N6 = 256
bm6 = 64

def vadd_spec6(x, y):
    return x + y

# Kernel is provided (solved):
def vadd_kernel_solved(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


# %%
x6 = jax.random.uniform(jax.random.key(10), (N6,))
y6 = jax.random.uniform(jax.random.key(11), (N6,))

# YOUR TASK: Define grid, in_specs, out_specs to tile the computation
# into bm6-sized blocks. The kernel processes one block per invocation.
vadd_grid = ...       # TODO: how many tiles? (should be a tuple)
vadd_in_specs = ...   # TODO: list of BlockSpec, one per input
vadd_out_specs = ...  # TODO: BlockSpec for output

check(vadd_kernel_solved, vadd_spec6, (x6, y6),
      grid=vadd_grid,
      in_specs=vadd_in_specs,
      out_specs=vadd_out_specs)

# %% [markdown]
# <details><summary>Hint 1 of 2 — What to fill in</summary>
#
# ```python
# vadd_grid = (N6 // bm6,)  # 256 // 64 = 4 tiles
# vadd_in_specs = [
#     pl.BlockSpec((bm6,), lambda i: (i,)),  # one per input
#     pl.BlockSpec((bm6,), lambda i: (i,)),
# ]
# vadd_out_specs = pl.BlockSpec((bm6,), lambda i: (i,))
# ```
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# vadd_grid = (N6 // bm6,)
# vadd_in_specs = [
#     pl.BlockSpec((bm6,), lambda i: (i,)),
#     pl.BlockSpec((bm6,), lambda i: (i,)),
# ]
# vadd_out_specs = pl.BlockSpec((bm6,), lambda i: (i,))
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
ROWS7, COLS7 = 16, 256
bm7, bk7 = 16, 64
tiles_k7 = COLS7 // bk7

# --- Reference ---
def rowsum_spec(x):
    """x: (ROWS7, COLS7) → (ROWS7,)"""
    return x.sum(axis=1)

# --- Kernel skeleton ---
def rowsum_kernel(x_ref, o_ref):
    # x_ref: (bm7, bk7) — one tile of x
    # o_ref: (bm7,) — accumulator for this row block
    # Grid: (ROWS7 // bm7, COLS7 // bk7) — iterates (row_block, k_block)
    k_i = pl.program_id(1)
    pass  # YOUR CODE HERE
    # 1. On first k tile (k_i == 0), initialize the output
    # 2. Add this tile's contribution to the running sum


# %%
x7 = jax.random.uniform(jax.random.key(6), (ROWS7, COLS7))
check(rowsum_kernel, rowsum_spec, (x7,),
      grid=(ROWS7 // bm7, tiles_k7),
      in_specs=[pl.BlockSpec((bm7, bk7), lambda i, k: (i, k))],
      out_specs=pl.BlockSpec((bm7,), lambda i, k: (i,)),
      out_shape=jax.ShapeDtypeStruct((ROWS7,), jnp.float32))

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
#     o_ref[...] = jnp.zeros((bm7,), dtype=jnp.float32)
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
#     o_ref[...] = jnp.zeros((bm7,), dtype=jnp.float32)
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
#    A (128×256)         B (256×128)        C (128×128)
#    K=256 → 2 tiles     N=128 → 2 tiles
#    ┌───────┬───────┐   ┌───┬───┐         ┌───┬───┐
#    │ A0,0  │ A0,1  │   │B0,0│B0,1│        │C0,0│C0,1│
#    │ 64×128│ 64×128│   │128│128 │        │64 │64  │
#    ├───────┼───────┤   │×64│×64 │        │×64│×64 │
#    │ A1,0  │ A1,1  │   ├───┼───┤         ├───┼───┤
#    │ 64×128│ 64×128│   │B1,0│B1,1│        │C1,0│C1,1│
#    └───────┴───────┘   │128│128 │        │64 │64  │
#    M=128 → 2 tiles     │×64│×64 │        │×64│×64 │
#                        └───┴───┘         └───┴───┘
#
# To compute C[0,0], sweep k=0..1:
#
#   k=0: acc  = A0,0 @ B0,0    (64×128) @ (128×64) → (64×64)
#   k=1: acc += A0,1 @ B1,0    (64×128) @ (128×64) → (64×64)
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
# (In earlier puzzles, the `check` helper inferred this automatically
# from the reference output. From here on, you'll see it explicitly.)
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
M8, K8, N8 = 128, 256, 128
bm8, bk8, bn8 = 64, 128, 64
tiles_m8 = M8 // bm8
tiles_n8 = N8 // bn8
tiles_k8 = K8 // bk8

# --- Reference ---
def matmul_spec(a, b):
    """a: (M8, K8), b: (K8, N8) → (M8, N8)"""
    return a @ b

# --- Kernel skeleton ---
def matmul_kernel(a_ref, b_ref, o_ref, acc_ref):
    # a_ref: (bm8, bk8) — tile of A
    # b_ref: (bk8, bn8) — tile of B
    # o_ref: (bm8, bn8) — output tile
    # acc_ref: (bm8, bn8) — scratch accumulator (VMEM on TPU)
    k_i = pl.program_id(2)
    pass  # YOUR CODE HERE
    # 1. Zero acc_ref when k_i == 0
    # 2. Accumulate: acc_ref[...] += a_ref[...] @ b_ref[...]
    # 3. Store acc_ref → o_ref when k_i == tiles_k8 - 1


# %%
a8 = jax.random.normal(jax.random.key(7), (M8, K8))
b8 = jax.random.normal(jax.random.key(8), (K8, N8))

check(matmul_kernel, matmul_spec, (a8, b8),
      grid=(tiles_m8, tiles_n8, tiles_k8),
      in_specs=[
          pl.BlockSpec((bm8, bk8), lambda m, n, k: (m, k)),
          pl.BlockSpec((bk8, bn8), lambda m, n, k: (k, n)),
      ],
      out_specs=pl.BlockSpec((bm8, bn8), lambda m, n, k: (m, n)),
      out_shape=jax.ShapeDtypeStruct((M8, N8), jnp.float32),
      scratch_shapes=[pltpu.VMEM((bm8, bn8), jnp.float32)])

# %% [markdown]
# <details><summary>Hint 1 of 2 — Pattern skeleton</summary>
#
# ```python
# @pl.when(k_i == 0)
# def _zero():
#     acc_ref[...] = jnp.zeros((bm8, bn8), dtype=jnp.float32)
#
# acc_ref[...] += ...  # A_tile @ B_tile
#
# @pl.when(k_i == tiles_k8 - 1)
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
#     acc_ref[...] = jnp.zeros((bm8, bn8), dtype=jnp.float32)
#
# acc_ref[...] += a_ref[...] @ b_ref[...]
#
# @pl.when(k_i == tiles_k8 - 1)
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
M9, K9, N9 = 128, 256, 128
bm9, bk9, bn9 = 64, 128, 64
tiles_m9 = M9 // bm9
tiles_n9 = N9 // bn9
tiles_k9 = K9 // bk9

def matmul_spec9(a, b):
    return a @ b

# Kernel is provided (solved — same pattern as Puzzle 8):
def matmul_kernel_solved(a_ref, b_ref, o_ref, acc_ref):
    k_i = pl.program_id(2)
    @pl.when(k_i == 0)
    def _zero():
        acc_ref[...] = jnp.zeros((bm9, bn9), dtype=jnp.float32)
    acc_ref[...] += a_ref[...] @ b_ref[...]
    @pl.when(k_i == tiles_k9 - 1)
    def _store():
        o_ref[...] = acc_ref[...]


# %%
a9 = jax.random.normal(jax.random.key(20), (M9, K9))
b9 = jax.random.normal(jax.random.key(21), (K9, N9))

# YOUR TASK: Replace ALL arguments with correct values.
check(matmul_kernel_solved, matmul_spec9, (a9, b9),
      grid=(),                   # FIX THIS — 3D grid (tiles_m, tiles_n, tiles_k)
      in_specs=None,             # FIX THIS — BlockSpec for A and B
      out_specs=None,            # FIX THIS — BlockSpec for C
      out_shape=None,            # FIX THIS — output ShapeDtypeStruct
      scratch_shapes=())         # FIX THIS — VMEM scratch for accumulator

# %% [markdown]
# <details><summary>Hint 1 of 2 — What to fill in</summary>
#
# You need five things:
# - `grid = (tiles_m9, tiles_n9, tiles_k9)`
# - `in_specs` with two BlockSpecs: A maps `(m,n,k)→(m,k)`, B maps `(m,n,k)→(k,n)`
# - `out_specs` maps `(m,n,k)→(m,n)`
# - `out_shape = jax.ShapeDtypeStruct((M9, N9), jnp.float32)`
# - `scratch_shapes = [pltpu.VMEM((bm9, bn9), jnp.float32)]`
# </details>
#
# <details><summary>Hint 2 of 2 — Full solution</summary>
#
# ```python
# check(matmul_kernel_solved, matmul_spec9, (a9, b9),
#       grid=(tiles_m9, tiles_n9, tiles_k9),
#       in_specs=[
#           pl.BlockSpec((bm9, bk9), lambda m, n, k: (m, k)),
#           pl.BlockSpec((bk9, bn9), lambda m, n, k: (k, n)),
#       ],
#       out_specs=pl.BlockSpec((bm9, bn9), lambda m, n, k: (m, n)),
#       out_shape=jax.ShapeDtypeStruct((M9, N9), jnp.float32),
#       scratch_shapes=[pltpu.VMEM((bm9, bn9), jnp.float32)])
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
G10, M10, K10, N10 = 4, 64, 128, 64

# --- Reference ---
def batched_matmul_spec(lhs, rhs):
    """lhs: (G, M, K), rhs: (G, K, N) → (G, M, N)"""
    return jnp.einsum('gmk,gkn->gmn', lhs, rhs)

# --- Kernel skeleton ---
def batched_matmul_kernel(lhs_ref, rhs_ref, o_ref):
    # With None in block_shape, the batch dim is squeezed:
    # lhs_ref: (M10, K10) — one group's lhs (batch dim squeezed)
    # rhs_ref: (K10, N10) — one group's rhs (batch dim squeezed)
    # o_ref: (M10, N10) — one group's output (batch dim squeezed)
    pass  # YOUR CODE HERE


# %%
lhs10 = jax.random.normal(jax.random.key(12), (G10, M10, K10))
rhs10 = jax.random.normal(jax.random.key(13), (G10, K10, N10))

check(batched_matmul_kernel, batched_matmul_spec, (lhs10, rhs10),
      grid=(G10,),
      in_specs=[
          pl.BlockSpec((None, M10, K10), lambda g: (g, 0, 0)),
          pl.BlockSpec((None, K10, N10), lambda g: (g, 0, 0)),
      ],
      out_specs=pl.BlockSpec((None, M10, N10), lambda g: (g, 0, 0)),
      # Full output is (G, M, N) even though each kernel invocation writes (M, N)
      out_shape=jax.ShapeDtypeStruct((G10, M10, N10), jnp.float32))

# %% [markdown]
# <details><summary>Hint 1 of 2 — Approach</summary>
#
# With `None` in BlockSpec, the batch dimension is **squeezed** — the refs have shape `(M10, K10)` and `(K10, N10)` directly (no leading dim). So the kernel just needs a single matmul.
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
M11, K11, N11 = 128, 256, 128
bm11, bk11, bn11 = 64, 128, 64
tiles_m11 = M11 // bm11
tiles_n11 = N11 // bn11
tiles_k11 = K11 // bk11

# --- Reference ---
def fused_relu_spec(a, b):
    """a: (M11, K11), b: (K11, N11) → ReLU(a @ b)"""
    return jnp.maximum(a @ b, 0)

# --- Kernel skeleton ---
def fused_relu_kernel(a_ref, b_ref, o_ref, acc_ref):
    k_i = pl.program_id(2)
    pass  # YOUR CODE HERE
    # Same zero/accumulate/store as Puzzle 8, but apply ReLU before storing


# %%
a11 = jax.random.normal(jax.random.key(22), (M11, K11))
b11 = jax.random.normal(jax.random.key(23), (K11, N11))

check(fused_relu_kernel, fused_relu_spec, (a11, b11),
      grid=(tiles_m11, tiles_n11, tiles_k11),
      in_specs=[
          pl.BlockSpec((bm11, bk11), lambda m, n, k: (m, k)),
          pl.BlockSpec((bk11, bn11), lambda m, n, k: (k, n)),
      ],
      out_specs=pl.BlockSpec((bm11, bn11), lambda m, n, k: (m, n)),
      out_shape=jax.ShapeDtypeStruct((M11, N11), jnp.float32),
      scratch_shapes=[pltpu.VMEM((bm11, bn11), jnp.float32)])

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
#     acc_ref[...] = jnp.zeros((bm11, bn11), dtype=jnp.float32)
#
# acc_ref[...] += a_ref[...] @ b_ref[...]
#
# @pl.when(k_i == tiles_k11 - 1)
# def _store():
#     o_ref[...] = jnp.maximum(acc_ref[...], 0)
# ```
# </details>

# %% [markdown]
# ---
# ## Summary
#
# | Concept | Puzzle |
# |---------|--------|
# | `pallas_call`, Refs, `ref[...]` syntax | 1 |
# | `grid`, `BlockSpec`, `program_id` | 2 |
# | Index map manipulation | 3 |
# | 2D grids and BlockSpecs | 4 |
# | Broadcasting inside kernels | 5 |
# | Configure your own `pallas_call` | 6 |
# | `@pl.when` conditional execution, reduction | 7 |
# | Matmul with scratch accumulator (VMEM) | 8 |
# | Configure your own matmul `pallas_call` | 9 |
# | Batched matmul, `None` dim squeeze | 10 |
# | Activation fusion | 11 |
#
# ### Next
#
# Continue with **ragged_dot.py** — scalar prefetch, group metadata,
# and the full ragged_dot kernel for Mixture-of-Experts.
