# CLAUDE.md

## Project overview

Progressive puzzle notebooks for learning Pallas (JAX's kernel language).
Each `.py` file is a jupytext percent-format notebook → `.ipynb` for Google Colab.
All puzzles run on CPU via `interpret=True` — no TPU needed.

## Files

| File | Focus |
|------|-------|
| `basics.py` | Pallas foundations: Refs, grids, BlockSpec, tiled matmul, fusion |
| `ragged_dot.py` | Scalar prefetch, group metadata, grouped matmul for MoE |
| `splash_attention.py` | Online softmax, flash attention, causal & block-sparse masks, splash attention |

Edit `.py` files (source of truth), then `bash update_notebooks.sh` to
regenerate `.ipynb`. Commit both. Use `uv` for package management.

## Creating new puzzles

### 1. Understand the user

Before writing anything, ask about their **background** (what they already
know) and **goals** (what they want to build/learn). This determines
starting difficulty and what the final puzzle builds toward.

### 2. Research

Read official JAX/Pallas docs, relevant papers, and production
implementations. Look at existing puzzle/tutorial formats for inspiration
(Sasha Rush's GPU/Tensor puzzles, Triton tutorials).

### 3. Design the puzzle sequence

- One new concept per puzzle, building on the previous ones.
- Tell a story — clear arc from first building block to the real thing.
- When a puzzle's solution is identical to a previous one (e.g., same
  kernel body, just different grid/BlockSpecs), celebrate that as the aha
  moment — don't obscure it by implying changes are needed.

### 4. Write puzzles that are fun

- Theory sections build intuition (*why*, not just *what*).
- Diagrams for anything spatial (see Diagrams section below).
- Progressive hints: gentle nudge → pattern skeleton → full solution.
- Inline comments on new APIs in the code skeleton.

### 5. Conventions

- Read existing notebooks before writing — match the established patterns.
- Plain variable names (`M`, `bm`, `expected`). No puzzle-number suffixes.
- Use TPU memory terminology: VMEM (vector memory), SMEM (scalar memory).
- Dimension names from [How to Scale Your Model](https://jax-ml.github.io/scaling-book/transformers/):
  `B` batch, `T` query/target seq len, `S` KV seq len, `D` model dim,
  `H` head dim, `N` query heads, `K` KV heads, `F` ff dim, `V` vocab,
  `L` layers, `E` experts.

## Diagrams

Diagrams are draw.io SVGs displayed in Colab via raw GitHub URLs.

**When drafting puzzles**, add a TODO comment describing what the diagram
should show and why it helps — don't create the diagram yet:
```python
# TODO diagram: 4×4 grid showing block categories (FULL/PART/SKIP),
# row=Q blocks, col=KV blocks. Helps visualize which blocks are skipped.
```

**When content is stable**, generate all diagrams in one batch using the
`/drawio` skill. This enables consistent styling across diagrams.

**Files**: `images/<name>.drawio` (editable source) + `images/<name>.drawio.svg`
(displayed in notebook). Reference in the `.py` notebook as:
```
# ![Alt text](https://raw.githubusercontent.com/vorushin/pallas_puzzles/master/images/<name>.drawio.svg)
```

**Style**: blue (#dae8fc) for inputs/data, yellow (#fff2cc) for
intermediates, green (#d5e8d4) for outputs, red (#f8cecc) for warnings,
gray (#f5f5f5) for skip/neutral. Grid diagrams use 55×35 cells with
fontSize 11-13. Keep diagrams compact — similar footprint to the content
they illustrate.

## Reviewing puzzles

Solve each puzzle as if you're the user. Is the goal clear? Does the
theory build intuition? Do the hints unstick without spoiling? Is it fun?
