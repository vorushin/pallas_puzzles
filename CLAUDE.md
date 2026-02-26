# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

pallas_puzzles is a standalone collection of progressive Pallas kernel puzzles.
Each `.py` file is a jupytext percent-format notebook that generates a `.ipynb`
for use in Google Colab. All puzzles run on CPU via `interpret=True` — no TPU needed.

## Local development

Use `uv` for Python package management. Run scripts with `uv run`.

## File map

| File | Puzzles | Focus |
|------|---------|-------|
| `basics.py` | 1-11 | Pallas foundations: Refs, grids, BlockSpec, tiled matmul, fusion |
| `ragged_dot.py` | 1-9 | Scalar prefetch, group metadata, masked stores, grouped matmul |

Future: `splash_attention.py` — requires `basics.py` as prerequisite.

## Editing workflow

1. Edit the `.py` file (source of truth) — jupytext percent format
2. Run `bash update_notebooks.sh` to regenerate `.ipynb` via jupytext
3. Commit both `.py` and `.ipynb`

## How to create and update puzzles

Each puzzle file uses jupytext percent format:
- `# %%` marks a code cell
- `# %% [markdown]` marks a markdown cell
- All markdown lines are prefixed with `#` (they're Python comments)

### Puzzle structure

Every puzzle should have these components in order:

1. **Heading**: `## Puzzle N: Title` — clear, descriptive title
2. **Goal**: One sentence stating exactly what to implement
3. **Theory section** (`### Theory`): Explain the concept being taught
   - Build intuition, don't just state facts
   - Use ASCII diagrams for spatial/tiling concepts
   - Show the "before and after" — what changes from the previous puzzle
   - Introduce exactly one new concept per puzzle
4. **Parameters**: Size constants with puzzle-number suffix (e.g., `M8`, `K8`, `bm8`)
5. **Reference spec**: Pure JAX function that computes the expected result
6. **Kernel skeleton**: Function with `pass  # YOUR CODE HERE` placeholder
   - Add inline comments explaining new concepts
   - Include type/shape comments for Ref parameters
7. **Test cell**: Uses `check()` or inline `pallas_call` + comparison
8. **Progressive hints**: 2-3 `<details>` blocks, ordered:
   - Hint 1: High-level approach (what to do, not how)
   - Hint 2: Pattern skeleton (code structure with blanks)
   - Hint 3: Full solution (complete working code)

### Variable naming

Use puzzle number as suffix for all size/data variables to avoid collisions:
`M8, K8, N8, bm8, bk8, bn8, x8, y8` etc.

### Cross-references

When referencing puzzles in another file, prefix with the filename:
"basics.py Puzzle 8" not just "Puzzle 8".

## How to create .ipynb notebooks before commit and push

```bash
bash update_notebooks.sh
```

This runs `uv run jupytext --to ipynb --update` on each puzzle file.
Always run this before committing — both `.py` and `.ipynb` must be in sync.

## How to review puzzles

Run each puzzle file as a notebook (or read through it sequentially),
imagining yourself as a user trying to solve the puzzles. For each puzzle,
check:

1. **Description detail**: Is the Goal clear and specific? Can a reader
   understand exactly what to implement without reading the hints?

2. **Intuition explanation**: Does the Theory section build genuine
   understanding? Does it explain *why* this concept matters, not just
   *what* it is? Are ASCII diagrams used where they help visualize
   tiling, grids, or memory layout?

3. **Inline comments on new code**: When the kernel skeleton or test cell
   introduces a new API (`BlockSpec`, `@pl.when`, `PrefetchScalarGridSpec`,
   etc.), is it annotated with a brief comment explaining what it does?

4. **Progressive hints**: Are hints ordered from general to specific?
   Does Hint 1 help a stuck user think about the approach without giving
   away the answer? Does each subsequent hint add useful information?
   Is the final hint a complete, working solution?

## Dependencies

Colab cells install deps with `!pip install -q jax jaxtyping`.
For local development: `uv sync`.
