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

Use `uv` for package management. Edit `.py` files (source of truth), then
`bash update_notebooks.sh` to regenerate `.ipynb`. Commit both.

## Creating new puzzles

### 1. Understand the user

Before writing anything, ask:

- **What's your background?** How much do you know about the topic area
  (e.g., JAX, linear algebra, TPU architecture, attention mechanisms)?
  This determines the starting difficulty and how much theory to include.
- **What are your goals?** What do you want to learn or be able to build
  after completing these puzzles? (e.g., "understand how splash attention
  works", "write my own Pallas kernels for production"). This shapes which
  concepts to cover and what the final puzzle should build toward.

### 2. Research

- Search for the best documentation and source code on the topic. Read the
  official JAX/Pallas docs, relevant papers, and production implementations.
- If good references are scarce, ask the user to point you to an example
  implementation or paper they want to learn from.
- Look for existing puzzles, etudes, or tutorials on similar topics for
  inspiration (e.g., Sasha Rush's GPU/Tensor puzzles, Triton tutorials,
  educational notebooks). See what worked well in those and adapt the ideas.

### 3. Design the puzzle sequence

Tailor puzzles to the specific user based on their background and goals:

- Start where the user is, not where you think beginners should be.
  Skip basics they already know. Spend more time on concepts they find hard.
- Each puzzle should introduce exactly one new concept, building on previous ones.
- The sequence should tell a story — a clear arc from "here's the first
  building block" to "now you've built the real thing."
- The final puzzle (or final few) should connect to the user's stated goal.

### 4. Write puzzles that are fun

Puzzles can be hard, but they should never be frustrating or dry.

- Write Theory sections that build intuition, not just state facts.
  Explain *why* something works this way, not just *what* it is.
- Use ASCII diagrams for anything spatial (memory layout, tiling, grids).
- A bit of humor and empathy goes a long way — acknowledge when something
  is tricky, celebrate when the user is about to have an aha moment.
- Progressive hints are a safety net: the user should never be truly stuck.
  Order them from gentle nudge → pattern skeleton → full solution.
- Inline comments on new APIs and concepts in the code skeleton — don't
  make users guess what `pltpu.VMEM` or `pl.program_id` means.

### 5. Technical format

- Jupytext percent format: `# %%` for code, `# %% [markdown]` for markdown
- Each puzzle: heading, goal, theory, reference spec, kernel skeleton
  with `pass  # YOUR CODE HERE`, test cell, hints in `<details>` blocks
- Use puzzle-number suffixes on variables to avoid collisions (e.g., `M3`, `bm3`)
- Cross-file references: "basics.py Puzzle 8" not just "Puzzle 8"

## Reviewing puzzles

Go through each puzzle as if you're the user solving it. For each one:

1. **Can I understand what to do?** Is the goal clear without reading hints?
2. **Do I understand why?** Does the theory build intuition, not just state facts?
3. **Am I stuck?** If so, does Hint 1 unstick me without giving away the answer?
4. **Is the code commented?** When new APIs appear, are they explained inline?
5. **Is it fun?** Or does it feel like homework?
