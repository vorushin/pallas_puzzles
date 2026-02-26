#!/bin/bash
# Regenerate .ipynb files from .py sources using jupytext
set -e
cd "$(dirname "$0")"

for py_file in basics.py ragged_dot.py; do
    ipynb_file="${py_file%.py}.ipynb"
    echo "Converting $py_file -> $ipynb_file"
    uv run jupytext --to ipynb --update "$py_file"
done

echo "Done."
