from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_train.ipynb"


def test_notebook_is_valid_nbformat():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    sources = "\n".join(c.source for c in nb.cells)
    for step in ("build_dataset.py", "train_qlora.py", "eval_judge.py", "merge_adapter.py", "to_gguf.sh"):
        assert step in sources
