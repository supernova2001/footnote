"""Load the backend's judge prompt and metrics by path.

The training prompt must be the one the backend sends Claude (`backend/src/footnote/judge_prompt.py`),
so it is imported, never copied. The two files are standard-library only; they are loaded as a
stand-in package (`footnote_shared`) so their relative imports work without importing the backend
package itself (and its FastAPI/Pinecone dependencies).

On Kaggle or Colab, clone the repo and the default path works; otherwise point FOOTNOTE_SHARED_DIR at
the folder that holds judge_prompt.py.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import ModuleType

_PACKAGE = "footnote_shared"


def shared_dir() -> Path:
    override = os.environ.get("FOOTNOTE_SHARED_DIR")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2] / "backend" / "src" / "footnote"


def _package() -> ModuleType:
    existing = sys.modules.get(_PACKAGE)
    if existing is not None:
        return existing
    folder = shared_dir()
    if not (folder / "judge_prompt.py").is_file():
        raise FileNotFoundError(
            f"judge_prompt.py not found in {folder}. Run from a clone of the repo, or set FOOTNOTE_SHARED_DIR."
        )
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(folder)]
    package.__package__ = _PACKAGE
    sys.modules[_PACKAGE] = package
    return package


def judge_prompt() -> ModuleType:
    _package()
    return importlib.import_module(f"{_PACKAGE}.judge_prompt")


def judge_metrics() -> ModuleType:
    _package()
    return importlib.import_module(f"{_PACKAGE}.judge_metrics")
