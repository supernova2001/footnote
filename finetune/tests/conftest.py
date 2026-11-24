from __future__ import annotations

import pytest

from footnote_ft.dataset import BuildConfig, assign_splits, balance, build_examples, to_row
from footnote_ft.demo_corpus import chunk_rows


@pytest.fixture(scope="session")
def chunks() -> list[dict]:
    return chunk_rows()


@pytest.fixture(scope="session")
def doc_split(chunks) -> dict[str, str]:
    return assign_splits((c["document_id"] for c in chunks), (0.8, 0.1, 0.1), seed=13)


@pytest.fixture(scope="session")
def rows(chunks, doc_split) -> list[dict]:
    examples, _ = build_examples(chunks, doc_split, BuildConfig(seed=13))
    return [to_row(ex) for ex in balance(examples, seed=13)]
