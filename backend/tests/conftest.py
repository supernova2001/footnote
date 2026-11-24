"""Shared fixtures: an offline Services container over a tiny fixture corpus.

The fixture text is short passages written for these tests about public-domain era figures;
it is not quoted from any source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from footnote.corpus.licenses import make_license
from footnote.corpus.sources import sha256
from footnote.domain import Document, Scholar, Section
from footnote.services import Services, build_services
from footnote.settings import Settings

WOODSON_BIO = (
    "Carter G. Woodson was a historian who studied the history of African Americans. "
    "He founded the Association for the Study of Negro Life and History in 1915. "
    "In 1916 he started The Journal of Negro History, a scholarly journal.\n\n"
    "In February 1926 Woodson launched Negro History Week. "
    "The week was chosen because it included the birthdays of Frederick Douglass and Abraham Lincoln. "
    "Negro History Week later grew into Black History Month."
)
WOODSON_WORK = (
    "The education of the Negro before 1861 was shaped by missionaries and by the free Black community. "
    "Some schools in Northern cities taught reading, writing and arithmetic to free children.\n\n"
    "In several Southern states laws forbade teaching enslaved people to read. "
    "These laws grew harsher after uprisings, when officials feared the influence of educated men."
)
WELLS_WORK = (
    "Ida B. Wells published a pamphlet in 1892 that documented lynchings in the South. "
    "She gathered newspaper reports and testimony to show that the stated reasons were often false.\n\n"
    "Wells urged Black communities to use economic pressure, such as boycotts, and to leave towns that "
    "would not protect them. She argued that a free press was the means of self-help."
)


def _doc(doc_id: str, title: str, kind: str, scholar: str, text: str, year: int | None) -> Document:
    return Document(
        id=doc_id,
        title=title,
        kind=kind,
        year=year,
        scholar_ids=[scholar],  # type: ignore[arg-type]
        source_url=f"https://example.org/{doc_id}",
        license=make_license("PD-US", f"{title}: test fixture"),
        sha256=sha256(text),
        text=text,
        sections=[Section(title="Body", start=0, end=len(text))],
    )


FIXTURE_DOCS = [
    _doc(
        "wikipedia-carter-g-woodson",
        "Carter G. Woodson (Wikipedia)",
        "biography",
        "carter-g-woodson",
        WOODSON_BIO,
        None,
    ),
    _doc("gutenberg-11089", "The Education of the Negro Prior to 1861", "work", "carter-g-woodson", WOODSON_WORK, 1915),
    _doc("gutenberg-14975", "Southern Horrors", "work", "ida-b-wells", WELLS_WORK, 1892),
]
FIXTURE_SCHOLARS = [
    Scholar(
        id="carter-g-woodson",
        name="Carter G. Woodson",
        born=1875,
        died=1950,
        kind="historical",
        summary="Historian.",
        document_count=2,
    ),
    Scholar(
        id="ida-b-wells",
        name="Ida B. Wells",
        born=1862,
        died=1931,
        kind="historical",
        summary="Journalist.",
        document_count=1,
    ),
]


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    data = {
        "scholars": [s.model_dump(mode="json") for s in FIXTURE_SCHOLARS],
        "documents": [d.model_dump(mode="json") for d in FIXTURE_DOCS],
    }
    (tmp_path / "snapshot.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


@pytest.fixture
def settings(snapshot_dir: Path) -> Settings:
    # _env_file=None: tests never read the developer's .env (keys, judge backend, provider).
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        FOOTNOTE_OFFLINE=True,
        FOOTNOTE_DATA_DIR=snapshot_dir,
        RATE_LIMIT_PER_MINUTE=1000,
        FOOTNOTE_MCP_SERVERS=snapshot_dir / "none.json",
    )


@pytest.fixture
async def services(settings: Settings) -> Services:
    svc = await build_services(settings)
    yield svc  # type: ignore[misc]
    await svc.aclose()
