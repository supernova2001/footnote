from __future__ import annotations

import pytest

from footnote.corpus.chunking import MAX_CHARS, chunk_document, split_sentences
from footnote.corpus.licenses import make_license
from footnote.corpus.sources import (
    LicenceError,
    _heading_text,
    _xml_licence_ok,
    fetch_gutenberg,
    fetch_wikipedia,
    sha256,
)
from footnote.domain import Document, Section

PG_TEXT = """The Project Gutenberg eBook of A Test Book

This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever.

*** START OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***

Produced by a volunteer.

[Transcriber's Note: This pamphlet was first
published in 1900.]

CONTENTS

  I. THE FIRST CHAPTER   1

+CHAPTER I+

THE FIRST CHAPTER

This is the first paragraph of the book, hard wrapped
across two lines, with an _italic_ word.

A second paragraph follows here and it is long enough to be a passage of its own.

INDEX

Alpha, 1; Beta, 2; Gamma, 3; Delta, 4; Epsilon, 5; Zeta, 6.

End of Project Gutenberg's A Test Book, by Someone

*** END OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***
"""


class FakeFetcher:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    async def get_text(self, url: str, params: object = None) -> str:
        return str(self.payload)

    async def get_json(self, url: str, params: object = None) -> object:
        return self.payload


def _doc(text: str, sections: list[Section] | None = None) -> Document:
    return Document(
        id="doc-1",
        title="T",
        kind="work",
        year=1900,
        scholar_ids=["s"],
        source_url="https://example.org",
        license=make_license("PD-US", "test"),
        sha256=sha256(text),
        text=text,
        sections=sections or [],
    )


def test_split_sentences_keeps_initials_and_titles_whole() -> None:
    text = "W. E. B. Du Bois wrote the book in 1903. Dr. Woodson founded a journal. It mattered!"
    assert split_sentences(text) == [
        "W. E. B. Du Bois wrote the book in 1903.",
        "Dr. Woodson founded a journal.",
        "It mattered!",
    ]


def test_chunks_reproduce_their_source_span_and_respect_sections() -> None:
    para = "Sentence number {} is here to make the paragraph long enough to matter. "
    s1 = "\n\n".join("".join(para.format(i * 10 + j) for j in range(6)) for i in range(5))
    s2 = "A short closing section with enough words to keep it as a chunk of its own."
    text = s1 + "\n\n" + s2 + "\n\n"
    sections = [Section(title="One", start=0, end=len(s1) + 2), Section(title="Two", start=len(s1) + 2, end=len(text))]
    chunks = chunk_document(_doc(text, sections))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert " ".join(text[c.char_start : c.char_end].split()) == c.text
        assert len(c.text) <= MAX_CHARS + 200
        section = next(s for s in sections if s.start <= c.char_start < s.end)
        assert c.char_end <= section.end and c.section == section.title
    assert chunks[-1].section == "Two"


def test_chunking_splits_a_giant_paragraph_on_sentences() -> None:
    text = " ".join(f"This is sentence {i} of one very long paragraph without breaks." for i in range(120))
    chunks = chunk_document(_doc(text))
    assert len(chunks) > 3
    assert all(c.text.endswith(".") for c in chunks)


def test_heading_text_strips_markup_and_title_cases() -> None:
    assert _heading_text("+INTRODUCTION+") == "Introduction"
    assert _heading_text("4 _THE_ MALICIOUS AND UNTRUTHFUL WHITE PRESS") == "The Malicious And Untruthful White Press"
    assert _heading_text("HON. FRED. DOUGLASS'S LETTER") == "Hon. Fred. Douglass's Letter"


async def test_gutenberg_parse_strips_boilerplate_navigation_and_markup() -> None:
    doc = await fetch_gutenberg(FakeFetcher(PG_TEXT), 1, title="A Test Book", year=1900, scholar_id="s")  # type: ignore[arg-type]
    assert doc.license.id == "PD-US"
    assert "Project Gutenberg" not in doc.text and "Produced by" not in doc.text and "Transcriber" not in doc.text
    assert "Alpha, 1" not in doc.text  # the index is dropped
    assert "End of Project Gutenberg" not in doc.text
    assert "with an italic word" in doc.text
    assert "The First Chapter" in [s.title for s in doc.sections]


async def test_gutenberg_refuses_text_without_the_public_domain_notice() -> None:
    with pytest.raises(LicenceError):
        await fetch_gutenberg(
            FakeFetcher(PG_TEXT.replace("anyone anywhere", "someone")), 1, title="T", year=1, scholar_id="s"
        )  # type: ignore[arg-type]


async def test_wikipedia_sections_skip_navigation() -> None:
    payload = {
        "query": {
            "pages": [
                {
                    "title": "Ida B. Wells",
                    "fullurl": "https://en.wikipedia.org/wiki/Ida_B._Wells",
                    "pageprops": {"wikibase_item": "Q123"},
                    "revisions": [{"revid": 42}],
                    "extract": "Ida B. Wells was a journalist. She was born in 1862.\n\n== Early life ==\nShe grew up in Mississippi."
                    "\n\n== References ==\nA list of references.\n\n== See also ==\nOther pages.",
                }
            ]
        }
    }
    art = await fetch_wikipedia(FakeFetcher(payload), "Ida_B._Wells", "ida-b-wells", {"References", "See also"})  # type: ignore[arg-type]
    assert [s.title for s in art.document.sections] == ["Introduction", "Early life"]
    assert "references" not in art.document.text.lower()
    assert art.document.license.id == "CC-BY-SA-4.0" and "revision 42" in art.document.license.attribution
    assert art.document.source_url.endswith("oldid=42")


@pytest.mark.parametrize(
    ("href", "ok"),
    [
        ("https://creativecommons.org/licenses/by/4.0/", True),
        ("https://creativecommons.org/publicdomain/zero/1.0/", True),
        ("https://creativecommons.org/licenses/by-nc/4.0/", False),
        ("https://creativecommons.org/licenses/by-nd/4.0/", False),
    ],
)
def test_paper_licence_check(href: str, ok: bool) -> None:
    from lxml import etree

    xml = (
        f'<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><permissions><license>'
        f'<license-p><ext-link xlink:href="{href}">licence</ext-link></license-p></license></permissions></front></article>'
    )
    assert _xml_licence_ok(etree.fromstring(xml.encode())) is ok
