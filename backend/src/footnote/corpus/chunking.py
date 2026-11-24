"""Section-aware chunking with exact character offsets.

A chunk never crosses a section boundary, is built from whole paragraphs where it can, and splits
an over-long paragraph on sentence boundaries. Offsets point into the document text, so a citation
can always be traced back to the exact span of the source. There is no overlap between chunks:
overlap would make two passages cite the same sentence, and the judge and the UI would disagree
about which one supports a claim.

This is deterministic code by design (it is passage splitting, not a decision): changing it changes
chunk ids, which would orphan citations on saved drafts.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from footnote.domain import Chunk, Document, Section

TARGET_CHARS = 1000
MAX_CHARS = 1600
MIN_CHARS = 200
MIN_WORDS = 6
_WORD = re.compile(r"[A-Za-z]{2,}")

_PARA = re.compile(r"\n\s*\n")
# Sentence end: . ! ? (optionally followed by a closing quote/bracket), then whitespace and a
# capital, quote or digit. Abbreviations and initials are protected below.
_SENT_END = re.compile(r"(?<=[.!?])([\"'”’)\]]?)\s+(?=[\"'“‘(\[]?[A-Z0-9])")
_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Rev|Gen|Col|Capt|Lt|Sr|Jr|St|Mt|vs|etc|e\.g|i\.e|cf|al|No|Vol|pp|ch|Fig|U\.S|Ph\.D)\.$"
)
_INITIAL = re.compile(r"\b[A-Z]\.$")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping 'W. E. B. Du Bois' and 'Dr. Woodson' whole."""
    text = " ".join(text.split())
    return [text[a:b].strip() for a, b in _sentence_spans(text, 0, len(text)) if text[a:b].strip()]


def _paragraph_spans(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    pos = start
    for m in _PARA.finditer(text, start, end):
        if text[pos : m.start()].strip():
            yield pos, m.start()
        pos = m.end()
    if text[pos:end].strip():
        yield pos, end


def _sentence_spans(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    """Sentence-ish spans of text[start:end] by offset (whitespace inside is preserved)."""
    pos = start
    for m in _SENT_END.finditer(text, start, end):
        head = text[pos : m.start()].rstrip()
        if _ABBREV.search(head) or _INITIAL.search(head):
            continue
        yield pos, m.start() + len(m.group(1))  # end after the punctuation and any closing quote
        pos = m.end()
    if text[pos:end].strip():
        yield pos, end


def _pieces(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    """Paragraphs, with any paragraph longer than MAX_CHARS broken into sentence groups."""
    for p_start, p_end in _paragraph_spans(text, start, end):
        if p_end - p_start <= MAX_CHARS:
            yield p_start, p_end
            continue
        group_start = None
        group_end = p_start
        for s_start, s_end in _sentence_spans(text, p_start, p_end):
            if group_start is None:
                group_start = s_start
            elif s_end - group_start > TARGET_CHARS:
                yield group_start, group_end
                group_start = s_start
            group_end = s_end
            while group_end - group_start > MAX_CHARS:  # one enormous "sentence"
                cut = group_start + MAX_CHARS
                yield group_start, cut
                group_start = cut
        if group_start is not None and group_end > group_start:
            yield group_start, group_end


def chunk_document(doc: Document) -> list[Chunk]:
    sections = doc.sections or [Section(title="", start=0, end=len(doc.text))]
    chunks: list[Chunk] = []

    def emit(section: Section, a: int, b: int) -> None:
        body = " ".join(doc.text[a:b].split())
        if len(_WORD.findall(body)) < MIN_WORDS:  # a stray page number or signature line
            return
        chunks.append(
            Chunk(
                id=f"{doc.id}:{len(chunks):04d}",
                document_id=doc.id,
                ordinal=len(chunks),
                title=doc.title,
                section=section.title or None,
                text=body,
                char_start=a,
                char_end=b,
                scholar_ids=doc.scholar_ids,
                license_id=doc.license.id,
                source_url=doc.source_url,
                attribution=doc.license.attribution,
            )
        )

    for section in sections:
        cur_start: int | None = None
        cur_end = section.start
        for a, b in _pieces(doc.text, section.start, section.end):
            if cur_start is None:
                cur_start, cur_end = a, b
            elif b - cur_start > TARGET_CHARS and cur_end - cur_start >= MIN_CHARS:
                emit(section, cur_start, cur_end)
                cur_start, cur_end = a, b
            else:
                cur_end = b
        if cur_start is not None:
            # A short tail merges into the previous chunk of the same section when it fits.
            if (
                cur_end - cur_start < MIN_CHARS
                and chunks
                and chunks[-1].section == (section.title or None)
                and chunks[-1].document_id == doc.id
                and cur_end - chunks[-1].char_start <= MAX_CHARS
            ):
                last = chunks.pop()
                emit(section, last.char_start, cur_end)
            else:
                emit(section, cur_start, cur_end)
    return chunks
