"""Small, dependency-free text helpers: sentence spans, content words, names and numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words before a full stop that do not end a sentence ("Dr. Woodson", "St. Louis", "e.g. the").
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "st",
    "jr",
    "sr",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "no",
    "vol",
    "pp",
    "fig",
    "ca",
    "cf",
    "al",
    "inc",
    "co",
    "gen",
    "rev",
    "hon",
    "capt",
    "col",
    "lt",
    "sgt",
    "mt",
    "ft",
    "ed",
    "eds",
    "rep",
    "sen",
    "gov",
    "pres",
    "univ",
    "dept",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "u.s",
    "u.k",
    "a.m",
    "p.m",
}
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])")

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "as",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "they",
    "his",
    "her",
    "their",
    "him",
    "them",
    "we",
    "our",
    "you",
    "i",
    "had",
    "has",
    "have",
    "not",
    "no",
    "so",
    "than",
    "then",
    "there",
    "which",
    "who",
    "whom",
    "whose",
    "what",
    "when",
    "where",
    "while",
    "into",
    "over",
    "also",
    "such",
    "would",
    "could",
    "should",
    "will",
    "can",
    "may",
    "might",
    "do",
    "did",
    "does",
    "all",
    "any",
    "some",
    "more",
    "most",
    "other",
    "about",
    "after",
    "before",
    "between",
    "through",
    "during",
    "upon",
    "if",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d[\d,]*")
_NAME = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]\.)(?:\s+(?:[A-Z][a-z]+|[A-Z]\.|de|du|van|von|la))*\s+[A-Z][a-z]+\b")
# A whole number, with optional thousands separators. A comma or full stop right after it is
# punctuation ("in 1911, which"), unless a digit follows ("1,200", "3.5").
_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?!\w|,\d|\.\d)")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


def sentence_spans(text: str) -> list[Span]:
    """Split into sentences, keeping exact character offsets so a sentence is always a verbatim quote."""
    spans: list[Span] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        end = match.start() + 1
        # Include closing quotes/brackets that belong to this sentence.
        while end < len(text) and text[end] in "\"'”’)]":
            end += 1
        before = text[start:end].rstrip(".!?\"'”’)]")
        last_word = before.split()[-1].lower() if before.split() else ""
        if last_word.rstrip(".") in _ABBREVIATIONS or re.fullmatch(r"[a-z]", last_word):
            continue  # "W. E. B." or "Dr." is not the end of a sentence
        piece = text[start:end].strip()
        if piece:
            offset = text.index(piece, start)
            spans.append(Span(offset, offset + len(piece), piece))
        start = match.end()
    tail = text[start:].strip()
    if tail:
        offset = text.index(tail, start)
        spans.append(Span(offset, offset + len(tail), tail))
    return spans


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in _STOPWORDS and len(w) > 2}


def coverage(claim: str, passage_text: str) -> float:
    """Share of the claim's content words that occur in the passage (0..1)."""
    claim_words = content_words(claim)
    if not claim_words:
        return 0.0
    return len(claim_words & content_words(passage_text)) / len(claim_words)


def names(text: str) -> list[str]:
    """Multi-word capitalised names ("Ida B. Wells", "Atlanta University"), in order of appearance."""
    return [m.group(0) for m in _NAME.finditer(text)]


def numbers(text: str) -> list[re.Match[str]]:
    return list(_NUMBER.finditer(text))


def is_good_anchor(sentence: str) -> bool:
    """A sentence worth building claims from: long enough to say something, not a heading or a list."""
    if not 40 <= len(sentence) <= 320:
        return False
    tokens = sentence.split()
    if len(tokens) < 7:
        return False
    letters = [c for c in sentence if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) > 0.3:
        return False
    if sentence[-1] not in ".!?\"'”’)":
        return False
    return len(content_words(sentence)) >= 4
