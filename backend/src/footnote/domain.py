"""The shapes every layer agrees on. Mirrors docs/api-contract.md; change that file first."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

LicenseId = Literal["CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "PD-US"]
DocumentKind = Literal["biography", "work", "paper"]
ScholarKind = Literal["historical", "author"]
VerdictLabel = Literal["supported", "partial", "unsupported"]
Audience = Literal["grade4", "grade8", "adult"]
Strategy = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class License(BaseModel):
    id: LicenseId
    name: str
    url: str
    attribution: str


class Scholar(BaseModel):
    id: str
    name: str
    born: int | None = None
    died: int | None = None
    summary: str = ""
    kind: ScholarKind
    document_count: int = 0
    wikipedia_url: str | None = None


class Section(BaseModel):
    title: str
    start: int
    end: int


class Document(BaseModel):
    """A source as fetched: the full text is kept in MongoDB, never in the vector index."""

    id: str
    title: str
    kind: DocumentKind
    year: int | None = None
    scholar_ids: list[str]
    source_url: str
    license: License
    retrieved_at: datetime = Field(default_factory=utcnow)
    sha256: str
    text: str
    sections: list[Section] = []
    chunk_count: int = 0


class DocumentSummary(BaseModel):
    id: str
    title: str
    kind: DocumentKind
    year: int | None
    scholar_ids: list[str]
    source_url: str
    license: License
    chunk_count: int


class Chunk(BaseModel):
    """A citable passage. `id` is stable across re-ingestion as long as the text is unchanged."""

    id: str  # "<document_id>:<ordinal:04d>"
    document_id: str
    ordinal: int
    title: str
    section: str | None
    text: str
    char_start: int
    char_end: int
    scholar_ids: list[str]
    license_id: LicenseId
    source_url: str
    attribution: str = ""  # the licence's attribution line, shown wherever the passage is


class Passage(BaseModel):
    id: str
    document_id: str
    title: str
    section: str | None
    text: str
    scholar_ids: list[str]
    license_id: LicenseId
    source_url: str
    attribution: str = ""
    score: float | None = None

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float | None = None) -> Passage:
        return cls(
            id=chunk.id,
            document_id=chunk.document_id,
            title=chunk.title,
            section=chunk.section,
            text=chunk.text,
            scholar_ids=chunk.scholar_ids,
            license_id=chunk.license_id,
            source_url=chunk.source_url,
            attribution=chunk.attribution,
            score=score,
        )


class Citation(BaseModel):
    passage_id: str
    cited_text: str


class Verdict(BaseModel):
    label: VerdictLabel
    evidence: str
    rationale: str
    judge: str


class Segment(BaseModel):
    text: str
    citations: list[Citation] = []
    verdict: Verdict | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
        )
