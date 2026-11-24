"""Search backends.

`PineconeStore` uses one Pinecone document index whose `text` field is both BM25 full-text
searchable and embedded by a Pinecone-hosted model (llama-text-embed-v2), so lexical and semantic
search read the same records and no embedding model runs in this process. Reranking uses
Pinecone's hosted cross-encoder (bge-reranker-v2-m3).

`MemoryStore` is the offline twin: BM25 and a hashed bag-of-words "embedding", good enough to
exercise every code path without keys. It is never used to report quality numbers.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from footnote.domain import Chunk

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Hit:
    id: str
    score: float


class SearchStore(Protocol):
    async def ensure_index(self) -> None: ...
    async def upsert(self, chunks: Sequence[Chunk]) -> int: ...
    async def delete_document(self, document_id: str) -> None: ...
    async def lexical(
        self, query: str, k: int, scholar_id: str | None = None, document_id: str | None = None
    ) -> list[Hit]: ...
    async def semantic(
        self, query: str, k: int, scholar_id: str | None = None, document_id: str | None = None
    ) -> list[Hit]: ...
    async def rerank(self, query: str, docs: Sequence[tuple[str, str]], top_n: int) -> list[Hit]: ...


def _filter(scholar_id: str | None, document_id: str | None) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if scholar_id:
        clauses.append({"scholar_ids": {"$in": [scholar_id]}})
    if document_id:
        clauses.append({"document_id": {"$eq": document_id}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class PineconeStore:
    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        namespace: str,
        embed_model: str,
        embed_dimension: int,
        rerank_model: str,
        cloud: str,
        region: str,
    ) -> None:
        from pinecone import AsyncPinecone

        self._pc = AsyncPinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        self.embed_model = embed_model
        self.embed_dimension = embed_dimension
        self.rerank_model = rerank_model
        self.cloud = cloud
        self.region = region
        self._index: Any = None
        self._lock = asyncio.Lock()

    async def _idx(self) -> Any:
        async with self._lock:
            if self._index is None:
                self._index = await self._pc.index(name=self.index_name)
            return self._index

    async def ensure_index(self) -> None:
        if await self._pc.has_index(self.index_name):
            return
        log.info("pinecone_create_index", name=self.index_name, embed_model=self.embed_model)
        await self._pc.indexes.create(
            name=self.index_name,
            schema={
                "fields": {
                    # One text field, searched two ways: BM25 (full_text_search) and
                    # integrated dense embedding (embed). Pinecone embeds on write and on query.
                    "text": {
                        "type": "string",
                        "full_text_search": {"language": "en", "stemming": True},
                        "embed": {
                            "model": self.embed_model,
                            "dimension": self.embed_dimension,
                            "metric": "cosine",
                        },
                    },
                }
            },
            deployment={"deployment_type": "managed", "cloud": self.cloud, "region": self.region},
            tags={"app": "footnote"},
        )

    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        idx = await self._idx()
        records = [
            {
                "_id": c.id,
                "text": f"{c.title}. {c.section + '. ' if c.section else ''}{c.text}",
                "document_id": c.document_id,
                "scholar_ids": c.scholar_ids,
                "license_id": c.license_id,
            }
            for c in chunks
        ]
        result = await idx.documents.batch_upsert(
            namespace=self.namespace, documents=records, batch_size=48, show_progress=False
        )
        if result.failed_item_count:
            raise RuntimeError(f"Pinecone upsert: {result.failed_item_count} of {len(records)} records failed")
        return result.successful_item_count

    async def delete_document(self, document_id: str) -> None:
        idx = await self._idx()
        await idx.documents.delete(namespace=self.namespace, filter={"document_id": {"$eq": document_id}})

    async def _search(
        self, clause: dict[str, Any], k: int, scholar_id: str | None, document_id: str | None
    ) -> list[Hit]:
        idx = await self._idx()
        resp = await idx.documents.search(
            namespace=self.namespace, score_by=[clause], top_k=k, filter=_filter(scholar_id, document_id)
        )
        return [Hit(m.id, float(m.score or 0.0)) for m in resp.matches]

    async def lexical(
        self, query: str, k: int, scholar_id: str | None = None, document_id: str | None = None
    ) -> list[Hit]:
        return await self._search({"type": "text", "field": "text", "query": query}, k, scholar_id, document_id)

    async def semantic(
        self, query: str, k: int, scholar_id: str | None = None, document_id: str | None = None
    ) -> list[Hit]:
        return await self._search({"type": "embed", "field": "text", "query": query}, k, scholar_id, document_id)

    async def rerank(self, query: str, docs: Sequence[tuple[str, str]], top_n: int) -> list[Hit]:
        if not docs:
            return []
        result = await self._pc.inference.rerank(
            model=self.rerank_model,
            query=query,
            documents=[{"id": i, "text": t} for i, t in docs],
            top_n=min(top_n, len(docs)),
            return_documents=False,
            parameters={"truncate": "END"},
        )
        return [Hit(docs[r.index][0], float(r.score)) for r in result.data]

    async def close(self) -> None:
        await self._pc.close()


# ── offline twin ─────────────────────────────────────────────────────────────────────────────

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "were",
        "which",
        "who",
        "will",
        "with",
        "not",
        "what",
        "when",
        "where",
        "why",
        "how",
        "did",
        "does",
        "do",
    ]
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class MemoryStore:
    """BM25 + hashed bag-of-words cosine over chunks held in memory (offline mode, tests)."""

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims
        self.docs: dict[str, Chunk] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._vec: dict[str, list[float]] = {}
        self._df: Counter[str] = Counter()

    async def ensure_index(self) -> None:
        return None

    def _embed(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        for t in tokens(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        for c in chunks:
            if c.id in self._tf:
                self._df.subtract(set(self._tf[c.id]))
            tf = Counter(tokens(f"{c.title} {c.section or ''} {c.text}"))
            self.docs[c.id], self._tf[c.id], self._vec[c.id] = c, tf, self._embed(c.text)
            self._df.update(set(tf))
        return len(chunks)

    async def delete_document(self, document_id: str) -> None:
        for cid in [i for i, c in self.docs.items() if c.document_id == document_id]:
            self._df.subtract(set(self._tf.pop(cid)))
            del self.docs[cid], self._vec[cid]

    def _candidates(self, scholar_id: str | None, document_id: str | None = None) -> list[str]:
        return [
            i
            for i, c in self.docs.items()
            if (scholar_id is None or scholar_id in c.scholar_ids)
            and (document_id is None or c.document_id == document_id)
        ]

    async def lexical(
        self, query: str, k: int, scholar_id: str | None = None, document_id: str | None = None
    ) -> list[Hit]:
        q = tokens(query)
        n = max(len(self.docs), 1)
        avg = sum(sum(tf.values()) for tf in self._tf.values()) / n if self._tf else 1.0
        scored = []
        for cid in self._candidates(scholar_id, document_id):
            tf, length = self._tf[cid], sum(self._tf[cid].values())
            s = 0.0
            for t in q:
                if tf[t]:
                    idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                    s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * length / avg))
            if s > 0:
                scored.append(Hit(cid, s))
        return sorted(scored, key=lambda h: h.score, reverse=True)[:k]

    async def semantic(
        self, query: str, k: int, scholar_id: str | None = None, document_id: str | None = None
    ) -> list[Hit]:
        qv = self._embed(query)
        scored = [
            Hit(cid, sum(a * b for a, b in zip(qv, self._vec[cid], strict=True)))
            for cid in self._candidates(scholar_id, document_id)
        ]
        return sorted([h for h in scored if h.score > 0], key=lambda h: h.score, reverse=True)[:k]

    async def rerank(self, query: str, docs: Sequence[tuple[str, str]], top_n: int) -> list[Hit]:
        q = set(tokens(query))
        scored = [Hit(i, len(q & set(tokens(t))) / (len(q) or 1)) for i, t in docs]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_n]
