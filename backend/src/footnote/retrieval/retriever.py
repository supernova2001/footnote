"""Retrieval strategies, from plain BM25 to hybrid search with a cross-encoder rerank.

    bm25           lexical only
    dense          semantic only (Pinecone-hosted embeddings)
    hybrid         both, fused with reciprocal rank fusion
    hybrid_rerank  hybrid candidates reranked by a cross-encoder (the default)

All four exist so the eval can report what each stage buys (evals/retrieval.py); the UI exposes
the same switch. Fusion is RRF rather than a weighted score sum because BM25 and cosine scores are
on different scales and RRF needs no tuning to combine them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from footnote.db.repo import Repo
from footnote.domain import Passage, Strategy
from footnote.retrieval.store import Hit, SearchStore

RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hit]], k: int = RRF_K) -> list[Hit]:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (k + rank). Ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
    return [Hit(i, s) for i, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


class Retriever:
    def __init__(self, store: SearchStore, repo: Repo) -> None:
        self.store = store
        self.repo = repo

    async def search(
        self,
        query: str,
        *,
        strategy: Strategy = "hybrid_rerank",
        k: int = 8,
        candidates: int = 24,
        scholar_id: str | None = None,
        document_id: str | None = None,
    ) -> list[Passage]:
        hits = await self._rank(
            query, strategy=strategy, k=k, candidates=candidates, scholar_id=scholar_id, document_id=document_id
        )
        chunks = await self.repo.get_chunks(h.id for h in hits)
        # A hit whose chunk is missing from Mongo (index ahead of the database) is skipped,
        # never shown without its text and licence.
        return [Passage.from_chunk(chunks[h.id], h.score) for h in hits if h.id in chunks]

    async def search_many(
        self,
        queries: Sequence[str],
        *,
        strategy: Strategy = "hybrid_rerank",
        k: int = 8,
        candidates: int = 24,
        scholar_id: str | None = None,
    ) -> list[Passage]:
        """Search each sub-query, then fuse across queries and (for *_rerank) rerank once against the first."""
        if len(queries) == 1:
            return await self.search(queries[0], strategy=strategy, k=k, candidates=candidates, scholar_id=scholar_id)
        base = "hybrid" if strategy == "hybrid_rerank" else strategy
        per_query = await asyncio.gather(
            *(self._rank(q, strategy=base, k=candidates, candidates=candidates, scholar_id=scholar_id) for q in queries)
        )
        fused = reciprocal_rank_fusion(per_query)[:candidates]
        chunks = await self.repo.get_chunks(h.id for h in fused)
        fused = [h for h in fused if h.id in chunks]
        if strategy == "hybrid_rerank":
            question = " ".join(queries)
            fused = await self.store.rerank(question, [(h.id, chunks[h.id].text) for h in fused], top_n=k)
        return [Passage.from_chunk(chunks[h.id], h.score) for h in fused[:k]]

    async def _rank(
        self,
        query: str,
        *,
        strategy: Strategy,
        k: int,
        candidates: int,
        scholar_id: str | None,
        document_id: str | None = None,
    ) -> list[Hit]:
        if strategy == "bm25":
            return (await self.store.lexical(query, k, scholar_id, document_id))[:k]
        if strategy == "dense":
            return (await self.store.semantic(query, k, scholar_id, document_id))[:k]
        lexical, semantic = await asyncio.gather(
            self.store.lexical(query, candidates, scholar_id, document_id),
            self.store.semantic(query, candidates, scholar_id, document_id),
        )
        fused = reciprocal_rank_fusion([lexical, semantic])[:candidates]
        if strategy == "hybrid":
            return fused[:k]
        chunks = await self.repo.get_chunks(h.id for h in fused)
        docs = [(h.id, chunks[h.id].text) for h in fused if h.id in chunks]
        return await self.store.rerank(query, docs, top_n=k)
