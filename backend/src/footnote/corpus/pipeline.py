"""Ingest: fetch → check licence → normalise → chunk → store (MongoDB) → index (Pinecone).

Idempotent: a document whose text hash is unchanged is not re-chunked or re-indexed unless forced.
Every skip is reported with its reason; nothing without an accepted licence is stored.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from footnote.corpus.chunking import chunk_document, split_sentences
from footnote.corpus.http import Fetcher
from footnote.corpus.sources import (
    LicenceError,
    fetch_gutenberg,
    fetch_life_years,
    fetch_paper,
    fetch_wikipedia,
    search_papers,
    slugify,
)
from footnote.db.repo import Repo
from footnote.domain import Document, Scholar
from footnote.retrieval.store import SearchStore

log = structlog.get_logger(__name__)


@dataclass
class IngestReport:
    scholars: int = 0
    documents: int = 0
    unchanged: int = 0
    chunks: int = 0
    indexed: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__


def _summary_from_lead(lead: str) -> str:
    sentences = split_sentences(lead)
    return " ".join(sentences[:2])


class Ingestor:
    def __init__(self, *, repo: Repo, store: SearchStore | None, fetcher: Fetcher, manifest_path: Path) -> None:
        self.repo = repo
        self.store = store
        self.fetcher = fetcher
        self.manifest_path = manifest_path
        self.lock_path = manifest_path.with_name("corpus.lock.json")
        with manifest_path.open(encoding="utf-8") as f:
            self.manifest: dict[str, Any] = yaml.safe_load(f)

    async def run(self, *, only: set[str] | None = None, papers: bool = True, force: bool = False) -> IngestReport:
        report = IngestReport()
        if self.store is not None:
            await self.store.ensure_index()
        await self.repo.ensure_indexes()
        skip_sections = set(self.manifest.get("wikipedia_skip_sections", []))

        for entry in self.manifest["scholars"]:
            sid = entry["id"]
            if only and sid not in only:
                continue
            article, born, died = None, entry.get("born"), entry.get("died")
            try:
                article = await fetch_wikipedia(self.fetcher, entry["wikipedia"], sid, skip_sections)
                born, died = await fetch_life_years(self.fetcher, article.qid)
            except Exception as exc:  # one bad source must not stop the run; the works still load
                report.skipped.append({"id": f"wikipedia-{sid}", "reason": str(exc)})
            docs = [article.document] if article else []
            for work in entry.get("works", []):
                try:
                    docs.append(
                        await fetch_gutenberg(
                            self.fetcher,
                            work["gutenberg"],
                            title=work["title"],
                            year=work.get("year"),
                            scholar_id=sid,
                            author=entry["name"],
                        )
                    )
                except (LicenceError, ValueError, LookupError) as exc:
                    report.skipped.append({"id": f"gutenberg-{work['gutenberg']}", "reason": str(exc)})
            if not docs:
                continue
            for doc in docs:
                await self._store(doc, report, force)
            await self.repo.upsert_scholar(
                Scholar(
                    id=sid,
                    name=entry["name"],
                    born=born,
                    died=died,
                    summary=_summary_from_lead(article.lead) if article else "",
                    kind="historical",
                    document_count=len(docs),
                    wikipedia_url=article.url if article else None,
                )
            )
            report.scholars += 1

        if papers and not only and self.manifest.get("papers"):
            await self._ingest_papers(report, force)
        log.info(
            "ingest_done", **{k: v for k, v in report.as_dict().items() if k != "skipped"}, skipped=len(report.skipped)
        )
        return report

    async def _ingest_papers(self, report: IngestReport, force: bool) -> None:
        cfg = self.manifest["papers"]
        locked: list[str] = json.loads(self.lock_path.read_text())["pmcids"] if self.lock_path.exists() else []
        metas = await search_papers(self.fetcher, cfg["query"], cfg.get("sort", "CITED desc"), cfg.get("limit", 12) * 3)
        by_id = {m.pmcid: m for m in metas}
        if locked:
            missing = [p for p in locked if p not in by_id]
            for p in missing:
                report.skipped.append({"id": p, "reason": "locked paper no longer returned by the query"})
            chosen = [by_id[p] for p in locked if p in by_id]
        else:
            chosen = metas[: cfg.get("limit", 12)]
        stored: list[str] = []
        names: dict[str, str] = {}
        for meta in chosen:
            authors = meta.authors[: cfg.get("max_authors", 8)]
            author_ids = [slugify(a) for a in authors]
            try:
                doc = await fetch_paper(self.fetcher, meta, author_ids)
            except (LicenceError, ValueError) as exc:
                report.skipped.append({"id": meta.pmcid, "reason": str(exc)})
                continue
            await self._store(doc, report, force)
            stored.append(meta.pmcid)
            names.update(zip(author_ids, authors, strict=True))
        # Author pages say only what the papers' bylines say: name and which papers are here.
        for aid, name in names.items():
            docs = await self.repo.list_documents(aid)
            titles = "; ".join(f"“{d.title}” ({d.year})" for d in docs[:2])
            more = f" and {len(docs) - 2} more" if len(docs) > 2 else ""
            await self.repo.upsert_scholar(
                Scholar(
                    id=aid,
                    name=name,
                    kind="author",
                    summary=f"Author of {titles}{more}.",
                    document_count=len(docs),
                )
            )
            report.scholars += 1
        if not locked:
            self.lock_path.write_text(json.dumps({"query": cfg["query"], "pmcids": stored}, indent=2) + "\n")

    async def _store(self, doc: Document, report: IngestReport, force: bool) -> None:
        existing = await self.repo.get_document(doc.id)
        if existing and existing.sha256 == doc.sha256 and not force:
            report.unchanged += 1
            return
        chunks = chunk_document(doc)
        doc.chunk_count = len(chunks)
        await self.repo.upsert_document(doc)
        await self.repo.replace_chunks(doc.id, chunks)
        if self.store is not None:
            if existing:
                await self.store.delete_document(doc.id)
            report.indexed += await self.store.upsert(chunks)
        report.documents += 1
        report.chunks += len(chunks)


async def export_chunks(repo: Repo, path: Path) -> int:
    """One Chunk per line, for finetune/build_dataset.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        async for chunk in repo.iter_chunks():
            f.write(chunk.model_dump_json() + "\n")
            n += 1
    return n


async def export_snapshot(repo: Repo, path: Path, scholar_ids: list[str]) -> dict[str, int]:
    """A small, licensed slice of the corpus for the offline mode (FOOTNOTE_OFFLINE=1)."""
    scholars, documents = [], []
    for sid in scholar_ids:
        scholar = await repo.get_scholar(sid)
        if scholar is None:
            continue
        scholars.append(scholar.model_dump(mode="json"))
        for summary in await repo.list_documents(sid):
            doc = await repo.get_document(summary.id)
            if doc is not None:
                documents.append(doc.model_dump(mode="json"))
    payload = json.dumps({"scholars": scholars, "documents": documents}, ensure_ascii=False)
    await asyncio.to_thread(_write_text, path, payload)
    return {"scholars": len(scholars), "documents": len(documents)}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


async def load_snapshot(repo: Repo, store: SearchStore, path: Path) -> dict[str, int]:
    data = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
    chunks = 0
    for s in data["scholars"]:
        await repo.upsert_scholar(Scholar.model_validate(s))
    for d in data["documents"]:
        doc = Document.model_validate(d)
        doc_chunks = chunk_document(doc)
        doc.chunk_count = len(doc_chunks)
        await repo.upsert_document(doc)
        await repo.replace_chunks(doc.id, doc_chunks)
        await store.upsert(doc_chunks)
        chunks += len(doc_chunks)
    return {"scholars": len(data["scholars"]), "documents": len(data["documents"]), "chunks": chunks}
