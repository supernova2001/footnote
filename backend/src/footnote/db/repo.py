"""Persistence. MongoDB is the source of truth for text; Pinecone only holds what search needs.

`Repo` is the interface the services use. `MongoRepo` is production; `MemoryRepo` backs the
offline mode and the unit tests with the same behaviour (both are exercised by the same tests).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from datetime import date
from typing import Any, Protocol

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, ReplaceOne, ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from footnote.domain import Chunk, Document, DocumentSummary, Scholar, utcnow


class Repo(Protocol):
    async def ensure_indexes(self) -> None: ...
    async def upsert_scholar(self, scholar: Scholar) -> None: ...
    async def get_scholar(self, scholar_id: str) -> Scholar | None: ...
    async def list_scholars(self) -> list[Scholar]: ...
    async def upsert_document(self, document: Document) -> None: ...
    async def get_document(self, document_id: str) -> Document | None: ...
    async def list_documents(self, scholar_id: str | None = None) -> list[DocumentSummary]: ...
    async def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None: ...
    async def get_chunks(self, ids: Iterable[str]) -> dict[str, Chunk]: ...
    def iter_chunks(self, document_id: str | None = None) -> AsyncIterator[Chunk]: ...
    async def counts(self) -> dict[str, int]: ...
    async def save_draft(self, draft: dict[str, Any]) -> None: ...
    async def get_draft(self, draft_id: str, owner_id: str | None = None) -> dict[str, Any] | None: ...
    async def list_drafts(self, limit: int = 50, owner_id: str | None = None) -> list[dict[str, Any]]: ...
    async def claim_draft_review(self, draft_id: str, owner_id: str) -> bool: ...
    async def release_draft_review(self, draft_id: str, owner_id: str) -> None: ...
    async def record_judgment(self, judgment: dict[str, Any]) -> None: ...
    def iter_judgments(self) -> AsyncIterator[dict[str, Any]]: ...
    async def record_run(self, run: dict[str, Any]) -> None: ...
    async def add_spend(self, usd: float, day: date | None = None) -> float: ...
    async def get_spend(self, day: date | None = None) -> float: ...


_SUMMARY_FIELDS = {"text": 0, "sections": 0}


def _summary(doc: dict[str, Any]) -> DocumentSummary:
    doc = {k: v for k, v in doc.items() if k not in ("text", "sections")}
    doc["id"] = doc.pop("_id")
    return DocumentSummary.model_validate(doc)


class MongoRepo:
    def __init__(self, client: AsyncMongoClient[dict[str, Any]], db_name: str) -> None:
        self.client = client
        self.db: AsyncDatabase[dict[str, Any]] = client[db_name]

    async def ensure_indexes(self) -> None:
        await self.db.documents.create_index([("scholar_ids", ASCENDING)])
        await self.db.chunks.create_index([("document_id", ASCENDING), ("ordinal", ASCENDING)])
        await self.db.drafts.create_index([("updated_at", DESCENDING)])
        await self.db.judgments.create_index([("created_at", DESCENDING)])
        await self.db.judgments.create_index([("prompt_version", ASCENDING), ("judge", ASCENDING)])
        await self.db.runs.create_index([("created_at", DESCENDING)])

    async def upsert_scholar(self, scholar: Scholar) -> None:
        doc = scholar.model_dump()
        await self.db.scholars.replace_one({"_id": doc.pop("id")}, doc, upsert=True)

    async def get_scholar(self, scholar_id: str) -> Scholar | None:
        doc = await self.db.scholars.find_one({"_id": scholar_id})
        return Scholar.model_validate({**doc, "id": doc["_id"]}) if doc else None

    async def list_scholars(self) -> list[Scholar]:
        cursor = self.db.scholars.find({}).sort([("kind", DESCENDING), ("name", ASCENDING)])
        return [Scholar.model_validate({**d, "id": d["_id"]}) async for d in cursor]

    async def upsert_document(self, document: Document) -> None:
        doc = document.model_dump()
        await self.db.documents.replace_one({"_id": doc.pop("id")}, doc, upsert=True)

    async def get_document(self, document_id: str) -> Document | None:
        doc = await self.db.documents.find_one({"_id": document_id})
        return Document.model_validate({**doc, "id": doc["_id"]}) if doc else None

    async def list_documents(self, scholar_id: str | None = None) -> list[DocumentSummary]:
        query = {"scholar_ids": scholar_id} if scholar_id else {}
        cursor = self.db.documents.find(query, _SUMMARY_FIELDS).sort([("year", ASCENDING), ("title", ASCENDING)])
        return [_summary(d) async for d in cursor]

    async def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        keep = [c.id for c in chunks]
        await self.db.chunks.delete_many({"document_id": document_id, "_id": {"$nin": keep}})
        if chunks:
            ops = [ReplaceOne({"_id": c.id}, c.model_dump(exclude={"id"}), upsert=True) for c in chunks]
            for start in range(0, len(ops), 1000):
                await self.db.chunks.bulk_write(ops[start : start + 1000], ordered=False)

    async def get_chunks(self, ids: Iterable[str]) -> dict[str, Chunk]:
        wanted = list(dict.fromkeys(ids))
        cursor = self.db.chunks.find({"_id": {"$in": wanted}})
        return {d["_id"]: Chunk.model_validate({**d, "id": d["_id"]}) async for d in cursor}

    async def iter_chunks(self, document_id: str | None = None) -> AsyncIterator[Chunk]:
        query = {"document_id": document_id} if document_id else {}
        async for d in self.db.chunks.find(query).sort([("document_id", ASCENDING), ("ordinal", ASCENDING)]):
            yield Chunk.model_validate({**d, "id": d["_id"]})

    async def counts(self) -> dict[str, int]:
        return {
            "scholars": await self.db.scholars.estimated_document_count(),
            "documents": await self.db.documents.estimated_document_count(),
            "chunks": await self.db.chunks.estimated_document_count(),
        }

    async def save_draft(self, draft: dict[str, Any]) -> None:
        draft = {**draft, "updated_at": utcnow()}
        draft.setdefault("created_at", draft["updated_at"])
        await self.db.drafts.replace_one({"_id": draft["id"]}, draft, upsert=True)

    async def get_draft(self, draft_id: str, owner_id: str | None = None) -> dict[str, Any] | None:
        query: dict[str, Any] = {"_id": draft_id}
        if owner_id is not None:
            query["owner_id"] = owner_id
        doc = await self.db.drafts.find_one(query, {"_id": 0})
        return doc

    async def list_drafts(self, limit: int = 50, owner_id: str | None = None) -> list[dict[str, Any]]:
        fields = {"_id": 0, "id": 1, "title": 1, "audience": 1, "status": 1, "created_at": 1, "updated_at": 1}
        query = {"owner_id": owner_id} if owner_id is not None else {}
        cursor = self.db.drafts.find(query, fields).sort("updated_at", DESCENDING).limit(limit)
        return [d async for d in cursor]

    async def claim_draft_review(self, draft_id: str, owner_id: str) -> bool:
        doc = await self.db.drafts.find_one_and_update(
            {"_id": draft_id, "owner_id": owner_id, "status": "awaiting_review"},
            {"$set": {"status": "reviewing", "updated_at": utcnow()}},
            return_document=ReturnDocument.AFTER,
        )
        return doc is not None

    async def release_draft_review(self, draft_id: str, owner_id: str) -> None:
        await self.db.drafts.update_one(
            {"_id": draft_id, "owner_id": owner_id, "status": "reviewing"},
            {"$set": {"status": "awaiting_review", "updated_at": utcnow()}},
        )

    async def record_judgment(self, judgment: dict[str, Any]) -> None:
        await self.db.judgments.insert_one({**judgment, "created_at": utcnow()})

    async def iter_judgments(self) -> AsyncIterator[dict[str, Any]]:
        async for d in self.db.judgments.find({}, {"_id": 0}).sort("created_at", ASCENDING):
            yield d

    async def record_run(self, run: dict[str, Any]) -> None:
        await self.db.runs.insert_one({**run, "created_at": utcnow()})

    async def add_spend(self, usd: float, day: date | None = None) -> float:
        key = (day or utcnow().date()).isoformat()
        doc = await self.db.spend.find_one_and_update(
            {"_id": key}, {"$inc": {"usd": usd}}, upsert=True, return_document=ReturnDocument.AFTER
        )
        return float(doc["usd"]) if doc else usd

    async def get_spend(self, day: date | None = None) -> float:
        key = (day or utcnow().date()).isoformat()
        doc = await self.db.spend.find_one({"_id": key})
        return float(doc["usd"]) if doc else 0.0


class MemoryRepo:
    """In-process twin of MongoRepo for the offline mode and unit tests."""

    def __init__(self) -> None:
        self.scholars: dict[str, Scholar] = {}
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.drafts: dict[str, dict[str, Any]] = {}
        self.judgments: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.spend: dict[str, float] = {}

    async def ensure_indexes(self) -> None:
        return None

    async def upsert_scholar(self, scholar: Scholar) -> None:
        self.scholars[scholar.id] = scholar

    async def get_scholar(self, scholar_id: str) -> Scholar | None:
        return self.scholars.get(scholar_id)

    async def list_scholars(self) -> list[Scholar]:
        return sorted(self.scholars.values(), key=lambda s: (s.kind != "historical", s.name))

    async def upsert_document(self, document: Document) -> None:
        self.documents[document.id] = document

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def list_documents(self, scholar_id: str | None = None) -> list[DocumentSummary]:
        docs = [d for d in self.documents.values() if scholar_id is None or scholar_id in d.scholar_ids]
        docs.sort(key=lambda d: (d.year or 0, d.title))
        return [
            DocumentSummary.model_validate(d.model_dump(exclude={"text", "sections", "retrieved_at", "sha256"}))
            for d in docs
        ]

    async def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        for cid in [k for k, c in self.chunks.items() if c.document_id == document_id]:
            del self.chunks[cid]
        self.chunks.update({c.id: c for c in chunks})

    async def get_chunks(self, ids: Iterable[str]) -> dict[str, Chunk]:
        return {i: self.chunks[i] for i in ids if i in self.chunks}

    async def iter_chunks(self, document_id: str | None = None) -> AsyncIterator[Chunk]:
        for c in sorted(self.chunks.values(), key=lambda c: (c.document_id, c.ordinal)):
            if document_id is None or c.document_id == document_id:
                yield c

    async def counts(self) -> dict[str, int]:
        return {"scholars": len(self.scholars), "documents": len(self.documents), "chunks": len(self.chunks)}

    async def save_draft(self, draft: dict[str, Any]) -> None:
        now = utcnow()
        previous = self.drafts.get(draft["id"], {})
        self.drafts[draft["id"]] = {**draft, "created_at": previous.get("created_at", now), "updated_at": now}

    async def get_draft(self, draft_id: str, owner_id: str | None = None) -> dict[str, Any] | None:
        draft = self.drafts.get(draft_id)
        return draft if draft is not None and (owner_id is None or draft.get("owner_id") == owner_id) else None

    async def list_drafts(self, limit: int = 50, owner_id: str | None = None) -> list[dict[str, Any]]:
        owned = [d for d in self.drafts.values() if owner_id is None or d.get("owner_id") == owner_id]
        rows = sorted(owned, key=lambda d: d["updated_at"], reverse=True)[:limit]
        keys = ("id", "title", "audience", "status", "created_at", "updated_at")
        return [{k: d.get(k) for k in keys} for d in rows]

    async def claim_draft_review(self, draft_id: str, owner_id: str) -> bool:
        draft = self.drafts.get(draft_id)
        if draft is None or draft.get("owner_id") != owner_id or draft.get("status") != "awaiting_review":
            return False
        draft.update(status="reviewing", updated_at=utcnow())
        return True

    async def release_draft_review(self, draft_id: str, owner_id: str) -> None:
        draft = self.drafts.get(draft_id)
        if draft is not None and draft.get("owner_id") == owner_id and draft.get("status") == "reviewing":
            draft.update(status="awaiting_review", updated_at=utcnow())

    async def record_judgment(self, judgment: dict[str, Any]) -> None:
        self.judgments.append({**judgment, "created_at": utcnow()})

    async def iter_judgments(self) -> AsyncIterator[dict[str, Any]]:
        for j in self.judgments:
            yield j

    async def record_run(self, run: dict[str, Any]) -> None:
        self.runs.append({**run, "created_at": utcnow()})

    async def add_spend(self, usd: float, day: date | None = None) -> float:
        key = (day or utcnow().date()).isoformat()
        self.spend[key] = self.spend.get(key, 0.0) + usd
        return self.spend[key]

    async def get_spend(self, day: date | None = None) -> float:
        return self.spend.get((day or utcnow().date()).isoformat(), 0.0)
