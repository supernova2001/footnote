"""One contract, two implementations: MemoryRepo always, MongoRepo when MONGODB_URI_TEST is set (CI)."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest

from footnote.corpus.chunking import chunk_document
from footnote.db.repo import MemoryRepo, MongoRepo, Repo

from .conftest import FIXTURE_DOCS, FIXTURE_SCHOLARS

MONGO = os.environ.get("MONGODB_URI_TEST")
params = ["memory"] + (["mongo"] if MONGO else [])


@pytest.fixture(params=params)
async def repo(request: pytest.FixtureRequest) -> AsyncIterator[Repo]:
    if request.param == "memory":
        yield MemoryRepo()
        return
    from pymongo import AsyncMongoClient

    client: AsyncMongoClient = AsyncMongoClient(MONGO)  # type: ignore[type-arg]
    name = f"footnote_test_{uuid.uuid4().hex[:8]}"
    mongo = MongoRepo(client, name)
    await mongo.ensure_indexes()
    yield mongo
    await client.drop_database(name)
    await client.close()


async def test_documents_chunks_and_scholars_round_trip(repo: Repo) -> None:
    for s in FIXTURE_SCHOLARS:
        await repo.upsert_scholar(s)
    for d in FIXTURE_DOCS:
        await repo.upsert_document(d)
        await repo.replace_chunks(d.id, chunk_document(d))
    assert {s.id for s in await repo.list_scholars()} == {"carter-g-woodson", "ida-b-wells"}
    docs = await repo.list_documents("carter-g-woodson")
    assert [d.id for d in docs] == ["gutenberg-11089", "wikipedia-carter-g-woodson"] or len(docs) == 2
    full = await repo.get_document("gutenberg-14975")
    assert full is not None and full.text.startswith("Ida B. Wells")
    ids = [c.id async for c in repo.iter_chunks("gutenberg-14975")]
    got = await repo.get_chunks([*ids, "missing:0000"])
    assert set(got) == set(ids) and got[ids[0]].license_id == "PD-US" and got[ids[0]].attribution
    counts = await repo.counts()
    assert counts["documents"] == 3 and counts["scholars"] == 2


async def test_replace_chunks_removes_stale_ones(repo: Repo) -> None:
    doc = FIXTURE_DOCS[0]
    chunks = chunk_document(doc)
    await repo.replace_chunks(doc.id, chunks)
    await repo.replace_chunks(doc.id, chunks[:1])
    assert [c.id async for c in repo.iter_chunks(doc.id)] == [chunks[0].id]


async def test_drafts_judgments_runs_and_spend(repo: Repo) -> None:
    base = {"id": "d1", "owner_id": "alice", "title": "T", "audience": "grade8", "sections": []}
    await repo.save_draft({**base, "status": "running"})
    await repo.save_draft({**base, "status": "awaiting_review"})
    draft = await repo.get_draft("d1")
    assert draft is not None and draft["status"] == "awaiting_review" and draft["created_at"] <= draft["updated_at"]
    assert [d["id"] for d in await repo.list_drafts()] == ["d1"]
    assert await repo.get_draft("d1", "bob") is None
    assert await repo.list_drafts(owner_id="bob") == []
    assert await repo.claim_draft_review("d1", "alice")
    assert not await repo.claim_draft_review("d1", "alice")
    await repo.release_draft_review("d1", "alice")
    assert (await repo.get_draft("d1", "alice"))["status"] == "awaiting_review"  # type: ignore[index]
    await repo.record_judgment({"claim": "c", "label": "supported", "human_label": None})
    assert [j["claim"] async for j in repo.iter_judgments()] == ["c"]
    await repo.record_run({"kind": "ask"})
    assert await repo.add_spend(0.25) == pytest.approx(0.25)
    assert await repo.add_spend(0.5) == pytest.approx(0.75)
    assert await repo.get_spend() == pytest.approx(0.75)
