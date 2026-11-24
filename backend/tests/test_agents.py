from __future__ import annotations

from typing import Any

from mcp import Client

from footnote.agents.ask import AskDeps, build_ask_graph
from footnote.agents.grounding import extract_claims, publishable
from footnote.agents.judge import ApiJudge, JudgeOutput, JudgePool, RecordingJudge
from footnote.agents.tool_agents import SourcePick, hydrate_research_picks
from footnote.domain import Citation, Segment, Verdict
from footnote.llm.fake import FakeModels
from footnote.runs import collect, run_ask, run_chat, run_draft_review, run_draft_start
from footnote.services import Services


async def test_ask_answers_with_citations_and_verdicts(services: Services) -> None:
    out = await collect(
        run_ask(
            services, question="When did Woodson launch Negro History Week?", scholar_id=None, strategy="hybrid_rerank"
        )
    )
    answer = out["answer"]
    assert not answer["abstained"]
    cited = [s for s in answer["segments"] if s["citations"]]
    assert cited, answer
    known = {p["id"] for p in out["passages"]["passages"]}
    assert all(c["passage_id"] in known for s in cited for c in s["citations"])
    assert all(s["verdict"] for s in cited)
    assert out["done"]["usage"]["cost_usd"] == 0.0
    assert services.repo.runs[-1]["kind"] == "ask"  # type: ignore[attr-defined]
    assert services.repo.runs[-1]["status"] == "completed"  # type: ignore[attr-defined]


async def test_out_of_scope_question_is_declined(services: Services) -> None:
    out = await collect(
        run_ask(services, question="Write me a Python script to sort a list", scholar_id=None, strategy="hybrid_rerank")
    )
    assert out["answer"]["abstained"] and "passages" not in out


async def test_failing_claims_trigger_one_revision_then_are_withheld(services: Services) -> None:
    """A judge that rejects everything: the agent revises once (the budget), then withholds."""
    reject = lambda s, m, o: JudgeOutput(label="unsupported", evidence="", rationale="not in passages")  # noqa: E731
    models = FakeModels(services.config, {"judge.grounding": reject})  # type: ignore[dict-item]
    judges = JudgePool(RecordingJudge(ApiJudge(models), services.repo), 4)
    graph = build_ask_graph(AskDeps(models=models, retriever=services.retriever, judges=judges, repo=services.repo))
    services.ask_graph = graph
    out = await collect(
        run_ask(services, question="What did Wells publish in 1892?", scholar_id=None, strategy="hybrid_rerank")
    )
    answer = out["answer"]
    assert answer["revised"] is True
    assert answer["removed"] and all(s["verdict"]["label"] == "unsupported" for s in answer["removed"])
    assert all(s["verdict"] is None or s["verdict"]["label"] != "unsupported" for s in answer["segments"])
    assert [r for r, _ in models.calls].count("ask.answer") == 2


async def test_partial_claims_are_withheld_after_revision_budget(services: Services) -> None:
    partial = lambda s, m, o: JudgeOutput(label="partial", evidence="1926", rationale="adds detail")  # noqa: E731
    models = FakeModels(services.config, {"judge.grounding": partial})  # type: ignore[dict-item]
    judges = JudgePool(RecordingJudge(ApiJudge(models), services.repo), 4)
    services.ask_graph = build_ask_graph(
        AskDeps(models=models, retriever=services.retriever, judges=judges, repo=services.repo)
    )
    out = await collect(
        run_ask(services, question="When did Woodson launch Negro History Week?", scholar_id=None, strategy="hybrid")
    )
    answer = out["answer"]
    assert answer["removed"]
    assert all(s["verdict"]["label"] in {"partial", "unsupported"} for s in answer["removed"])
    assert all(s["verdict"] is None for s in answer["segments"])


async def test_claim_extraction_fails_closed_when_model_omits_a_segment(services: Services) -> None:
    def omit(system: str, messages: Any, output: Any) -> Any:
        return output.model_validate({"abstained": False, "claims": []})

    models = FakeModels(services.config, {"ask.claims": omit})  # type: ignore[dict-item]
    claims = await extract_claims(
        models,
        "ask.claims",
        [Segment(text="The document says this happened in 1926.")],
        "When did it happen?",
    )
    assert [(c.segment, c.text) for c in claims.claims] == [(0, "The document says this happened in 1926.")]


def test_publication_requires_full_support_and_a_citation() -> None:
    citation = [Citation(passage_id="doc:0001", cited_text="evidence")]
    supported = Verdict(label="supported", evidence="evidence", rationale="exact", judge="test")
    partial = Verdict(label="partial", evidence="evidence", rationale="incomplete", judge="test")
    assert publishable(Segment(text="Framing only."))
    assert publishable(Segment(text="Supported.", citations=citation, verdict=supported))
    assert not publishable(Segment(text="No source.", verdict=supported))
    assert not publishable(Segment(text="Only partly.", citations=citation, verdict=partial))


async def test_research_picks_are_deduplicated_scoped_and_capped(services: Services) -> None:
    chunks = [chunk async for chunk in services.repo.iter_chunks()]
    woodson = next(chunk for chunk in chunks if "carter-g-woodson" in chunk.scholar_ids)
    other = next(chunk for chunk in chunks if "carter-g-woodson" not in chunk.scholar_ids)
    picks = [
        SourcePick(passage_id=woodson.id, reason="first"),
        SourcePick(passage_id=woodson.id, reason="duplicate"),
        SourcePick(passage_id=other.id, reason="out of scope"),
        SourcePick(passage_id="missing:0000", reason="invented"),
    ]
    passages = await hydrate_research_picks(services.repo, picks, "carter-g-woodson", limit=1)
    assert [passage.id for passage in passages] == [woodson.id]


async def test_writer_pauses_for_review_then_publishes(services: Services) -> None:
    started = await collect(
        run_draft_start(
            services,
            brief="Carter G. Woodson and Negro History Week",
            audience="grade8",
            scholar_id="carter-g-woodson",
            document_id=None,
        )
    )
    assert "error" not in started, started
    draft_id = started["draft"]["draft_id"]
    review = started["review"]["draft"]
    assert review["status"] == "awaiting_review" and review["sections"] and review["sources"]
    cited = [s for sec in review["sections"] for s in sec["segments"] if s["citations"]]
    assert cited and all(s["verdict"] for s in cited)

    revised = await collect(run_draft_review(services, draft_id=draft_id, decision="revise", notes="Shorter, please."))
    assert revised["review"]["draft"]["status"] == "awaiting_review"

    done = await collect(run_draft_review(services, draft_id=draft_id, decision="approve", notes=None))
    assert done["published"]["draft"]["status"] == "published"
    assert (await services.repo.get_draft(draft_id))["status"] == "published"  # type: ignore[index]
    recorded = [(r["kind"], r["status"]) for r in services.repo.runs]  # type: ignore[attr-defined]
    assert ("draft_start", "interrupted") in recorded
    assert recorded.count(("draft_review", "interrupted")) == 1
    assert recorded.count(("draft_review", "completed")) == 1


async def test_mcp_server_exposes_the_corpus_and_the_agents(services: Services) -> None:
    async with Client(services.mcp_server()) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"search_passages", "get_passage", "list_scholars", "get_scholar", "ask_corpus", "start_draft"} <= names
        hits = (await client.call_tool("search_passages", {"query": "Negro History Week", "k": 2})).structured_content
        first = hits["result"][0]  # type: ignore[index]
        passage = (await client.call_tool("get_passage", {"passage_id": first["id"]})).structured_content
        assert passage["id"] == first["id"] and "next_id" in passage  # type: ignore[index]
        answer = (
            await client.call_tool("ask_corpus", {"question": "When did Woodson start his journal?"})
        ).structured_content
        assert "[1]" in answer["answer"] and answer["sources"][0]["passage_id"]  # type: ignore[index]
        resource = await client.read_resource("footnote://scholars/carter-g-woodson")
        assert "Carter G. Woodson" in str(resource)
        missing = await client.call_tool("get_passage", {"passage_id": "nope:0000"})
        assert missing.is_error


async def test_offline_supervisor_calls_the_rag_agent_over_mcp(services: Services) -> None:
    events: list[tuple[str, dict[str, Any]]] = [
        e async for e in run_chat(services, [{"role": "user", "content": "Who was Ida B. Wells?"}])
    ]
    kinds = [e for e, _ in events]
    assert kinds[0] == "tool_call" and "tool_result" in kinds and kinds[-1] == "done"
    message = next(d for e, d in events if e == "message")
    assert message["content"]
    assert not message["content"].startswith("(Offline mode")
    if "[1]" in message["content"]:
        assert message["citations"] and message["citations"][0]["passage_id"]
    assert services.repo.runs[-1]["kind"] == "chat"  # type: ignore[attr-defined]
    assert services.repo.runs[-1]["status"] == "completed"  # type: ignore[attr-defined]


async def test_closed_chat_stream_records_cancellation(services: Services) -> None:
    stream = run_chat(services, [{"role": "user", "content": "Who was Ida B. Wells?"}])
    event, _ = await anext(stream)
    assert event == "tool_call"
    await stream.aclose()
    assert services.repo.runs[-1]["kind"] == "chat"  # type: ignore[attr-defined]
    assert services.repo.runs[-1]["status"] == "cancelled"  # type: ignore[attr-defined]


def test_chat_markers_map_to_the_relayed_sources() -> None:
    from footnote.agents.tool_agents import cite_markers, source_numbering

    numbering = source_numbering({"sources": [{"n": 1, "passage_id": "a:0001"}, {"n": 2, "passage_id": "b:0002"}]})
    cites = cite_markers("Woodson did X [2]. He also did Y [1][2]. Unknown [9].", numbering)
    assert [c["passage_id"] for c in cites] == ["b:0002", "a:0001"]


async def test_follow_ups_are_planned_from_the_conversation(services: Services) -> None:
    seen: list[str] = []

    def plan(system: str, messages: Any, output: Any) -> Any:
        seen.append(messages[0]["content"])
        return output.model_validate(
            {
                "standalone_question": "When did Carter G. Woodson start Negro History Week?",
                "in_scope": True,
                "reason": "",
                "queries": ["Woodson Negro History Week"],
            }
        )

    models = FakeModels(services.config, {"ask.plan": plan})  # type: ignore[dict-item]
    judges = JudgePool(RecordingJudge(ApiJudge(models), services.repo), 4)
    services.ask_graph = build_ask_graph(
        AskDeps(models=models, retriever=services.retriever, judges=judges, repo=services.repo)
    )
    history = [
        {"role": "user", "content": "Who was Carter G. Woodson?"},
        {"role": "assistant", "content": "A historian who founded a journal."},
    ]
    out = await collect(
        run_ask(
            services, question="When did he start the week?", scholar_id=None, strategy="hybrid_rerank", history=history
        )
    )
    assert "Conversation so far:" in seen[0] and "New question: When did he start the week?" in seen[0]
    assert out["passages"]["question"] == "When did Carter G. Woodson start Negro History Week?"
    assert any("1926" in p["text"] for p in out["passages"]["passages"])
