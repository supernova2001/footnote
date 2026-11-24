"""The RAG agent: a question in, a judged and cited answer out.

    START → plan ─┬─(out of scope)→ decline → END
                  └→ retrieve → answer → verify ─┬─(claims fail, budget left)→ revise → answer
                                                 └→ finalize → END

- plan      one model call decides whether the question is in scope and writes 1-3 search queries
            (multi-part and comparison questions get one query per part)
- retrieve  hybrid search + rerank over every query (retrieval/retriever.py)
- answer    Claude answers over `search_result` blocks, so every citation is a passage id
- verify    atomic claims → grounding judge (agents/grounding.py)
- revise    the failing claims are named and the answer is rewritten, at most `max_revisions` times
- finalize  anything not fully supported and explicitly cited is returned in `removed`

The loop is the point: retrieval quality bounds what can be said, and the judge is what stops the
answer from saying more than that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from footnote.agents.events import emit, step
from footnote.agents.grounding import blocks_to_segments, extract_claims, failing, judge_segments, publishable
from footnote.agents.judge import JudgePool
from footnote.corpus.chunking import split_sentences
from footnote.corpus.profile import CORPUS_DESCRIPTION
from footnote.db.repo import Repo
from footnote.domain import Passage, Segment, Strategy
from footnote.llm.client import AnswerDone, Models, TextDelta, search_result_block
from footnote.retrieval.retriever import Retriever

PLAN_SYSTEM = f"""\
You are the planner for Footnote, a research assistant over {CORPUS_DESCRIPTION}.

Decide two things.
1. in_scope: true when the collection may contain sources relevant to the question. Do not assume \
a named person or subject is present: retrieval decides that. False for unrelated tasks (coding, \
personal advice, current events the collection cannot cover) and harmful requests.
2. queries: 1-3 search queries that together retrieve what an answer needs. Use names and key \
terms, not pronouns. Give a multi-part or comparison question one query per part.

When a conversation comes before the new question, first rewrite the new question so it stands on \
its own: after a question about Wells and the press, "what about Woodson?" becomes "What did Carter \
G. Woodson write about the press?". Then plan for the rewritten question."""

ANSWER_SYSTEM = f"""\
You answer questions using only the search results provided from {CORPUS_DESCRIPTION}.

- Base every factual statement on the search results and cite them; do not use outside knowledge.
- If the results do not answer the question, say so plainly in one or two sentences. Do not guess.
- Attribute arguments to their authors ("Washington argued...", "Du Bois wrote..."): these texts \
take positions, and some reflect the views of their era.
- The older texts use the terminology of their time. Quote it accurately when you quote, but write \
in present-day, respectful language yourself.
- Plain prose, one to three short paragraphs. No headings; no lists unless asked."""

REVISION_NOTE = """\
A fact-checker found that these statements in your previous answer are not supported by the \
search results:
{failures}

Write the answer again. Remove or correct those statements so that everything you say is \
supported by the search results, and add nothing that the results do not say."""


class AskPlan(BaseModel):
    standalone_question: str = Field(
        description="The new question rewritten to stand on its own, with pronouns and references from the "
        "conversation spelled out; the question unchanged if it already stands alone"
    )
    in_scope: bool
    reason: str = Field(description="One sentence; shown to the reader when the question is out of scope")
    queries: list[str] = Field(description="1-3 search queries")


class AskState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    standalone_question: str
    scholar_id: str | None
    scholar_name: str | None
    strategy: Strategy
    in_scope: bool
    scope_reason: str
    queries: list[str]
    passages: list[Passage]
    segments: list[Segment]
    removed: list[Segment]
    revision: int
    revision_note: str | None
    abstained: bool
    claims_checked: int


@dataclass
class AskDeps:
    models: Models
    retriever: Retriever
    judges: JudgePool
    repo: Repo


def build_ask_graph(deps: AskDeps) -> Any:
    limits = deps.models.config.limits.ask

    async def plan(state: AskState) -> AskState:
        step("plan")
        scholar = f"\n(The reader is on {state['scholar_name']}'s page.)" if state.get("scholar_name") else ""
        prompt = render_conversation(state.get("history", []), state["question"]) + scholar
        out = await deps.models.structured(
            "ask.plan", system=PLAN_SYSTEM, messages=[{"role": "user", "content": prompt}], output=AskPlan
        )
        standalone = out.standalone_question.strip() or state["question"]
        queries = [q.strip() for q in out.queries if q.strip()][: limits.max_subqueries] or [standalone]
        return {
            "in_scope": out.in_scope,
            "scope_reason": out.reason,
            "queries": queries,
            "standalone_question": standalone,
            "revision": 0,
        }

    async def decline(state: AskState) -> AskState:
        text = state.get("scope_reason") or "That is outside what this collection covers."
        return {"segments": [Segment(text=text)], "removed": [], "abstained": True}

    async def retrieve(state: AskState) -> AskState:
        step("retrieve", detail=state.get("strategy", "hybrid_rerank"))
        passages = await deps.retriever.search_many(
            state["queries"],
            strategy=state.get("strategy", "hybrid_rerank"),
            k=limits.rerank_top_n,
            candidates=limits.retrieve_k,
            scholar_id=state.get("scholar_id"),
        )
        emit(
            "passages",
            {"question": _question(state), "queries": state["queries"], "passages": [p.model_dump() for p in passages]},
        )
        return {"passages": passages}

    async def answer(state: AskState) -> AskState:
        step("answer", detail=f"revision {state.get('revision', 0)}" if state.get("revision") else None)
        passages = state.get("passages", [])
        if not passages:
            return {
                "segments": [Segment(text="The sources in this collection do not cover that question.")],
                "abstained": True,
            }
        content: list[dict[str, Any]] = [
            search_result_block(p.id, _title(p), split_sentences(p.text)) for p in passages
        ]
        content.append({"type": "text", "text": f"Question: {_question(state)}"})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if state.get("revision_note"):
            previous = "".join(s.text for s in state.get("segments", []))
            messages += [
                {"role": "assistant", "content": previous},
                {"role": "user", "content": state["revision_note"]},
            ]
        emit("token", {"text": "", "reset": True})
        done: AnswerDone | None = None
        async for ev in deps.models.stream_answer("ask.answer", system=ANSWER_SYSTEM, messages=messages):
            if isinstance(ev, TextDelta):
                emit("token", {"text": ev.text})
            else:
                done = ev
        assert done is not None
        return {"segments": blocks_to_segments(done.blocks, {p.id for p in passages})}

    async def verify(state: AskState) -> AskState:
        step("verify")
        segments = [s.model_copy(deep=True) for s in state["segments"]]
        claim_set = await extract_claims(deps.models, "ask.claims", segments, _question(state))
        if claim_set.abstained and not claim_set.claims:
            for s in segments:
                s.verdict = None
            return {"segments": segments, "abstained": True, "revision_note": None, "claims_checked": 0}
        pairs = await judge_segments(deps.judges, segments, claim_set.claims, state["passages"])
        fails = failing(pairs)
        note = None
        if fails and state.get("revision", 0) < limits.max_revisions:
            note = REVISION_NOTE.format(
                failures="\n".join(f"- {c.text} ({r.verdict.label}: {r.verdict.rationale})" for c, r in fails)
            )
        emit("verdicts", {"checked": len(pairs), "failing": len(fails)})
        return {"segments": segments, "abstained": False, "revision_note": note, "claims_checked": len(pairs)}

    async def revise(state: AskState) -> AskState:
        step("revise")
        return {"revision": state.get("revision", 0) + 1}

    async def finalize(state: AskState) -> AskState:
        step("finalize")
        kept = [s for s in state["segments"] if publishable(s)]
        removed = [s for s in state["segments"] if not publishable(s)]
        abstained = state.get("abstained", False) or not any(s.verdict for s in kept)
        if not kept:
            kept = [Segment(text="The sources in this collection do not support an answer to that question.")]
        return {"segments": kept, "removed": removed, "abstained": abstained}

    def after_plan(state: AskState) -> Literal["retrieve", "decline"]:
        return "retrieve" if state.get("in_scope") else "decline"

    def after_verify(state: AskState) -> Literal["revise", "finalize"]:
        return "revise" if state.get("revision_note") else "finalize"

    g = StateGraph(AskState)
    g.add_node("plan", plan)
    g.add_node("decline", decline)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_node("verify", verify)
    g.add_node("revise", revise)
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan)
    g.add_edge("decline", END)
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "verify")
    g.add_conditional_edges("verify", after_verify)
    g.add_edge("revise", "answer")
    g.add_edge("finalize", END)
    return g.compile(name="ask")


def _title(p: Passage) -> str:
    return f"{p.title} — {p.section}" if p.section else p.title


def answer_payload(state: AskState) -> dict[str, Any]:
    """The `answer` SSE event (docs/api-contract.md)."""
    return {
        "segments": [s.model_dump() for s in state.get("segments", [])],
        "removed": [s.model_dump() for s in state.get("removed", [])],
        "revised": state.get("revision", 0) > 0,
        "abstained": state.get("abstained", False),
    }


MAX_HISTORY_MESSAGES = 8  # four exchanges: enough to resolve "what about him?", little enough to stay cheap
HISTORY_CHARS = 1200


def initial_state(
    question: str,
    strategy: Strategy,
    scholar_id: str | None,
    scholar_name: str | None,
    history: Sequence[dict[str, str]] = (),
) -> AskState:
    return {
        "question": question,
        "strategy": strategy,
        "scholar_id": scholar_id,
        "scholar_name": scholar_name,
        "history": list(history)[-MAX_HISTORY_MESSAGES:],
    }


def render_conversation(history: Sequence[dict[str, str]], question: str) -> str:
    """The planner's input: earlier turns (trimmed) and the new question.

    Only the planner sees the conversation. Retrieval and the answer work from the standalone
    question it writes, so an earlier answer can never be cited as if it were a source.
    """
    if not history:
        return f"Question: {question}"
    lines = []
    for m in history[-MAX_HISTORY_MESSAGES:]:
        who = "User" if m.get("role") == "user" else "Assistant"
        text = " ".join(str(m.get("content", "")).split())
        lines.append(f"{who}: {text[:HISTORY_CHARS]}{'…' if len(text) > HISTORY_CHARS else ''}")
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nNew question: {question}"


def _question(state: AskState) -> str:
    return state.get("standalone_question") or state["question"]


def cited_passages(segments: Sequence[Segment], passages: Sequence[Passage]) -> list[Passage]:
    used = {c.passage_id for s in segments for c in s.citations}
    return [p for p in passages if p.id in used]
