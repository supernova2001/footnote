"""The writer agent: a brief in, a cited article for a chosen audience out, with a human in the loop.

    START → research → outline → (Send: one write_section per section, in parallel) → assemble
                                         ↑                                                 │
                                         └──── level too hard/easy, or reviewer asks ──────┤
                                                                                           ↓
                                                                     review (interrupt) → publish → END

- research       the research agent (agents/research.py) picks passages through MCP tools; with a
                 document_id it searches that document only
- outline        title + 2-5 sections, each bound to the passages it may use
- write_section  per section: draft (Claude over `search_result` blocks) → atomic claims → judge →
                 redraft naming the failing claims, up to `section_retries`; then unsupported
                 sentences are dropped, and a section left with nothing is dropped with a warning
- assemble       orders sections, measures reading level (Flesch-Kincaid), and an editor model
                 rules whether it fits the audience; misfit → one more drafting round with the note
- review         `interrupt()`: the graph stops, the draft is saved as awaiting_review, and the run
                 resumes only when a person approves or sends notes (Command(resume=...)). The
                 checkpointer (MongoDB in production) makes the pause survive a restart.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

from footnote.agents.events import emit, step
from footnote.agents.grounding import blocks_to_segments, extract_claims, failing, judge_segments, publishable
from footnote.agents.judge import JudgePool
from footnote.corpus.chunking import split_sentences
from footnote.db.repo import Repo
from footnote.domain import Audience, Passage, Segment
from footnote.llm.client import AnswerDone, Models, search_result_block
from footnote.retrieval.retriever import Retriever

AUDIENCES: dict[str, dict[str, Any]] = {
    "grade4": {
        "label": "readers aged 9-10 (US grade 4)",
        "grade": 4.0,
        "style": "Short sentences, everyday words, explain any word a ten-year-old might not know. "
        "Describe injustice honestly but without graphic detail.",
    },
    "grade8": {
        "label": "readers aged 13-14 (US grade 8)",
        "grade": 8.0,
        "style": "Clear sentences, define specialist terms once, keep paragraphs short.",
    },
    "adult": {
        "label": "general adult readers",
        "grade": 11.0,
        "style": "Plain, engaging prose for a curious non-specialist.",
    },
}

OUTLINE_SYSTEM = """\
You plan a short, sourced article (2 to {max_sections} sections) for {audience}. You can use only \
the passages listed; each has an id. Give each section a heading, a one-sentence goal, and the ids \
of the passages it should draw on (2-{per_section} per section, only ids from the list). Prefer an \
order that tells a story: who, what they did or argued, why it mattered. Never invent a section \
the passages cannot support."""

SECTION_SYSTEM = """\
You write one section of an article for {audience}. {style}

- Use only the search results; cite them. Do not add facts from outside them.
- Write about the people in the sources in the third person. Never write as them.
- Attribute arguments ("Woodson argued..."); quote exactly and only from the search results.
- The older texts use the terminology of their time; write in present-day, respectful language.
- {words} words, in one to three short paragraphs. Do not write a heading."""

REDRAFT_NOTE = """\
A fact-checker found these statements in your draft are not supported by the search results:
{failures}
Rewrite the section so that every statement is supported by the search results. Remove what you \
cannot support; do not add anything new."""


class OutlineSection(BaseModel):
    heading: str
    goal: str
    passage_ids: list[str]


class Outline(BaseModel):
    title: str
    sections: list[OutlineSection]


class EditorVerdict(BaseModel):
    fit: Literal["fits", "too_hard", "too_easy"]
    note: str = Field(description="One or two sentences of concrete guidance for the writer")


class SectionResult(TypedDict):
    index: int
    round: int
    heading: str
    segments: list[dict[str, Any]]
    warnings: list[str]
    attempts: int
    status: str


class WriterState(TypedDict, total=False):
    draft_id: str
    owner_id: str
    brief: str
    audience: Audience
    scholar_id: str | None
    document_id: str | None
    sources: list[Passage]
    outline: Outline
    round: int
    sections: Annotated[list[SectionResult], operator.add]
    guidance: str | None
    readability: dict[str, Any] | None
    level_rounds: int
    review_notes: str | None
    decision: str | None
    status: str


class SectionTask(TypedDict):
    index: int
    round: int
    heading: str
    goal: str
    passages: list[Passage]
    audience: Audience
    brief: str
    guidance: str | None
    previous: list[dict[str, Any]] | None


FindSources = Callable[[str, str | None, str | None], Awaitable[list[Passage]]]


@dataclass
class WriterDeps:
    models: Models
    judges: JudgePool
    retriever: Retriever
    repo: Repo
    find_sources: FindSources  # the research agent


def build_writer_graph(deps: WriterDeps, checkpointer: Any) -> Any:
    limits = deps.models.config.limits.writer

    async def save(state: WriterState, **extra: Any) -> None:
        await deps.repo.save_draft(draft_document({**state, **extra}))

    async def research(state: WriterState) -> WriterState:
        step("research")
        if state.get("document_id"):
            passages = await deps.retriever.search(
                state["brief"], k=limits.max_sections * limits.passages_per_section, document_id=state["document_id"]
            )
        else:
            passages = await deps.find_sources(state["brief"], state.get("scholar_id"), state["draft_id"])
        emit("sources", {"passages": [p.model_dump() for p in passages]})
        await save(state, sources=passages, status="running")
        return {"sources": passages, "round": 0, "level_rounds": 0}

    async def outline(state: WriterState) -> WriterState:
        step("outline")
        sources = state["sources"]
        if not sources:
            raise ValueError("No sources found for this brief; nothing can be written from the collection.")
        listing = "\n\n".join(
            f"[{p.id}] {p.title}{' / ' + p.section if p.section else ''}\n{p.text[:600]}" for p in sources
        )
        aud = AUDIENCES[state["audience"]]
        out = await deps.models.structured(
            "writer.outline",
            system=OUTLINE_SYSTEM.format(
                max_sections=limits.max_sections, audience=aud["label"], per_section=limits.passages_per_section
            ),
            messages=[{"role": "user", "content": f"Brief: {state['brief']}\n\nPassages:\n{listing}"}],
            output=Outline,
            cache_system=False,
        )
        known = {p.id for p in sources}
        sections = [
            s.model_copy(
                update={"passage_ids": [i for i in s.passage_ids if i in known][: limits.passages_per_section]}
            )
            for s in out.sections[: limits.max_sections]
        ]
        sections = [s for s in sections if s.passage_ids]
        result = Outline(title=out.title, sections=sections)
        emit("outline", result.model_dump())
        return {"outline": result}

    def fan_out(state: WriterState) -> list[Send]:
        by_id = {p.id: p for p in state["sources"]}
        rnd = state.get("round", 0)
        previous = _latest_sections(state.get("sections", []), rnd - 1) if rnd else {}
        return [
            Send(
                "write_section",
                SectionTask(
                    index=i,
                    round=rnd,
                    heading=s.heading,
                    goal=s.goal,
                    passages=[by_id[x] for x in s.passage_ids],
                    audience=state["audience"],
                    brief=state["brief"],
                    guidance=" ".join(x for x in (state.get("guidance"), state.get("review_notes")) if x) or None,
                    previous=previous[i]["segments"] if i in previous else None,
                ),
            )
            for i, s in enumerate(state["outline"].sections)
        ]

    async def write_section(task: SectionTask) -> WriterState:
        aud = AUDIENCES[task["audience"]]
        words = "60-110" if task["audience"] == "grade4" else "110-200"
        system = SECTION_SYSTEM.format(audience=aud["label"], style=aud["style"], words=words)
        passages = task["passages"]
        content: list[dict[str, Any]] = [search_result_block(p.id, p.title, split_sentences(p.text)) for p in passages]
        ask = f"Article brief: {task['brief']}\nSection: {task['heading']}\nGoal: {task['goal']}"
        if task.get("guidance"):
            ask += f"\nEditor's guidance for this round: {task['guidance']}"
        if previous_segments := task.get("previous"):
            ask += "\nPrevious version of this section:\n" + "".join(s["text"] for s in previous_segments)
        content.append({"type": "text", "text": ask})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        segments: list[Segment] = []
        warnings: list[str] = []
        attempt = 0
        for attempt in range(1, limits.section_retries + 2):
            _section_event(task, attempt, "drafting", [])
            done: AnswerDone | None = None
            async for ev in deps.models.stream_answer("writer.section", system=system, messages=messages):
                if isinstance(ev, AnswerDone):
                    done = ev
            assert done is not None
            segments = blocks_to_segments(done.blocks, {p.id for p in passages})
            _section_event(task, attempt, "judging", segments)
            claims = await extract_claims(deps.models, "ask.claims", segments, f"{task['heading']}: {task['goal']}")
            pairs = await judge_segments(deps.judges, segments, claims.claims, passages)
            fails = failing(pairs)
            if not fails:
                break
            if attempt <= limits.section_retries:
                _section_event(task, attempt, "redrafting", segments)
                messages = [
                    *messages[:1],
                    {"role": "assistant", "content": "".join(s.text for s in segments)},
                    {
                        "role": "user",
                        "content": REDRAFT_NOTE.format(
                            failures="\n".join(f"- {c.text} ({r.verdict.rationale})" for c, r in fails)
                        ),
                    },
                ]
        dropped = [s for s in segments if not publishable(s)]
        kept = [s for s in segments if publishable(s)]
        if dropped:
            warnings.append(
                f"{len(dropped)} sentence(s) without full support and a citation removed after {attempt} attempt(s)."
            )
        status = "accepted" if any(s.verdict for s in kept) else "dropped"
        if status == "dropped":
            warnings.append("No supported prose was left for this section; it was dropped.")
            kept = []
        _section_event(task, attempt, status, kept)
        return {
            "sections": [
                SectionResult(
                    index=task["index"],
                    heading=task["heading"],
                    segments=[s.model_dump() for s in kept],
                    warnings=warnings,
                    attempts=attempt,
                    status=status,
                    round=task["round"],
                )
            ]
        }

    async def assemble(state: WriterState) -> WriterState:
        step("assemble")
        current = _latest_sections(state["sections"], state.get("round", 0))
        text = "\n\n".join("".join(s["text"] for s in sec["segments"]) for _, sec in sorted(current.items()))
        aud = AUDIENCES[state["audience"]]
        grade = flesch_kincaid_grade(text)
        verdict = await deps.models.structured(
            "writer.editor",
            system="You are an editor. Judge whether this article reads at the right level for its audience. "
            "The Flesch-Kincaid grade is one measurement, not the verdict: judge vocabulary, sentence "
            "length, and whether ideas are explained for this reader.",
            messages=[
                {
                    "role": "user",
                    "content": f"Audience: {aud['label']}\nTarget grade: {aud['grade']}\n"
                    f"Measured Flesch-Kincaid grade: {grade:.1f}\n\nArticle:\n{text}",
                }
            ],
            output=EditorVerdict,
        )
        readability = {
            "grade_level": round(grade, 1),
            "target": state["audience"],
            "editor_note": verdict.note,
            "fit": verdict.fit,
        }
        rounds = state.get("level_rounds", 0)
        retrying = verdict.fit != "fits" and rounds < limits.editor_retries
        emit("readability", {**readability, "retrying": retrying})
        if retrying:
            return {
                "readability": readability,
                "guidance": verdict.note,
                "level_rounds": rounds + 1,
                "round": state.get("round", 0) + 1,
                "review_notes": None,
            }
        return {"readability": readability, "guidance": None}

    async def review(state: WriterState) -> WriterState:
        await save(state, status="awaiting_review")
        decision = interrupt({"draft_id": state["draft_id"]})
        if decision.get("decision") == "approve":
            return {"decision": "approve", "review_notes": None}
        return {
            "decision": "revise",
            "review_notes": decision.get("notes") or "Improve clarity.",
            "round": state.get("round", 0) + 1,
            "level_rounds": limits.editor_retries,
        }

    async def publish(state: WriterState) -> WriterState:
        step("publish")
        await save(state, status="published")
        return {"status": "published"}

    def after_assemble(state: WriterState) -> list[Send] | Literal["review"]:
        return fan_out(state) if state.get("guidance") else "review"

    def after_review(state: WriterState) -> list[Send] | Literal["publish"]:
        return "publish" if state.get("decision") == "approve" else fan_out(state)

    g = StateGraph(WriterState)
    g.add_node("research", research)
    g.add_node("outline", outline)
    g.add_node("write_section", write_section)  # type: ignore[arg-type]
    g.add_node("assemble", assemble)
    g.add_node("review", review)
    g.add_node("publish", publish)
    g.add_edge(START, "research")
    g.add_edge("research", "outline")
    g.add_conditional_edges("outline", fan_out, ["write_section"])
    g.add_edge("write_section", "assemble")
    g.add_conditional_edges("assemble", after_assemble, ["write_section", "review"])
    g.add_conditional_edges("review", after_review, ["write_section", "publish"])
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer, name="writer")


def _section_event(task: SectionTask, attempt: int, status: str, segments: Sequence[Segment]) -> None:
    emit(
        "section",
        {
            "index": task["index"],
            "heading": task["heading"],
            "attempt": attempt,
            "status": status,
            "segments": [s.model_dump() for s in segments],
        },
    )


def _latest_sections(sections: Sequence[SectionResult], rnd: int) -> dict[int, SectionResult]:
    """The sections written in round `rnd` (each drafting round appends a full set)."""
    out: dict[int, SectionResult] = {}
    for s in sections:
        if s.get("round", 0) == rnd:
            out[s["index"]] = s
    return out


def draft_document(state: dict[str, Any]) -> dict[str, Any]:
    """The Draft shape in docs/api-contract.md, built from graph state."""
    outline: Outline | None = state.get("outline")
    current = _latest_sections(state.get("sections", []), state.get("round", 0))
    sources: list[Passage] = state.get("sources", [])
    return {
        "id": state["draft_id"],
        "owner_id": state.get("owner_id", "anonymous"),
        "title": outline.title if outline else None,
        "brief": state.get("brief", ""),
        "audience": state.get("audience", "adult"),
        "status": state.get("status", "running"),
        "sections": [
            {"heading": sec["heading"], "segments": sec["segments"], "warnings": sec["warnings"]}
            for _, sec in sorted(current.items())
            if sec["status"] != "dropped"
        ],
        "readability": {k: v for k, v in (state.get("readability") or {}).items() if k != "fit"} or None,
        "sources": [p.model_dump() for p in sources],
    }


# ── readability (a measurement: counting, not deciding) ─────────────────────────────────────

_VOWELS = re.compile(r"[aeiouy]+")


def _syllables(word: str) -> int:
    w = word.lower().strip(".,;:!?\"'()")
    if not w:
        return 0
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def flesch_kincaid_grade(text: str) -> float:
    sentences = split_sentences(text) or [text]
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
