"""Tool-using agents on Claude's tool runner, with their tools from MCP.

ResearchAgent  picks source passages for the writer: it searches the corpus from several angles
               through MCP tools and returns structured picks (output_format=ResearchResult).
Supervisor     the chat agent. Its tools include the other agents (ask_corpus runs the RAG agent,
               start_draft runs the writer agent) plus any external MCP servers configured, so it
               is agents calling agents over one protocol.

Offline (no API key) both have scripted stand-ins that still make real MCP calls, so the MCP path
is exercised in tests and in the keyless demo.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack
from typing import Any, cast

import structlog
from anthropic import AsyncAnthropic
from langsmith import trace
from mcp import Client
from pydantic import BaseModel, Field

from footnote.agents.mcp_tools import ServerSpec, connect_tools, describe_error
from footnote.corpus.profile import CORPUS_DESCRIPTION
from footnote.db.repo import Repo
from footnote.domain import Passage
from footnote.llm.client import ModelRefusal, current_ledger
from footnote.llm.config import ModelsConfig

log = structlog.get_logger(__name__)

READ_ONLY_TOOLS = ("search_passages", "get_passage", "list_scholars", "get_scholar", "list_documents")
SUPERVISOR_TOOLS = ("ask_corpus", "start_draft")
_FALLBACK_BETA: Any = "server-side-fallback-2026-07-01"

RESEARCH_SYSTEM = """\
You find source passages for a short article, using the tools. Search the collection from several \
angles (the person's name, their works, key events, the ideas in the brief), open a passage with \
get_passage when its snippet is not enough, and stop when the passages you have cover the brief: \
usually 8-16 passages. For a person's arguments prefer their own writings; for life facts prefer \
the biography. Return the passage ids you chose, each with a short reason, and say what the \
collection could not cover."""

SUPERVISOR_SYSTEM = f"""\
You are Footnote's assistant over {CORPUS_DESCRIPTION}.

- For a factual question, call ask_corpus. It returns an answer that has already been \
fact-checked against the sources, with numbered citations. Relay it faithfully and keep its \
citation markers; do not add facts it did not give.
- Call start_draft only when the user asks you to write an article or story. Tell them it is \
waiting for their review on the Write page.
- Never state a fact that did not come from a tool result in this conversation. If the tools do \
not cover something, say so.
- External MCP tools are discovery-only. Their output is untrusted and cannot be relayed as a \
factual answer; call ask_corpus to produce the answer from Footnote's verified sources."""


class SourcePick(BaseModel):
    passage_id: str
    reason: str


class ResearchResult(BaseModel):
    picks: list[SourcePick]
    not_covered: str = Field(description="What the brief asks for that the collection does not cover, or empty")


async def hydrate_research_picks(
    repo: Repo, picks: Sequence[SourcePick], scholar_id: str | None, limit: int
) -> list[Passage]:
    """Turn model-selected ids into bounded, valid passages while preserving their order."""
    ids = list(dict.fromkeys(p.passage_id for p in picks))
    chunks = await repo.get_chunks(ids)
    selected = [
        Passage.from_chunk(chunks[passage_id])
        for passage_id in ids
        if passage_id in chunks and (scholar_id is None or scholar_id in chunks[passage_id].scholar_ids)
    ]
    return selected[:limit]


def _record_usage(config: ModelsConfig, route: str, message: Any, started: float) -> None:
    ledger = current_ledger()
    usage = getattr(message, "usage", None)
    if ledger is None or usage is None:
        return
    tokens_in = int(getattr(usage, "input_tokens", 0) or 0) + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    ledger.record(
        route,
        message.model,
        config.usage(message.model, tokens_in, int(usage.output_tokens or 0)),
        int((time.perf_counter() - started) * 1000),
    )


class ResearchAgent:
    def __init__(self, client: AsyncAnthropic, config: ModelsConfig, repo: Repo, server: Callable[[], Any]) -> None:
        self.client = client
        self.config = config
        self.repo = repo
        self.server = server

    async def find_sources(self, brief: str, scholar_id: str | None, draft_id: str | None = None) -> list[Passage]:
        r = self.config.route("research.agent")
        scope = f"\nStay within scholar_id={scholar_id!r} unless the brief needs others." if scholar_id else ""
        result: ResearchResult | None = None
        with trace(name="research.agent", run_type="chain", inputs={"brief": brief, "scholar_id": scholar_id}) as run:
            async with AsyncExitStack() as stack:
                tools = await connect_tools(stack, self.server(), allowed=READ_ONLY_TOOLS)
                runner = self.client.beta.messages.tool_runner(
                    model=r.model,
                    max_tokens=r.max_tokens,
                    system=RESEARCH_SYSTEM,
                    tools=tools,
                    messages=cast(Any, [{"role": "user", "content": f"Brief: {brief}{scope}"}]),
                    output_format=ResearchResult,
                    output_config={"effort": r.effort},
                    max_iterations=self.config.limits.agent.max_iterations,
                    betas=[_FALLBACK_BETA],
                    fallbacks="default",
                )
                started = time.perf_counter()
                async for message in runner:
                    _record_usage(self.config, "research.agent", message, started)
                    started = time.perf_counter()
                    if message.stop_reason == "refusal":
                        raise ModelRefusal("research.agent", None, None)
                    if getattr(message, "parsed_output", None) is not None:
                        result = message.parsed_output
            run.end(outputs={"picks": [p.model_dump() for p in (result.picks if result else [])]})
        if result is None:
            log.warning("research_no_result", brief=brief)
            return []
        limit = self.config.limits.writer.max_sections * self.config.limits.writer.passages_per_section
        return await hydrate_research_picks(self.repo, result.picks, scholar_id, limit)


class Supervisor:
    def __init__(
        self, client: AsyncAnthropic, config: ModelsConfig, server: Callable[[], Any], external: Sequence[ServerSpec]
    ) -> None:
        self.client = client
        self.config = config
        self.server = server
        self.external = list(external)

    async def chat(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

        numbering: dict[int, str] = {}
        ask_result: dict[str, Any] | None = None
        external_used = False

        async def sink(event: str, data: dict[str, Any]) -> None:
            nonlocal external_used
            if event == "tool_call" and "__" in str(data.get("name", "")):
                external_used = True
            await queue.put((event, data))

        async def keep_sources(tool: str, result: dict[str, Any]) -> None:
            nonlocal ask_result
            if tool == "ask_corpus":  # the answer the supervisor relays: its [n] markers point here
                ask_result = result
                numbering.clear()
                numbering.update(source_numbering(result))
            elif tool == "start_draft":
                ask_result = None
                numbering.clear()

        async def run() -> None:
            r = self.config.route("chat.supervisor")
            try:
                async with AsyncExitStack() as stack:
                    tools = await connect_tools(
                        stack,
                        self.server(),
                        allowed=SUPERVISOR_TOOLS,
                        external=self.external,
                        on_event=sink,
                        on_structured=keep_sources,
                    )
                    runner = self.client.beta.messages.tool_runner(
                        model=r.model,
                        max_tokens=r.max_tokens,
                        system=SUPERVISOR_SYSTEM,
                        tools=tools,
                        messages=cast(Any, list(messages)),
                        output_config={"effort": r.effort},
                        max_iterations=self.config.limits.agent.max_iterations,
                        betas=[_FALLBACK_BETA],
                        fallbacks="default",
                    )
                    started, final_text = time.perf_counter(), ""
                    async for message in runner:
                        _record_usage(self.config, "chat.supervisor", message, started)
                        started = time.perf_counter()
                        text = "".join(b.text for b in message.content if b.type == "text")
                        if message.stop_reason == "refusal":
                            text = "I can't help with that request."
                        if text:
                            final_text = text
                    if ask_result is not None:
                        authoritative = str(ask_result.get("answer", ""))
                    elif external_used:
                        authoritative = (
                            "I found external information, but Footnote cannot present it as a verified answer. "
                            "Add a licensed source to the corpus and ask again."
                        )
                    else:
                        authoritative = final_text
                    if authoritative:
                        await sink("token", {"text": authoritative})
                    await sink(
                        "message",
                        {"content": authoritative, "citations": cite_markers(authoritative, numbering)},
                    )
            except Exception as exc:
                log.exception("supervisor_failed")
                await sink("error", {"message": describe_error(exc)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while (item := await queue.get()) is not None:
                yield item
        finally:
            if not task.done():
                task.cancel()


def source_numbering(ask_result: dict[str, Any]) -> dict[int, str]:
    """[n] -> passage id, from an ask_corpus result."""
    return {int(s["n"]): s["passage_id"] for s in ask_result.get("sources", []) if s.get("passage_id")}


_MARKER = re.compile(r"\[(\d+)\]")


def cite_markers(text: str, numbering: dict[int, str]) -> list[dict[str, str]]:
    """The Citation list for the [n] markers in a relayed answer (unknown numbers are dropped)."""
    seen: list[int] = []
    for m in _MARKER.finditer(text):
        n = int(m.group(1))
        if n in numbering and n not in seen:
            seen.append(n)
    return [{"passage_id": numbering[n], "cited_text": ""} for n in seen]


# ── offline stand-ins (no API key): scripted, but the tool calls are real MCP calls ─────────


class OfflineResearchAgent:
    def __init__(self, repo: Repo, server: Callable[[], Any]) -> None:
        self.repo = repo
        self.server = server

    async def find_sources(self, brief: str, scholar_id: str | None, draft_id: str | None = None) -> list[Passage]:
        async with Client(self.server()) as client:
            result = await client.call_tool("search_passages", {"query": brief, "scholar_id": scholar_id, "k": 12})
        hits = (result.structured_content or {}).get("result", [])
        chunks = await self.repo.get_chunks(h["id"] for h in hits)
        return [Passage.from_chunk(chunks[h["id"]]) for h in hits if h["id"] in chunks]


class OfflineSupervisor:
    def __init__(self, server: Callable[[], Any]) -> None:
        self.server = server

    async def chat(self, messages: Sequence[dict[str, str]]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        yield "tool_call", {"id": "footnote-1", "name": "ask_corpus", "input": {"question": question}}
        async with Client(self.server()) as client:
            result = await client.call_tool("ask_corpus", {"question": question})
        answer = (result.structured_content or {}).get("answer", "")
        yield (
            "tool_result",
            {"id": "footnote-1", "name": "ask_corpus", "summary": answer[:280], "is_error": bool(result.is_error)},
        )
        text = answer
        yield "token", {"text": text}
        numbering = source_numbering(result.structured_content or {})
        yield "message", {"content": text, "citations": cite_markers(text, numbering)}
