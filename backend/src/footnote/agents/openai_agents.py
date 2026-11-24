"""The research agent and the supervisor on OpenAI, over the same MCP tools as the Claude versions.

Claude's agents use the Anthropic SDK's tool runner; here the loop is written out on the Responses
API: call the model with the MCP tools as function tools, run every function call it asks for
(concurrently), send the outputs back with `previous_response_id`, and stop when it answers without
calling a tool or the iteration budget is spent. The research agent then asks for its final answer
as a structured `ResearchResult`.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from typing import Any, cast

import structlog
from langsmith import trace
from openai import AsyncOpenAI
from pydantic import BaseModel

from footnote.agents.mcp_tools import ObservedSession, ServerSpec, describe_error, open_tools, result_text
from footnote.agents.tool_agents import (
    READ_ONLY_TOOLS,
    RESEARCH_SYSTEM,
    SUPERVISOR_SYSTEM,
    SUPERVISOR_TOOLS,
    ResearchResult,
    cite_markers,
    hydrate_research_picks,
    source_numbering,
)
from footnote.db.repo import Repo
from footnote.domain import Passage
from footnote.llm.config import ModelsConfig
from footnote.llm.openai_client import OpenAIModels, _check

log = structlog.get_logger(__name__)
TextSink = Callable[[str], Awaitable[None]]


def function_tools(pairs: Sequence[tuple[Any, ObservedSession]]) -> list[dict[str, Any]]:
    tools = []
    for t, _ in pairs:
        schema = dict(getattr(t, "input_schema", None) or {"type": "object", "properties": {}})
        schema.setdefault("type", "object")
        tools.append(
            {
                "type": "function",
                "name": t.name,
                "description": t.description or "",
                "parameters": schema,
                "strict": False,
            }
        )
    return tools


async def run_tool_loop[T: BaseModel](
    client: AsyncOpenAI,
    config: ModelsConfig,
    route: str,
    *,
    system: str,
    messages: Sequence[dict[str, Any]],
    pairs: Sequence[tuple[Any, ObservedSession]],
    on_text: TextSink | None = None,
    output: type[T] | None = None,
) -> tuple[str, T | None]:
    r = config.route(route)
    meter = OpenAIModels(config, client)
    tools = function_tools(pairs)
    sessions = {t.name: s for t, s in pairs}
    common: dict[str, Any] = {
        "model": r.model,
        "instructions": system,
        "reasoning": {"effort": r.effort},
        "max_output_tokens": r.max_tokens,
    }

    async def run_call(call: Any) -> dict[str, Any]:
        session = sessions.get(call.name)
        if session is None:
            text = f"Error: no tool named {call.name!r}"
        else:
            try:
                result = await session.call_tool(call.name, json.loads(call.arguments or "{}"))
                text = ("Error: " if result.is_error else "") + result_text(result)
            except Exception as exc:  # the model sees the failure and can try something else
                text = f"Error: {type(exc).__name__}: {exc}"
        return {"type": "function_call_output", "call_id": call.call_id, "output": text}

    final_text = ""
    with trace(
        name=route, run_type="chain", inputs={"messages": list(messages)}, metadata={"provider": "openai"}
    ) as run:
        started = time.perf_counter()
        resp = await client.responses.create(**common, input=cast(Any, list(messages)), tools=cast(Any, tools))
        meter._record(route, resp, started)
        for _ in range(config.limits.agent.max_iterations):
            _check(route, resp)
            if resp.output_text:
                final_text = resp.output_text
                if on_text is not None:
                    await on_text(final_text)
            calls = [o for o in resp.output if o.type == "function_call"]
            if not calls:
                break
            outputs = await asyncio.gather(*(run_call(c) for c in calls))
            started = time.perf_counter()
            resp = await client.responses.create(
                **common, previous_response_id=resp.id, input=cast(Any, outputs), tools=cast(Any, tools)
            )
            meter._record(route, resp, started)
        parsed: T | None = None
        if output is not None:
            started = time.perf_counter()
            done = await client.responses.parse(
                **common,
                previous_response_id=resp.id,
                text_format=output,
                input=[{"role": "user", "content": "Give your final answer now, in the required format."}],
            )
            meter._record(route, done, started)
            _check(route, done)
            parsed = done.output_parsed
        run.end(outputs={"text": final_text, "parsed": parsed.model_dump() if parsed else None})
    return final_text, parsed


class OpenAIResearchAgent:
    def __init__(self, client: AsyncOpenAI, config: ModelsConfig, repo: Repo, server: Callable[[], Any]) -> None:
        self.client = client
        self.config = config
        self.repo = repo
        self.server = server

    async def find_sources(self, brief: str, scholar_id: str | None, draft_id: str | None = None) -> list[Passage]:
        scope = f"\nStay within scholar_id={scholar_id!r} unless the brief needs others." if scholar_id else ""
        async with AsyncExitStack() as stack:
            pairs = await open_tools(stack, self.server(), allowed=READ_ONLY_TOOLS)
            _, result = await run_tool_loop(
                self.client,
                self.config,
                "research.agent",
                system=RESEARCH_SYSTEM,
                pairs=pairs,
                messages=[{"role": "user", "content": f"Brief: {brief}{scope}"}],
                output=ResearchResult,
            )
        if result is None:
            log.warning("research_no_result", brief=brief)
            return []
        limit = self.config.limits.writer.max_sections * self.config.limits.writer.passages_per_section
        return await hydrate_research_picks(self.repo, result.picks, scholar_id, limit)


class OpenAISupervisor:
    def __init__(
        self, client: AsyncOpenAI, config: ModelsConfig, server: Callable[[], Any], external: Sequence[ServerSpec]
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
            if tool == "ask_corpus":
                ask_result = result
                numbering.clear()
                numbering.update(source_numbering(result))
            elif tool == "start_draft":
                ask_result = None
                numbering.clear()

        async def run() -> None:
            try:
                async with AsyncExitStack() as stack:
                    pairs = await open_tools(
                        stack,
                        self.server(),
                        allowed=SUPERVISOR_TOOLS,
                        external=self.external,
                        on_event=sink,
                        on_structured=keep_sources,
                    )
                    text, _ = await run_tool_loop(
                        self.client,
                        self.config,
                        "chat.supervisor",
                        system=SUPERVISOR_SYSTEM,
                        messages=list(messages),
                        pairs=pairs,
                    )
                    if ask_result is not None:
                        authoritative = str(ask_result.get("answer", ""))
                    elif external_used:
                        authoritative = (
                            "I found external information, but Footnote cannot present it as a verified answer. "
                            "Add a licensed source to the corpus and ask again."
                        )
                    else:
                        authoritative = text
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
