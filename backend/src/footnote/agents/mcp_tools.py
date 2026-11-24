"""MCP plumbing for the tool-using agents.

The agents get every tool through MCP, including Footnote's own (served in-process by
mcp_server/server.py), so the tools an agent uses are exactly the tools Claude Code or Claude
Desktop see when they connect to Footnote. MCP tools become Claude tools with the Anthropic SDK's
`async_mcp_tool`; the session handed to it is wrapped so every call is streamed to the UI, traced
to LangSmith, and can be renamed (external servers are namespaced to avoid collisions).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from anthropic.lib.tools.mcp import async_mcp_tool
from langsmith import trace
from mcp import Client, StdioServerParameters

log = structlog.get_logger(__name__)

EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]
StructuredSink = Callable[[str, dict[str, Any]], Awaitable[None]]


async def _no_events(event: str, data: dict[str, Any]) -> None:
    return None


def _summarise(result: Any, limit: int = 280) -> str:
    texts = [getattr(c, "text", "") for c in getattr(result, "content", []) or []]
    text = " ".join(t for t in texts if t)
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class ObservedSession:
    """Stands in for an MCP session: forwards call_tool, reporting each call as it happens."""

    client: Client
    server: str
    on_event: EventSink = _no_events
    prefix: str = ""
    on_structured: StructuredSink | None = None  # gets each successful call's structured result
    calls: int = field(default=0)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        real = name.removeprefix(self.prefix)
        self.calls += 1
        call_id = f"{self.server}-{self.calls}"
        await self.on_event("tool_call", {"id": call_id, "name": name, "input": arguments or {}})
        with trace(name=f"mcp:{self.server}.{real}", run_type="tool", inputs=arguments or {}) as run:
            try:
                result = await self.client.call_tool(real, arguments or {}, **kwargs)
            except Exception as exc:
                await self.on_event("tool_result", {"id": call_id, "name": name, "summary": str(exc), "is_error": True})
                raise
            run.end(outputs={"summary": _summarise(result, 2000), "is_error": bool(result.is_error)})
        await self.on_event(
            "tool_result",
            {
                "id": call_id,
                "name": name,
                "summary": _summarise(result),
                "is_error": bool(result.is_error),
            },
        )
        if self.on_structured is not None and result.structured_content and not result.is_error:
            await self.on_structured(real, result.structured_content)
        return result


@dataclass
class ServerSpec:
    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    allowed_tools: list[str] | None = None
    env: dict[str, str] | None = None


def load_external_servers(path: Path) -> list[ServerSpec]:
    """MCP servers the supervisor may use besides Footnote's own (config/mcp_servers.json)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for name, cfg in data.get("servers", {}).items():
        if not cfg.get("enabled", False):
            continue
        allowed = cfg.get("allowed_tools")
        if not isinstance(allowed, list) or not allowed:
            raise ValueError(f"Enabled external MCP server {name!r} requires a non-empty allowed_tools list")
        specs.append(
            ServerSpec(
                name=name,
                command=cfg.get("command"),
                args=cfg.get("args", []),
                url=cfg.get("url"),
                allowed_tools=allowed,
                env=cfg.get("env"),
            )
        )
    return specs


async def open_tools(
    stack: AsyncExitStack,
    footnote_server: Any,
    *,
    allowed: Sequence[str] | None,
    external: Sequence[ServerSpec] = (),
    on_event: EventSink = _no_events,
    on_structured: StructuredSink | None = None,
) -> list[tuple[Any, ObservedSession]]:
    """Open MCP sessions (Footnote in-process, externals over stdio/HTTP): (MCP tool, its session) pairs.

    Provider-neutral: Claude agents wrap these with the SDK's `async_mcp_tool` (connect_tools);
    OpenAI agents turn them into function tools (agents/openai_agents.py).
    """
    pairs: list[tuple[Any, ObservedSession]] = []
    own = await stack.enter_async_context(Client(footnote_server))
    session = ObservedSession(own, "footnote", on_event, on_structured=on_structured)
    pairs += [(t, session) for t in (await own.list_tools()).tools if allowed is None or t.name in allowed]
    for spec in external:
        target: Any = spec.url or StdioServerParameters(command=spec.command or "", args=spec.args, env=spec.env)
        try:
            ext = await stack.enter_async_context(Client(target))
        except Exception:
            log.warning("mcp_external_unavailable", server=spec.name, exc_info=True)
            continue
        prefix = f"{spec.name}__"
        ext_session = ObservedSession(ext, spec.name, on_event, prefix=prefix)
        for t in (await ext.list_tools()).tools:
            if spec.allowed_tools is None or t.name in spec.allowed_tools:
                pairs.append((t.model_copy(update={"name": prefix + t.name}), ext_session))
    return pairs


async def connect_tools(
    stack: AsyncExitStack,
    footnote_server: Any,
    *,
    allowed: Sequence[str] | None,
    external: Sequence[ServerSpec] = (),
    on_event: EventSink = _no_events,
    on_structured: StructuredSink | None = None,
) -> list[Any]:
    """Claude tools for the MCP servers' tools (the Anthropic SDK's MCP helper)."""
    pairs = await open_tools(
        stack, footnote_server, allowed=allowed, external=external, on_event=on_event, on_structured=on_structured
    )
    return [async_mcp_tool(t, session) for t, session in pairs]  # type: ignore[arg-type]


def describe_error(exc: BaseException) -> str:
    """The root cause, not the wrapper: MCP sessions run in task groups, which wrap failures in groups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def result_text(result: Any, limit: int = 60_000) -> str:
    """An MCP tool result as the text a model reads: structured JSON when present, else the text parts."""
    import json

    if getattr(result, "structured_content", None):
        text = json.dumps(result.structured_content, ensure_ascii=False)
    else:
        text = "\n".join(getattr(c, "text", "") for c in getattr(result, "content", []) or [])
    return text[:limit]
