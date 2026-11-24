"""Claude behind named routes.

Every call names a route from config/models.yaml; this module turns it into a Messages API
request through the official Anthropic SDK, records cost into a per-request ledger, and traces it
to LangSmith. Graph code never touches SDK types: answers come back as `AnswerEvent`s, typed
outputs as Pydantic models, so the offline fakes (llm/fake.py) can stand in for the real thing.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import anthropic
import structlog
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaCitationSearchResultLocation
from langsmith import trace
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree
from langsmith.utils import tracing_is_enabled
from pydantic import BaseModel

from footnote.domain import Usage
from footnote.llm.config import ModelsConfig, RouteConfig

log = structlog.get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

# Models that accept the server-side refusal fallback. Haiku has none.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_SUPPORTS_FALLBACKS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5")


class ModelError(RuntimeError):
    """A model call ended without a usable result."""


class ModelRefusal(ModelError):
    def __init__(self, route: str, category: str | None, explanation: str | None) -> None:
        super().__init__(f"{route}: the model declined ({category or 'no category'})")
        self.category = category
        self.explanation = explanation


# ── per-request cost ledger ──────────────────────────────────────────────────────────────────


@dataclass
class Ledger:
    usage: Usage = field(default_factory=Usage)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, route: str, model: str, usage: Usage, latency_ms: int) -> None:
        self.usage = self.usage.add(usage)
        self.calls.append({"route": route, "model": model, "latency_ms": latency_ms, **usage.model_dump()})


_ledger: contextvars.ContextVar[Ledger | None] = contextvars.ContextVar("footnote_ledger", default=None)


def start_ledger() -> Ledger:
    """Begin accounting for one request (an API call, an eval case). Returns the ledger."""
    ledger = Ledger()
    _ledger.set(ledger)
    return ledger


def current_ledger() -> Ledger | None:
    return _ledger.get()


def _record(config: ModelsConfig, route: str, model: str, raw_usage: Any, started: float) -> Usage:
    input_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
    input_tokens += int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0)
    input_tokens += int(getattr(raw_usage, "cache_creation_input_tokens", 0) or 0)
    output_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)
    usage = config.usage(model, input_tokens, output_tokens)
    latency_ms = int((time.perf_counter() - started) * 1000)
    if (ledger := _ledger.get()) is not None:
        ledger.record(route, model, usage, latency_ms)
    return usage


# ── answer streams (text with native citations) ─────────────────────────────────────────────


@dataclass
class CitedSpan:
    source: str  # the search_result `source` we sent: a passage id
    cited_text: str


@dataclass
class AnswerBlock:
    text: str
    citations: list[CitedSpan]


@dataclass
class TextDelta:
    text: str


@dataclass
class AnswerDone:
    blocks: list[AnswerBlock]
    usage: Usage
    stop_reason: str | None


AnswerEvent = TextDelta | AnswerDone


class Models(Protocol):
    """What the agents need from a model provider. Implemented by ClaudeModels and FakeModels."""

    config: ModelsConfig

    async def structured(
        self,
        route: str,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        output: type[T],
        cache_system: bool = False,
    ) -> T: ...

    def stream_answer(
        self,
        route: str,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[AnswerEvent]: ...

    def model_for(self, route: str) -> str: ...


def search_result_block(passage_id: str, title: str, sentences: Sequence[str]) -> dict[str, Any]:
    """A passage as a `search_result` content block, so Claude cites it natively.

    Each sentence is its own text block: the text block is the smallest unit Claude can cite,
    so per-sentence blocks give per-sentence citations instead of whole-passage ones.
    """
    return {
        "type": "search_result",
        "source": passage_id,
        "title": title,
        "content": [{"type": "text", "text": s} for s in sentences if s.strip()],
        "citations": {"enabled": True},
    }


class ClaudeModels:
    def __init__(self, config: ModelsConfig, client: AsyncAnthropic | None = None) -> None:
        self.config = config
        self.client = client or AsyncAnthropic(max_retries=3)

    def model_for(self, route: str) -> str:
        return self.config.route(route).model

    def _params(
        self, r: RouteConfig, system: str, messages: Sequence[dict[str, Any]], cache_system: bool
    ) -> dict[str, Any]:
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if cache_system:
            # Stable, long system prompts (the writer's source passages) are reused across every
            # section of a draft; caching them makes sections 2..n read the prefix at ~0.1x cost.
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}
        params: dict[str, Any] = {
            "model": r.model,
            "max_tokens": r.max_tokens,
            "system": system_blocks,
            "messages": list(messages),
            "output_config": {"effort": r.effort},
        }
        if r.model.startswith(_SUPPORTS_FALLBACKS):
            params["betas"] = [_FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params

    @staticmethod
    def _check(route: str, resp: Any) -> None:
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise ModelRefusal(route, getattr(details, "category", None), getattr(details, "explanation", None))
        if resp.stop_reason == "max_tokens":
            raise ModelError(f"{route}: hit max_tokens; raise it in models.yaml")

    async def structured(
        self,
        route: str,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        output: type[T],
        cache_system: bool = False,
    ) -> T:
        r = self.config.route(route)
        params = self._params(r, system, messages, cache_system)
        started = time.perf_counter()
        with trace(
            name=route,
            run_type="llm",
            inputs={"system": system, "messages": list(messages)},
            metadata={"model": r.model, "effort": r.effort, "route": route},
        ) as run:
            resp = await self.client.beta.messages.parse(**params, output_format=output)
            self._check(route, resp)
            usage = _record(self.config, route, resp.model, resp.usage, started)
            parsed = resp.parsed_output
            if parsed is None:
                raise ModelError(f"{route}: no structured output in the reply")
            run.end(outputs={"output": parsed.model_dump(), "usage_metadata": _usage_meta(usage)})
        return parsed

    async def stream_answer(
        self,
        route: str,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[AnswerEvent]:
        r = self.config.route(route)
        params = self._params(r, system, messages, cache_system=False)
        started = time.perf_counter()
        # A `with trace(...)` cannot span the yields of an async generator (its context variable
        # would be reset in another context if the consumer stops early), so the LLM run is
        # posted by hand as a child of whatever node is running.
        run = _open_llm_run(route, r, {"system": system, "messages": list(messages)})
        try:
            async with self.client.beta.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield TextDelta(event.text)
                final = await stream.get_final_message()
            self._check(route, final)
        except BaseException as exc:
            _close_llm_run(run, error=repr(exc))
            raise
        usage = _record(self.config, route, final.model, final.usage, started)
        blocks = [
            AnswerBlock(
                text=b.text,
                citations=[
                    CitedSpan(source=c.source, cited_text=c.cited_text)
                    for c in (b.citations or [])
                    if isinstance(c, BetaCitationSearchResultLocation)
                ],
            )
            for b in final.content
            if b.type == "text"
        ]
        _close_llm_run(run, outputs={"blocks": [b.__dict__ for b in blocks], "usage_metadata": _usage_meta(usage)})
        yield AnswerDone(blocks=blocks, usage=usage, stop_reason=final.stop_reason)


def _open_llm_run(route: str, r: RouteConfig, inputs: dict[str, Any]) -> RunTree | None:
    extra = {"metadata": {"model": r.model, "effort": r.effort, "route": route}}
    parent = get_current_run_tree()
    if parent is not None:
        run = parent.create_child(name=route, run_type="llm", inputs=inputs, extra=extra)
    elif tracing_is_enabled():
        run = RunTree(name=route, run_type="llm", inputs=inputs, extra=extra)
    else:
        return None
    try:
        run.post()
    except Exception:  # tracing must never break a request
        log.warning("langsmith_post_failed", route=route)
        return None
    return run


def _close_llm_run(run: RunTree | None, *, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
    if run is None:
        return
    try:
        run.end(outputs=outputs, error=error)
        run.patch()
    except Exception:
        log.warning("langsmith_patch_failed", run=str(run.id))


def _usage_meta(usage: Usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }


def is_retryable(exc: BaseException) -> bool:
    """Errors worth one more try at the graph level (the SDK has already retried transport errors)."""
    if isinstance(exc, anthropic.RateLimitError | anthropic.APIConnectionError | anthropic.APITimeoutError):
        return True
    return isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500
