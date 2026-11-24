"""OpenAI behind the same named routes (Responses API).

Implements the `Models` interface of llm/client.py, so every graph runs unchanged on either provider.
The one real difference is citations: Claude cites `search_result` blocks natively, while here the
passages are numbered in the prompt and the model cites them as [n] markers. The markers are mapped
back to passage ids in code, and a number that names no passage is dropped, so a citation can only
ever point at a passage that was actually shown.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar

import structlog
from langsmith import trace
from openai import AsyncOpenAI
from pydantic import BaseModel

from footnote.domain import Usage
from footnote.llm.client import (
    AnswerBlock,
    AnswerDone,
    AnswerEvent,
    CitedSpan,
    ModelError,
    ModelRefusal,
    TextDelta,
    _close_llm_run,
    _open_llm_run,
    _usage_meta,
    current_ledger,
)
from footnote.llm.config import ModelsConfig, RouteConfig

log = structlog.get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

CITE_INSTRUCTIONS = """

The search results are numbered [1], [2], ... in the user's message. Cite them by number in square \
brackets right after each sentence that uses them, like "... in 1926 [2]." or "[1][3]". Cite only \
those numbers, and cite every sentence you take from them."""

_WORD = re.compile(r"[a-z0-9]+")


def _overlap(a: str, b: str) -> float:
    wa = set(_WORD.findall(a.lower()))
    return len(wa & set(_WORD.findall(b.lower()))) / (len(wa) or 1)


def to_openai_input(messages: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Anthropic-shaped messages → Responses API input, with `search_result` blocks numbered.

    Returns the input and {n: {"source": passage id, "sentences": [...]}} for mapping markers back.
    """
    numbered: dict[int, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        parts: list[str] = []
        for block in content:
            if block.get("type") == "search_result":
                n = len(numbered) + 1
                sentences = [c["text"] for c in block.get("content", []) if c.get("type") == "text"]
                numbered[n] = {"source": block["source"], "sentences": sentences}
                parts.append(f"[{n}] {block.get('title', '')}\n" + " ".join(sentences))
            elif block.get("type") == "text":
                parts.append(block["text"])
        if numbered and any(b.get("type") == "search_result" for b in content):
            parts.insert(0, "Search results:")
        out.append({"role": m["role"], "content": "\n\n".join(parts)})
    return out, numbered


_MARKS = re.compile(r"\s*((?:\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\])+)([.,;:!?]?)")


def parse_cited_text(text: str, numbered: dict[int, dict[str, Any]]) -> list[AnswerBlock]:
    """Split an answer at its [n] markers into blocks that carry passage-id citations.

    `cited_text` is the source sentence closest to the segment, for highlighting only: the judge
    decides support, not this.
    """
    blocks: list[AnswerBlock] = []
    pos = 0
    for m in _MARKS.finditer(text):
        body = text[pos : m.start()] + m.group(2)
        numbers = list(dict.fromkeys(int(x) for x in re.findall(r"\d+", m.group(1))))
        cites = []
        for n in numbers:
            if n in numbered:
                src = numbered[n]
                best = max(src["sentences"], key=lambda s: _overlap(body, s), default="")
                cites.append(CitedSpan(source=src["source"], cited_text=best))
        if body.strip() or cites:
            blocks.append(AnswerBlock(text=body, citations=cites))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        blocks.append(AnswerBlock(text=tail, citations=[]))
    return blocks


def _check(route: str, resp: Any) -> None:
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", None) or []:
            if getattr(c, "type", None) == "refusal":
                raise ModelRefusal(route, "refusal", getattr(c, "refusal", None))
    if getattr(resp, "status", None) == "incomplete":
        reason = getattr(getattr(resp, "incomplete_details", None), "reason", None)
        raise ModelError(f"{route}: reply incomplete ({reason}); raise max_tokens in models.yaml")


class OpenAIModels:
    def __init__(self, config: ModelsConfig, client: AsyncOpenAI | None = None) -> None:
        self.config = config
        self.client = client or AsyncOpenAI(max_retries=3)

    def model_for(self, route: str) -> str:
        return self.config.route(route).model

    @staticmethod
    def _params(r: RouteConfig, system: str) -> dict[str, Any]:
        return {
            "model": r.model,
            "instructions": system,
            "max_output_tokens": r.max_tokens,
            "reasoning": {"effort": r.effort},
        }

    def _record(self, route: str, resp: Any, started: float) -> Usage:
        u = resp.usage
        usage = self.config.usage(resp.model, int(u.input_tokens if u else 0), int(u.output_tokens if u else 0))
        if (ledger := current_ledger()) is not None:
            ledger.record(route, resp.model, usage, int((time.perf_counter() - started) * 1000))
        return usage

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
        inp, _ = to_openai_input(messages)
        started = time.perf_counter()
        with trace(
            name=route,
            run_type="llm",
            inputs={"system": system, "input": inp},
            metadata={"model": r.model, "effort": r.effort, "route": route, "provider": "openai"},
        ) as run:
            resp = await self.client.responses.parse(**self._params(r, system), input=inp, text_format=output)  # type: ignore[arg-type]
            _check(route, resp)
            usage = self._record(route, resp, started)
            parsed = resp.output_parsed
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
        inp, numbered = to_openai_input(messages)
        instructions = system + (CITE_INSTRUCTIONS if numbered else "")
        started = time.perf_counter()
        run = _open_llm_run(route, r, {"system": instructions, "input": inp})
        try:
            async with self.client.responses.stream(**self._params(r, instructions), input=inp) as stream:  # type: ignore[arg-type]
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield TextDelta(event.delta)
                final = await stream.get_final_response()
            _check(route, final)
        except BaseException as exc:
            _close_llm_run(run, error=repr(exc))
            raise
        usage = self._record(route, final, started)
        blocks = parse_cited_text(final.output_text, numbered)
        _close_llm_run(run, outputs={"text": final.output_text, "usage_metadata": _usage_meta(usage)})
        yield AnswerDone(blocks=blocks, usage=usage, stop_reason=final.status)
