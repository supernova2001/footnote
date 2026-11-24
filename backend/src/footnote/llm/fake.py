"""Deterministic stand-ins for every model route, for the offline mode and the unit tests.

They imitate the *shape* of each decision with simple rules (word overlap for the judge, first
sentences for answers) so graphs, MCP tools, the API and the UI can run without keys. Production
code never uses them, and no quality number is ever reported from them.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel

from footnote.domain import Usage
from footnote.llm.client import AnswerBlock, AnswerDone, AnswerEvent, CitedSpan, TextDelta
from footnote.llm.config import ModelsConfig
from footnote.retrieval.store import tokens

T = TypeVar("T", bound=BaseModel)
Handler = Callable[[str, Sequence[dict[str, Any]], type[BaseModel]], BaseModel]


def _user_text(messages: Sequence[dict[str, Any]]) -> str:
    out = []
    for m in messages:
        if m["role"] != "user":
            continue
        content = m["content"]
        if isinstance(content, str):
            out.append(content)
        else:
            out += [b.get("text", "") for b in content if b.get("type") == "text"]
    return "\n".join(out)


def _search_results(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            found = [b for b in m["content"] if b.get("type") == "search_result"]
            if found:
                return found
    return []


def _short_title(text: str, limit: int = 60) -> str:
    text = text.strip().rstrip(".")
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:(")


def overlap(a: str, b: str) -> float:
    ta = set(tokens(a))
    return len(ta & set(tokens(b))) / (len(ta) or 1)


class FakeModels:
    def __init__(self, config: ModelsConfig, handlers: dict[str, Handler] | None = None) -> None:
        self.config = config
        self.handlers = handlers or {}
        self.calls: list[tuple[str, str]] = []

    def model_for(self, route: str) -> str:
        return f"fake:{route}"

    async def structured(
        self,
        route: str,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
        output: type[T],
        cache_system: bool = False,
    ) -> T:
        self.calls.append((route, output.__name__))
        if route in self.handlers:
            return self.handlers[route](system, messages, output)  # type: ignore[return-value]
        return getattr(self, f"_{output.__name__}", self._default)(system, messages, output)

    def _default(self, system: str, messages: Sequence[dict[str, Any]], output: type[T]) -> T:
        raise NotImplementedError(f"FakeModels has no default for {output.__name__}")

    # ── per-schema defaults ──

    def _AskPlan(self, system: str, messages: Sequence[dict[str, Any]], output: type[T]) -> T:
        text = _user_text(messages).split("\n(The reader")[0]
        found = re.findall(r"^(?:New question|Question): (.*)$", text, re.M)
        q = (found[-1] if found else text).strip()
        off_topic = re.search(r"\b(python|javascript|recipe|stock price|weather)\b", q, re.I)
        return output.model_validate(
            {
                "standalone_question": q,
                "in_scope": not off_topic,
                "reason": "That is outside this collection." if off_topic else "",
                "queries": [q],
            }
        )

    def _ClaimSet(self, system: str, messages: Sequence[dict[str, Any]], output: type[T]) -> T:
        text = _user_text(messages)
        segs = re.findall(r"^\[(\d+)\] (.+)$", text, re.M)
        claims = [{"segment": int(i), "text": s} for i, s in segs if len(tokens(s)) >= 3]
        abstained = any("do not cover" in s or "does not cover" in s for _, s in segs) and not claims
        return output.model_validate({"abstained": abstained, "claims": claims})

    def _JudgeOutput(self, system: str, messages: Sequence[dict[str, Any]], output: type[T]) -> T:
        text = _user_text(messages)
        claim = re.search(r"<claim>\n(.*?)\n</claim>", text, re.S)
        passages = re.findall(r"<passage [^>]*>\n(.*?)\n</passage>", text, re.S)
        claim_text = claim.group(1) if claim else ""
        best, best_sentence = 0.0, ""
        for p in passages:
            for sentence in re.split(r"(?<=[.!?])\s+", p):
                score = overlap(claim_text, sentence)
                if score > best:
                    best, best_sentence = score, sentence
        label = "supported" if best >= 0.6 else "partial" if best >= 0.35 else "unsupported"
        return output.model_validate(
            {
                "label": label,
                "evidence": best_sentence if label != "unsupported" else "",
                "rationale": f"(offline judge) {best:.0%} of the claim's content words appear in the closest sentence.",
            }
        )

    def _Outline(self, system: str, messages: Sequence[dict[str, Any]], output: type[T]) -> T:
        text = _user_text(messages)
        ids = re.findall(r"^\[([^\]]+)\]", text, re.M)
        brief = re.search(r"Brief: (.*)", text)
        groups = [ids[i : i + 3] for i in range(0, min(len(ids), 9), 3)] or [[]]
        headings = ["Who they were", "What they wrote", "Why it still matters"]
        return output.model_validate(
            {
                "title": _short_title(brief.group(1) if brief else "A short history"),
                "sections": [
                    {"heading": headings[i % 3], "goal": "Tell this part from the sources.", "passage_ids": g}
                    for i, g in enumerate(groups)
                    if g
                ],
            }
        )

    def _EditorVerdict(self, system: str, messages: Sequence[dict[str, Any]], output: type[T]) -> T:
        text = _user_text(messages)
        measured = re.search(r"Measured Flesch-Kincaid grade: ([\d.-]+)", text)
        target = re.search(r"Target grade: ([\d.]+)", text)
        gap = float(measured.group(1)) - float(target.group(1)) if measured and target else 0.0
        if gap > 3:
            return output.model_validate(
                {
                    "fit": "too_hard",
                    "note": f"(offline editor) Reads {gap:.0f} grades above "
                    "the target: shorter sentences and plainer words.",
                }
            )
        if gap < -3:
            return output.model_validate({"fit": "too_easy", "note": "(offline editor) Reads below the target level."})
        return output.model_validate({"fit": "fits", "note": "(offline editor) Within three grades of the target."})

    def _ResearchResult(self, system: str, messages: Sequence[dict[str, Any]], output: type[T]) -> T:
        return output.model_validate({"picks": [], "not_covered": ""})

    # ── cited answers ──

    async def stream_answer(
        self,
        route: str,
        *,
        system: str,
        messages: Sequence[dict[str, Any]],
    ) -> AsyncIterator[AnswerEvent]:
        self.calls.append((route, "answer"))
        results = _search_results(messages)
        question = _user_text(messages)
        revising = any(m["role"] == "assistant" for m in messages)
        ranked = sorted(
            results, key=lambda r: overlap(question, " ".join(b["text"] for b in r["content"])), reverse=True
        )
        blocks: list[AnswerBlock] = []
        if not ranked:
            blocks.append(AnswerBlock(text="The sources here do not cover that question.", citations=[]))
        else:
            blocks.append(AnswerBlock(text="According to the sources, ", citations=[]))
            for r in ranked[: 1 if revising else 2]:
                sentence = max(r["content"], key=lambda b: overlap(question, b["text"]))["text"]
                blocks.append(
                    AnswerBlock(text=sentence + " ", citations=[CitedSpan(source=r["source"], cited_text=sentence)])
                )
        for b in blocks:
            for word in re.findall(r"\S+\s*", b.text):
                yield TextDelta(word)
        yield AnswerDone(blocks=blocks, usage=Usage(), stop_reason="end_turn")
