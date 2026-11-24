"""The grounding judge: does this claim follow from these passages?

Three interchangeable backends behind one interface:

    ApiJudge      - the `judge.grounding` route, on whichever provider it names (never the writer's model)
    LocalJudge    - the QLoRA fine-tuned small model, served by Ollama / llama.cpp
    CascadeJudge  - local first, escalate to Claude when the local verdict is unsure or invalid

All three use the exact prompt in judge_prompt.py, so their scores are comparable and the fine-tune
is trained on what it will be asked in production. Every verdict is recorded (RecordingJudge),
which is both an audit trail and the data flywheel: reviewed production judgments become training
rows for the next fine-tune.

A supported/partial verdict whose evidence quote is not actually in the passages is a judge
hallucination; it is downgraded to unsupported (fail closed) and flagged.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx
import structlog
from pydantic import BaseModel

from footnote.db.repo import Repo
from footnote.domain import Passage, Verdict, VerdictLabel
from footnote.judge_prompt import (
    JUDGE_OUTPUT_SCHEMA,
    PROMPT_VERSION,
    evidence_is_verbatim,
    parse_judge_output,
    render_judge_prompt,
)
from footnote.llm.client import Models

log = structlog.get_logger(__name__)


class JudgeOutput(BaseModel):
    label: VerdictLabel
    evidence: str
    rationale: str


@dataclass
class Ruling:
    verdict: Verdict
    valid: bool  # False when the backend's reply did not meet the output contract
    latency_ms: int
    escalated: bool = False
    raw: JudgeOutput | None = None  # what the judge said before the fail-closed rule (audit, training)
    failed_closed: bool = False  # True when an unverifiable evidence quote turned the verdict to unsupported


class Judge(Protocol):
    name: str

    async def rule(self, claim: str, passages: Sequence[Passage]) -> Ruling: ...


def _finalise(raw: JudgeOutput, passages: Sequence[Passage], judge: str, started: float) -> Ruling:
    label, rationale = raw.label, raw.rationale
    failed_closed = label != "unsupported" and not evidence_is_verbatim(raw.evidence, passages)
    if failed_closed:
        label, rationale = "unsupported", f"Evidence quote not found in the passages. ({raw.rationale})"
    return Ruling(
        verdict=Verdict(
            label=label, evidence=raw.evidence if label != "unsupported" else "", rationale=rationale, judge=judge
        ),
        valid=True,
        latency_ms=int((time.perf_counter() - started) * 1000),
        raw=raw,
        failed_closed=failed_closed,
    )


def _invalid(judge: str, started: float, why: str) -> Ruling:
    return Ruling(
        verdict=Verdict(label="unsupported", evidence="", rationale=f"Judge reply was invalid: {why}", judge=judge),
        valid=False,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


class ApiJudge:
    def __init__(self, models: Models, route: str = "judge.grounding") -> None:
        self.models = models
        self.route = route
        self.name = models.model_for(route)

    async def _ask(self, claim: str, passages: Sequence[Passage]) -> JudgeOutput:
        system, user = render_judge_prompt(claim, passages)
        return await self.models.structured(
            self.route, system=system, messages=[{"role": "user", "content": user}], output=JudgeOutput
        )

    async def raw_reply(self, claim: str, passages: Sequence[Passage]) -> str:
        """The verdict as JSON text, before the agents' fail-closed rule (the judge eval scores both)."""
        return (await self._ask(claim, passages)).model_dump_json()

    async def rule(self, claim: str, passages: Sequence[Passage]) -> Ruling:
        started = time.perf_counter()
        return _finalise(await self._ask(claim, passages), passages, self.name, started)


class LocalJudge:
    """The fine-tuned judge behind an OpenAI-compatible chat endpoint (Ollama, llama.cpp server)."""

    def __init__(
        self, base_url: str, model: str, timeout_s: float = 60, client: httpx.AsyncClient | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"local:{model}"
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def raw_reply(self, claim: str, passages: Sequence[Passage]) -> str:
        """The model's reply text, unparsed (the judge eval scores raw replies)."""
        system, user = render_judge_prompt(claim, passages)
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0,
                "max_tokens": 400,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "verdict", "schema": JUDGE_OUTPUT_SCHEMA, "strict": True},
                },
            },
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"] or "")

    async def rule(self, claim: str, passages: Sequence[Passage]) -> Ruling:
        started = time.perf_counter()
        try:
            content = await self.raw_reply(claim, passages)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            return _invalid(self.name, started, f"{type(exc).__name__}: {exc}")
        parsed = parse_judge_output(content)
        if parsed is None:
            return _invalid(self.name, started, "not the verdict JSON")
        return _finalise(JudgeOutput.model_validate(parsed), passages, self.name, started)

    async def aclose(self) -> None:
        await self._client.aclose()


class CascadeJudge:
    """Cheap judge first; the expensive one only where the cheap one is unsure.

    Escalating on `partial` and on invalid replies keeps recall on unsupported claims close to the
    strong judge while most supported claims never reach it (evals/judge.py measures both the
    escalation rate and the cost per 1k rulings).
    """

    def __init__(self, primary: Judge, fallback: Judge, escalate_on: Sequence[str]) -> None:
        self.primary = primary
        self.fallback = fallback
        self.escalate_on = set(escalate_on)
        self.name = f"cascade({primary.name}->{fallback.name})"

    async def rule(self, claim: str, passages: Sequence[Passage]) -> Ruling:
        first = await self.primary.rule(claim, passages)
        reason = "invalid" if not first.valid else first.verdict.label
        if reason not in self.escalate_on:
            return first
        second = await self.fallback.rule(claim, passages)
        second.escalated = True
        second.latency_ms += first.latency_ms
        return second


class RecordingJudge:
    """Writes every ruling to the `judgments` collection, with the prompt version."""

    def __init__(self, inner: Judge, repo: Repo, *, source: str = "app") -> None:
        self.inner = inner
        self.repo = repo
        self.source = source
        self.name = inner.name

    async def rule(self, claim: str, passages: Sequence[Passage]) -> Ruling:
        ruling = await self.inner.rule(claim, passages)
        try:
            await self.repo.record_judgment(
                {
                    "prompt_version": PROMPT_VERSION,
                    "claim": claim,
                    "passages": [
                        {"id": p.id, "title": p.title, "attribution": p.attribution, "text": p.text} for p in passages
                    ],
                    "passages_hash": hashlib.sha256("|".join(p.id for p in passages).encode()).hexdigest()[:16],
                    "label": ruling.verdict.label,
                    "evidence": ruling.verdict.evidence,
                    "rationale": ruling.verdict.rationale,
                    "judge": ruling.verdict.judge,
                    "valid": ruling.valid,
                    "escalated": ruling.escalated,
                    "failed_closed": ruling.failed_closed,
                    "raw": ruling.raw.model_dump() if ruling.raw else None,
                    "latency_ms": ruling.latency_ms,
                    "source": self.source,
                    "human_label": None,  # set by a person before the row may be used for training (finetune/)
                }
            )
        except Exception:  # the audit write must not fail the request
            log.warning("judgment_record_failed", exc_info=True)
        return ruling


class JudgePool:
    """Rules on many claims concurrently, bounded so a long answer cannot flood the API."""

    def __init__(self, judge: Judge, concurrency: int) -> None:
        self.judge = judge
        self._sem = asyncio.Semaphore(concurrency)

    @property
    def name(self) -> str:
        return self.judge.name

    async def rule_all(self, items: Sequence[tuple[str, Sequence[Passage]]]) -> list[Ruling]:
        async def one(claim: str, passages: Sequence[Passage]) -> Ruling:
            async with self._sem:
                return await self.judge.rule(claim, passages)

        return list(await asyncio.gather(*(one(c, p) for c, p in items)))
