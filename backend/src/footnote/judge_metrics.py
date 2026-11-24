"""How well a grounding judge agrees with gold labels.

Standard library only, like `judge_prompt.py`: `finetune/eval_judge.py` loads this file by path to
score the base and fine-tuned small models, and the backend's eval scores the Claude judge with the
same function, so every judge in the comparison table is measured the same way.

Definitions (see `score_judge`):
- A reply that `parse_judge_output` rejects is predicted as "invalid": it counts against recall of
  its gold label and towards no label's precision.
- macro-F1 averages F1 over the three labels (zero when a label is never predicted).
- unsupported recall is the hallucination catch rate: of the claims the sources do not support,
  how many the judge flagged.
- "deployed" scores apply the rule the agents follow: an invalid reply, or a supported/partial
  verdict whose evidence is not a verbatim quote from the passages, is treated as unsupported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .judge_prompt import LABELS, evidence_is_verbatim, parse_judge_output

INVALID = "invalid"


@dataclass(frozen=True)
class JudgeCase:
    """One judged claim: the gold label, the judge's raw reply, and what it was judged against."""

    gold: str
    reply: str
    passages: Sequence[Any] = ()
    latency_ms: float | None = None
    id: str = ""


@dataclass
class LabelScores:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class JudgeReport:
    n: int
    accuracy: float
    macro_f1: float
    unsupported_recall: float
    json_valid_rate: float
    evidence_verbatim_rate: float | None
    deployed_macro_f1: float
    deployed_unsupported_recall: float
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    per_label: dict[str, LabelScores] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predicted_label(case: JudgeCase, *, as_deployed: bool = False) -> str:
    parsed = parse_judge_output(case.reply)
    if parsed is None:
        return "unsupported" if as_deployed else INVALID
    label = parsed["label"]
    if as_deployed and label != "unsupported" and not evidence_is_verbatim(parsed["evidence"], case.passages):
        return "unsupported"
    return label


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy's default); q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _label_scores(gold: Sequence[str], pred: Sequence[str]) -> dict[str, LabelScores]:
    scores: dict[str, LabelScores] = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p == label)
        predicted = sum(1 for p in pred if p == label)
        support = sum(1 for g in gold if g == label)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores[label] = LabelScores(precision=precision, recall=recall, f1=f1, support=support)
    return scores


def score_judge(cases: Sequence[JudgeCase]) -> JudgeReport:
    if not cases:
        raise ValueError("score_judge needs at least one case")
    unknown = {c.gold for c in cases} - set(LABELS)
    if unknown:
        raise ValueError(f"gold labels outside the contract: {sorted(unknown)}")

    gold = [c.gold for c in cases]
    pred = [predicted_label(c) for c in cases]
    deployed = [predicted_label(c, as_deployed=True) for c in cases]

    per_label = _label_scores(gold, pred)
    deployed_per_label = _label_scores(gold, deployed)

    confusion = {g: {p: 0 for p in (*LABELS, INVALID)} for g in LABELS}
    for g, p in zip(gold, pred, strict=True):
        confusion[g][p] += 1

    parsed = [parse_judge_output(c.reply) for c in cases]
    with_evidence = [
        (p, c) for p, c in zip(parsed, cases, strict=True) if p is not None and p["label"] != "unsupported"
    ]
    evidence_rate = (
        sum(1 for p, c in with_evidence if evidence_is_verbatim(p["evidence"], c.passages)) / len(with_evidence)
        if with_evidence
        else None
    )
    latencies = [c.latency_ms for c in cases if c.latency_ms is not None]

    return JudgeReport(
        n=len(cases),
        accuracy=sum(1 for g, p in zip(gold, pred, strict=True) if g == p) / len(cases),
        macro_f1=sum(s.f1 for s in per_label.values()) / len(LABELS),
        unsupported_recall=per_label["unsupported"].recall,
        json_valid_rate=sum(1 for p in parsed if p is not None) / len(cases),
        evidence_verbatim_rate=evidence_rate,
        deployed_macro_f1=sum(s.f1 for s in deployed_per_label.values()) / len(LABELS),
        deployed_unsupported_recall=deployed_per_label["unsupported"].recall,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        per_label=per_label,
        confusion=confusion,
    )


def eval_row(report: JudgeReport, judge: str, cost_per_1k_usd: float = 0.0) -> dict[str, Any]:
    """One `judge` row of the API's EvalReport (docs/api-contract.md), exactly those keys."""
    return {
        "judge": judge,
        "n": report.n,
        "macro_f1": round(report.macro_f1, 4),
        "unsupported_recall": round(report.unsupported_recall, 4),
        "json_valid_rate": round(report.json_valid_rate, 4),
        "p50_latency_ms": round(report.p50_latency_ms, 1) if report.p50_latency_ms is not None else 0.0,
        "cost_per_1k_usd": round(cost_per_1k_usd, 4),
    }


def cases_from_rows(rows: Sequence[Mapping[str, Any]], replies: Mapping[str, str]) -> list[JudgeCase]:
    """Pair dataset rows ({"id", "label", "passages"}) with replies keyed by row id."""
    return [
        JudgeCase(
            gold=str(r["label"]), reply=replies.get(str(r["id"]), ""), passages=r.get("passages", ()), id=str(r["id"])
        )
        for r in rows
    ]
