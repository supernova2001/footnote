"""Ranking metrics (counting, so code): recall@k, reciprocal rank, nDCG@k with binary relevance."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of the gold passages found in the top k."""
    if not gold:
        return 0.0
    top = set(ranked[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str]) -> float:
    golds = set(gold)
    for i, pid in enumerate(ranked, start=1):
        if pid in golds:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    golds = set(gold)
    dcg = sum(1.0 / math.log2(i + 1) for i, pid in enumerate(ranked[:k], start=1) if pid in golds)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(golds), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def p50(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2)
