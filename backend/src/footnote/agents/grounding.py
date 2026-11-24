"""Shared by the RAG and writer agents: cited text → segments → atomic claims → verdicts.

1. Claude answers over `search_result` blocks, so citations come back natively as passage ids.
2. A claims model splits the text into atomic, self-contained factual claims (a sentence like
   "Woodson founded the Journal of Negro History in 1916 and launched Negro History Week" is two).
3. Each claim is judged against the passages its segment cited (or all passages, if it cited none).
4. A segment takes the verdict of its weakest claim.
5. A word-bearing segment omitted by the claim model is checked as a claim itself (fail closed).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from footnote.agents.judge import JudgePool, Ruling
from footnote.domain import Citation, Passage, Segment, Verdict
from footnote.llm.client import AnswerBlock, Models

_RANK = {"unsupported": 0, "partial": 1, "supported": 2}
_WORDY = re.compile(r"[A-Za-z]{3,}")


def blocks_to_segments(blocks: Sequence[AnswerBlock], known_ids: set[str]) -> list[Segment]:
    """Turn Claude's text blocks into display segments.

    Blocks split at citation boundaries, so an uncited fragment such as ", and" or "." is merged
    into its neighbour rather than shown (and judged) as a sentence of its own. Citations to ids
    we did not send are dropped.
    """
    segments: list[Segment] = []
    for b in blocks:
        cites = [Citation(passage_id=c.source, cited_text=c.cited_text) for c in b.citations if c.source in known_ids]
        if not cites and segments and not _WORDY.search(b.text):
            segments[-1].text += b.text
            continue
        if not cites and segments and not segments[-1].citations and not b.text.startswith("\n"):
            segments[-1].text += b.text
            continue
        segments.append(Segment(text=b.text, citations=cites))
    for s in segments:
        s.text = s.text.replace("\n\n\n", "\n\n")
    return [s for s in segments if s.text.strip()]


class Claim(BaseModel):
    segment: int = Field(description="Index of the segment the claim comes from")
    text: str = Field(description="One atomic factual claim, self-contained (names instead of pronouns)")


class ClaimSet(BaseModel):
    abstained: bool = Field(description="True when the text's substance is that the sources do not answer the question")
    claims: list[Claim]


CLAIMS_SYSTEM = """\
You prepare an answer for fact-checking. Split it into atomic factual claims.

- One claim per fact: "Woodson founded the Journal of Negro History in 1916 and started Negro \
History Week" is two claims.
- Make each claim self-contained: replace pronouns with names, keep dates and numbers exact.
- Keep the source's framing: if the text says "Washington argued that...", the claim is what \
Washington argued, not the argument stated as fact.
- Skip sentences that state no fact about the world (greetings, "here is what the sources say", \
offers to help).
- `segment` is the index of the numbered segment the claim came from.
- `abstained` is true when the answer's substance is that the sources do not cover the question."""


def render_segments(segments: Sequence[Segment]) -> str:
    return "\n".join(f"[{i}] {s.text.strip()}" for i, s in enumerate(segments))


async def extract_claims(models: Models, route: str, segments: Sequence[Segment], question: str) -> ClaimSet:
    if not segments:
        return ClaimSet(abstained=True, claims=[])
    user = f"Question: {question}\n\nAnswer, by segment:\n{render_segments(segments)}"
    out = await models.structured(
        route, system=CLAIMS_SYSTEM, messages=[{"role": "user", "content": user}], output=ClaimSet
    )
    out.claims = [c for c in out.claims if 0 <= c.segment < len(segments) and c.text.strip()]
    if not out.abstained:
        covered = {c.segment for c in out.claims}
        out.claims.extend(
            Claim(segment=i, text=segment.text.strip())
            for i, segment in enumerate(segments)
            if i not in covered and _WORDY.search(segment.text)
        )
    return out


async def judge_segments(
    pool: JudgePool, segments: list[Segment], claims: Sequence[Claim], passages: Sequence[Passage]
) -> list[tuple[Claim, Ruling]]:
    """Rule on every claim and set each segment's verdict to its weakest claim's. Returns the rulings."""
    by_id = {p.id: p for p in passages}
    items = []
    for c in claims:
        cited = [by_id[x.passage_id] for x in segments[c.segment].citations if x.passage_id in by_id]
        items.append((c.text, cited or list(passages)))
    rulings = await pool.rule_all(items)
    pairs = list(zip(claims, rulings, strict=True))
    worst: dict[int, Verdict] = {}
    for claim, ruling in pairs:
        current = worst.get(claim.segment)
        if current is None or _RANK[ruling.verdict.label] < _RANK[current.label]:
            worst[claim.segment] = ruling.verdict
    for i, seg in enumerate(segments):
        seg.verdict = worst.get(i)  # None: the segment states no fact (framing text)
    return pairs


def failing(pairs: Sequence[tuple[Claim, Ruling]]) -> list[tuple[Claim, Ruling]]:
    return [(c, r) for c, r in pairs if r.verdict.label != "supported"]


def publishable(segment: Segment) -> bool:
    """Only verified factual prose with a source can cross the publication boundary.

    A segment with no verdict is non-factual framing (or a deliberate abstention). Factual
    segments must be fully supported and carry at least one citation; `partial` is withheld.
    """
    return segment.verdict is None or (segment.verdict.label == "supported" and bool(segment.citations))
