"""The grounding judge's prompt and output contract.

Standard library only, on purpose: `finetune/` imports this exact file to build training data and
to evaluate the fine-tuned model, so the prompt Claude is judged on and the prompt the small model
is trained on are byte-for-byte the same. Change the prompt here and the fine-tune data must be rebuilt
(`PROMPT_VERSION` is stamped on every training row and every recorded judgment).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

PROMPT_VERSION = "judge.v2"  # v2: passages carry their source (author, work, year); authorship rule
LABELS = ("supported", "partial", "unsupported")

JUDGE_SYSTEM = """\
You check one claim against source passages, for a research assistant that must never state \
anything its sources do not support.

Rule on the claim as a whole:
- "supported": everything the claim says is stated in the passages or follows directly from them.
- "partial": the core is supported, but the claim adds a detail, number, scope, date or degree of \
certainty that the passages do not give.
- "unsupported": the passages do not say it, say the opposite, or are about something else.

Judge only against the passages, never against what you know. A claim that is true in the world \
but absent from the passages is unsupported. Paraphrase is fine; the meaning must match.

Each passage's source says who wrote it and in which work. A passage from an author's own work is \
that author speaking: if Wells's pamphlet argues something, "Wells argued it" is supported. A \
passage that quotes or reports someone else's words supports only that they said them.

Reply with one JSON object and nothing else:
{"label": "supported" | "partial" | "unsupported", "evidence": "...", "rationale": "..."}
- evidence: the shortest exact quote, copied character for character from one passage, that decides \
the label; "" when unsupported.
- rationale: one sentence saying what is, or is not, in the passages."""

JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "evidence": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "evidence", "rationale"],
    "additionalProperties": False,
}


class JudgePassage(Protocol):
    id: str
    title: str
    text: str


def _field(p: JudgePassage | Mapping[str, Any], name: str, default: str | None = None) -> str:
    if isinstance(p, Mapping):
        value = p.get(name, default) if default is not None else p[name]
    else:
        value = getattr(p, name, default) if default is not None else getattr(p, name)
    return str(value if value is not None else "")


def _attr(value: str) -> str:
    return value.replace('"', "'").strip()


def render_judge_prompt(claim: str, passages: Sequence[JudgePassage | Mapping[str, Any]]) -> tuple[str, str]:
    """Return (system, user) for one claim.

    Passages may be objects or dicts with id/title/text and, optionally, `attribution` (who wrote it,
    which work, which year): the judge cannot rule on "Wells argued..." without knowing the passage
    is Wells's own writing.
    """
    blocks = []
    for p in passages:
        source = _field(p, "attribution", "")
        source_attr = f' source="{_attr(source)}"' if source else ""
        blocks.append(
            f'<passage id="{_attr(_field(p, "id"))}" title="{_attr(_field(p, "title"))}"{source_attr}>\n'
            f"{_field(p, 'text').strip()}\n</passage>"
        )
    if not blocks:
        blocks = ['<passage id="none" title="">\n(no passages)\n</passage>']
    user = "\n\n".join(blocks) + f"\n\n<claim>\n{claim.strip()}\n</claim>"
    return JUDGE_SYSTEM, user


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(raw: str) -> dict[str, str] | None:
    """Parse a model's reply into the contract, or None if it does not meet it.

    Tolerates a code fence or stray text around the object (small models do this); does not
    tolerate a missing key or a label outside the enum. None is counted as an invalid reply.
    """
    match = _JSON_OBJECT.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("label") not in LABELS:
        return None
    if not isinstance(data.get("evidence"), str) or not isinstance(data.get("rationale"), str):
        return None
    return {"label": data["label"], "evidence": data["evidence"], "rationale": data["rationale"]}


def evidence_is_verbatim(evidence: str, passages: Sequence[JudgePassage | Mapping[str, Any]]) -> bool:
    """True when the quoted evidence really occurs in a passage (whitespace-insensitive).

    A verdict of supported/partial whose evidence is not in the passages is a judge hallucination;
    the agents treat it as unsupported and the evals count it.
    """
    # An elided quote ("A ... B") is verbatim when every fragment is: each must occur, in one passage.
    fragments = [f for f in (_normalise(x) for x in _ELLIPSIS.split(evidence)) if len(f.split()) >= 2 or len(f) >= 8]
    if not fragments:
        return False
    texts = [_normalise(_field(p, "text")) for p in passages]
    return any(all(f in text for f in fragments) for text in texts)


_ELLIPSIS = re.compile(r"\s*(?:\.\s?\.\s?\.|…|\[\.\.\.\])\s*")


_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _normalise(text: str) -> str:
    """Case, whitespace, curly quotes and dashes do not make a quote non-verbatim."""
    return " ".join(text.translate(_PUNCT).split()).lower().strip(" .\"'")
