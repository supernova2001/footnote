"""Rule-based claims whose label is known by construction.

Each function takes one anchor sentence (verbatim from a passage) and the passage text, and returns
a `Candidate` or None when the rule does not apply safely. "Safely" is the point: a rule that might
produce a claim whose true label differs from the intended one declines instead. The label
definitions are the contract's (docs/api-contract.md, "The grounding judge"):

- supported: the anchor itself, or the anchor minus a non-restrictive clause (says less, adds nothing).
- partial: the anchor plus something the passage does not give (a date, a place, a qualifier, more
  certainty than the source).
- unsupported: the anchor contradicted (negated or reversed, a number or a name changed).

Changing a number the passage gives contradicts it (unsupported); adding a number it does not give
only goes beyond it (partial). Cross-document and same-document "outside fact" negatives need other
passages, so they are built in dataset.py.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .text import content_words, names, normalize, numbers

SUPPORTED, PARTIAL, UNSUPPORTED = "supported", "partial", "unsupported"


@dataclass(frozen=True)
class Candidate:
    claim: str
    label: str
    method: str
    evidence: str
    rationale: str


def _capitalize_like(original: str, replacement: str) -> str:
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _in_passage(fragment: str, passage: str) -> bool:
    return normalize(fragment) in normalize(passage)


# --- supported ---------------------------------------------------------------------------------


def verbatim(anchor: str, passage: str, rng: random.Random) -> Candidate | None:
    return Candidate(anchor, SUPPORTED, "verbatim", anchor, "The passage states this directly.")


_TRAILING_CLAUSE = re.compile(r",\s+(?:which|who|whose|where)\b[^,;]*(?=[.!?][\"'”’)]*$)")
_PARENTHETICAL = re.compile(r"\s*\([^()]{3,120}\)")


def trim_clause(anchor: str, passage: str, rng: random.Random) -> Candidate | None:
    """Drop a non-restrictive clause or a parenthetical: the claim says less but nothing new."""
    for pattern in (_TRAILING_CLAUSE, _PARENTHETICAL):
        match = pattern.search(anchor)
        if not match:
            continue
        claim = (anchor[: match.start()] + anchor[match.end() :]).strip()
        if len(claim) >= 30 and len(content_words(claim)) >= 4 and claim != anchor:
            return Candidate(
                claim,
                SUPPORTED,
                "trim_clause",
                anchor,
                "The passage states this; the claim leaves out a detail and adds nothing.",
            )
    return None


# --- partial -----------------------------------------------------------------------------------

# (pattern, replacement, case_sensitive). One substitution per claim, so the claim differs from the
# source in exactly one degree of certainty or scope.
_CERTAINTY: list[tuple[str, str, bool]] = [
    (r"\bsuggests\b", "proves", False),
    (r"\bsuggested\b", "proved", False),
    (r"\bsuggest\b", "prove", False),
    (r"\bindicates\b", "proves", False),
    (r"\bindicated\b", "proved", False),
    (r"\bmay\b", "will", True),  # lowercase only: "May 1890" is a month
    (r"\bmight\b", "will", True),
    (r"\blikely\b", "certainly", False),
    (r"\bprobably\b", "certainly", False),
    (r"\bpossibly\b", "certainly", False),
    (r"\bperhaps\b", "certainly", False),
    (r"\bsometimes\b", "always", False),
    (r"\boften\b", "always", False),
    (r"\busually\b", "always", False),
    (r"\bfrequently\b", "always", False),
    (r"\b(?:some|many|several|most) of\b", "all of", False),
    (r"\b(?:about|around|approximately|nearly|roughly|almost) (?=\d)", "exactly ", False),
    (r"\bseems\b", "is", False),
    (r"\bseemed\b", "was", False),
    (r"\bhoped\b", "ensured", False),
    (r"\b(?:tried|attempted) to\b", "managed to", False),
    (r"\bargued\b", "proved", False),
    (r"\bargues\b", "proves", False),
    (r"\bclaimed\b", "proved", False),
    (r"\bbelieved\b", "knew", False),
]


def certainty_upgrade(anchor: str, passage: str, rng: random.Random) -> Candidate | None:
    options = []
    for pattern, replacement, case_sensitive in _CERTAINTY:
        match = re.search(pattern, anchor, 0 if case_sensitive else re.IGNORECASE)
        if match:
            options.append((match, replacement))
    if not options:
        return None
    match, replacement = rng.choice(options)
    new = _capitalize_like(match.group(0), replacement)
    claim = anchor[: match.start()] + new + anchor[match.end() :]
    return Candidate(
        claim,
        PARTIAL,
        "certainty_upgrade",
        anchor,
        f'The passage says "{match.group(0).strip()}", but the claim says "{new.strip()}", '
        "which is more certain than the source.",
    )


_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_LOWERABLE_STARTS = {
    "The",
    "A",
    "An",
    "This",
    "That",
    "These",
    "Those",
    "It",
    "Its",
    "He",
    "She",
    "They",
    "His",
    "Her",
    "Their",
    "Many",
    "Some",
    "Most",
    "Several",
    "After",
    "Before",
    "During",
    "When",
    "While",
    "At",
    "By",
    "For",
    "With",
    "As",
    "Both",
    "Each",
    "Every",
    "Such",
    "Our",
    "We",
}


def added_year(anchor: str, passage: str, rng: random.Random) -> Candidate | None:
    """Prefix a year the passage never mentions, to a sentence that gives no date."""
    if _YEAR.search(anchor) or anchor.startswith(("In ", "On ", "By ")):
        return None
    present = {int(y) for y in _YEAR.findall(passage)}
    if present:
        centre = sorted(present)[len(present) // 2]
        choices = [y for y in range(centre - 20, centre + 21) if y not in present and abs(y - centre) >= 2]
    else:
        choices = list(range(1860, 1991))
    if not choices:
        return None
    year = rng.choice(choices)
    first, _, rest = anchor.partition(" ")
    lead = first.lower() if first in _LOWERABLE_STARTS else first
    claim = f"In {year}, {lead} {rest}"
    return Candidate(
        claim,
        PARTIAL,
        "added_year",
        anchor,
        f"The passage supports the claim but gives no date; {year} is not in the passages.",
    )


_PLACES = [
    "Chicago",
    "Atlanta",
    "Boston",
    "Philadelphia",
    "Baltimore",
    "Nashville",
    "Detroit",
    "New Orleans",
    "Cleveland",
    "Richmond",
    "London",
    "Paris",
    "Toronto",
    "Berlin",
]
# Any "in/at/near <Capitalised>" after the first word: the passage may already locate the event,
# and adding a different place would contradict it rather than go beyond it. Decline then.
_LOCATED = re.compile(r"\b(?:in|at|near|from|to)\s+(?:the\s+)?[A-Z][a-z]+")


def added_place(anchor: str, passage: str, rng: random.Random) -> Candidate | None:
    if _LOCATED.search(passage) or any(_in_passage(p, passage) for p in _PLACES):
        return None
    if not anchor.endswith("."):
        return None
    place = rng.choice(_PLACES)
    claim = anchor[:-1] + f", in {place}."
    return Candidate(
        claim,
        PARTIAL,
        "added_place",
        anchor,
        f"The passage supports the claim but never says it happened in {place}.",
    )


_PASSIVE = re.compile(r"\b(was|were|is|are|had been|has been|have been)\s+(?!not\b)([a-z]{3,}ed)\b")
# Qualifiers that read naturally before most passive verbs ("was unanimously elected", "was quickly
# opened"); "widely" or "universally" fit only some verbs and produced odd claims.
_QUALIFIERS = ["unanimously", "immediately", "officially", "quickly"]


def added_qualifier(anchor: str, passage: str, rng: random.Random) -> Candidate | None:
    match = _PASSIVE.search(anchor)
    if not match:
        return None
    options = [q for q in _QUALIFIERS if not _in_passage(q, passage)]
    if not options:
        return None
    qualifier = rng.choice(options)
    claim = anchor[: match.end(1)] + f" {qualifier}" + anchor[match.end(1) :]
    return Candidate(
        claim,
        PARTIAL,
        "added_qualifier",
        anchor,
        f'The passage supports the core claim but not "{qualifier}".',
    )


# --- unsupported -------------------------------------------------------------------------------

_AUX = re.compile(r"\b(is|was|were|are|has|had|have|can|could|will|would|did|does|should|must)\b(?=\s+\w)")
_NEGATED = re.compile(
    r"\b(?:(is|was|were|are|has|had|have|can|could|will|would|did|does|should|must)\s+not|never)\b\s*"
)
_NOT_BEFORE_VERB = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "by",
    "for",
    "with",
    "and",
    "or",
    "her",
    "his",
    "their",
    "its",
    "was",
    "were",
    "is",
    "are",
    "been",
    "be",
    "had",
    "has",
    "have",
    "that",
    "this",
    "which",
    "who",
    "as",
    "at",
    "from",
    "into",
    "not",
    "never",
}


def negation(anchor: str, passage: str, rng: random.Random) -> Candidate | None:
    """Say the opposite: remove a negation the source has, or add one it lacks."""
    negated = _NEGATED.search(anchor)
    if negated:
        replacement = negated.group(1) + " " if negated.group(1) else ""
        claim = anchor[: negated.start()] + replacement + anchor[negated.end() :]
        claim = claim[:1].upper() + claim[1:]
    elif re.search(r"\b(?:no|nor|n't|neither)\b", anchor, re.IGNORECASE):
        return None  # other negations are hard to reverse without changing more than intended
    else:
        aux = _AUX.search(anchor)
        if aux:
            claim = anchor[: aux.end()] + " not" + anchor[aux.end() :]
        else:
            tokens = list(re.finditer(r"\b[a-z]{3,}ed\b", anchor))
            target = next(
                (
                    t
                    for t in tokens
                    if t.start() > 0
                    and anchor[: t.start()].split()
                    and anchor[: t.start()].split()[-1].lower().strip(",") not in _NOT_BEFORE_VERB
                ),
                None,
            )
            if target is None:
                return None
            claim = anchor[: target.start()] + "never " + anchor[target.start() :]
    if normalize(claim) == normalize(anchor) or _in_passage(claim, passage):
        return None
    return Candidate(claim, UNSUPPORTED, "negation", "", "The passage says the opposite of the claim.")


def number_swap(anchor: str, passage: str, rng: random.Random) -> Candidate | None:
    found = numbers(anchor)
    if not found:
        return None
    match = rng.choice(found)
    raw = match.group(0)
    value = int(raw.replace(",", ""))
    in_passage = {int(m.group(0).replace(",", "")) for m in numbers(passage)}
    if 1500 <= value <= 2100 and len(raw) == 4:
        options = [value + d for d in range(-25, 26) if abs(d) >= 2 and value + d not in in_passage]
    elif value < 20:
        options = [
            v for v in (value + 1, value + 2, value + 3, value * 2, value + 10) if v != value and v not in in_passage
        ]
    else:
        options = [
            v
            for v in (value * 2, value // 2, value + value // 3, value - value // 4)
            if v > 0 and v != value and v not in in_passage
        ]
    if not options:
        return None
    new_value = rng.choice(options)
    new = f"{new_value:,}" if "," in raw else str(new_value)
    claim = anchor[: match.start()] + new + anchor[match.end() :]
    return Candidate(claim, UNSUPPORTED, "number_swap", "", f"The passage gives {raw}, not {new}.")


_NAME_STOP_STARTS = {
    "The",
    "A",
    "An",
    "In",
    "On",
    "At",
    "By",
    "This",
    "That",
    "These",
    "Those",
    "His",
    "Her",
    "Their",
    "Its",
    "When",
    "After",
    "Before",
    "During",
    "While",
    "As",
    "For",
    "From",
    "With",
    "But",
    "And",
    "Or",
    "It",
    "He",
    "She",
    "They",
    "We",
    "There",
}


def _clean_names(found: Sequence[str]) -> list[str]:
    return [n for n in found if n.split()[0] not in _NAME_STOP_STARTS]


def entity_swap(anchor: str, passage: str, rng: random.Random, pool: Sequence[str] = ()) -> Candidate | None:
    """Replace a name in the claim with a name from another document that the passage never mentions."""
    own = _clean_names(names(anchor))
    if not own:
        return None
    target = rng.choice(own)
    width = len(target.split())
    if not pool:
        return None
    # Test a random sample of the pool against the passage, normalised once. Checking every name in
    # the pool against a freshly normalised passage made the build quadratic in corpus size.
    names_pool = pool if isinstance(pool, list) else sorted(set(pool))
    sample = rng.sample(names_pool, k=min(len(names_pool), 64))
    seen_passage = normalize(passage)
    usable = [n for n in sample if n != target and normalize(n) not in seen_passage]
    if not usable:
        return None
    same_width = [n for n in usable if len(n.split()) == width] or usable
    new = rng.choice(same_width)
    claim = anchor.replace(target, new, 1)
    return Candidate(claim, UNSUPPORTED, "entity_swap", "", f"The passage names {target}, not {new}.")


SENTENCE_RULES: dict[str, Callable[..., Candidate | None]] = {
    "verbatim": verbatim,
    "trim_clause": trim_clause,
    "certainty_upgrade": certainty_upgrade,
    "added_year": added_year,
    "added_place": added_place,
    "added_qualifier": added_qualifier,
    "negation": negation,
    "number_swap": number_swap,
    "entity_swap": entity_swap,
}
