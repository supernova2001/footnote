"""Build the judge's training data from the corpus's chunks.

Pipeline: chunks → split documents (train/val/test) → for each chunk, pick anchor sentences and the
passages a judge would see (the chunk plus 0-2 neighbouring chunks) → candidate claims from rules,
from other passages (cross-document, outside-fact) and optionally from a model → verify each label
against the passages → balance labels per split → chat-format rows built with the backend's
`render_judge_prompt`.

Splits are by document, and every negative that borrows text from another document borrows it from
the same split, so no test passage or claim text is ever seen in training.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import shared
from .corruptions import PARTIAL, SENTENCE_RULES, SUPPORTED, UNSUPPORTED, Candidate, _clean_names
from .generate import GENERATION_VERSION, Generator, build_generation_prompt, parse_generation, validate_generated
from .text import coverage, is_good_anchor, names, normalize, sentence_spans

LABELS = (SUPPORTED, PARTIAL, UNSUPPORTED)
SPLITS = ("train", "val", "test")
REQUIRED_CHUNK_FIELDS = ("id", "document_id", "ordinal", "title", "text")


@dataclass
class BuildConfig:
    seed: int = 13
    anchors_per_chunk: int = 2
    max_distractors: int = 2
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    gold_size: int = 200
    max_foreign_coverage: float = 0.34  # a borrowed claim must share at most this much with the passages
    borrowed_per_chunk: int = 1  # cross-document and outside-fact negatives per chunk, each


@dataclass
class Example:
    claim: str
    label: str
    method: str
    evidence: str
    rationale: str
    passages: list[dict[str, str]]
    document_id: str
    split: str
    source: str
    anchor_chunk_id: str
    extra: dict[str, Any] = field(default_factory=dict)


def load_chunks(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    chunks = []
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED_CHUNK_FIELDS if k not in row]
        if missing:
            raise ValueError(f"chunk {i} is missing {missing}; expected the backend's Chunk shape")
        chunks.append(dict(row))
    if not chunks:
        raise ValueError("no chunks to build from")
    return chunks


def assign_splits(document_ids: Iterable[str], ratios: Sequence[float], seed: int) -> dict[str, str]:
    """Deterministic split of whole documents. With 3+ documents every split gets at least one."""
    ordered = sorted(set(document_ids), key=lambda d: hashlib.sha256(f"{seed}:{d}".encode()).hexdigest())
    n = len(ordered)
    if n < 3:
        return {d: "train" for d in ordered}
    n_test = max(1, round(n * ratios[2]))
    n_val = max(1, round(n * ratios[1]))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    split = {}
    for i, doc in enumerate(ordered):
        split[doc] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return split


def _passage(chunk: Mapping[str, Any]) -> dict[str, str]:
    passage = {"id": str(chunk["id"]), "title": str(chunk["title"]), "text": str(chunk["text"])}
    if chunk.get("attribution"):
        # judge.v2: the judge sees who wrote each passage, so training prompts must too.
        passage["attribution"] = str(chunk["attribution"])
    return passage


def _joined(passages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(p["text"] for p in passages)


def _verified(candidate: Candidate, passages: Sequence[Mapping[str, str]]) -> bool:
    """The label holds against the exact passages the judge will see."""
    jp = shared.judge_prompt()
    text = normalize(_joined(passages))
    if candidate.label == UNSUPPORTED:
        return candidate.evidence == "" and normalize(candidate.claim) not in text
    if not jp.evidence_is_verbatim(candidate.evidence, passages):
        return False
    if candidate.label == PARTIAL:
        return normalize(candidate.claim) not in text
    return True


def build_examples(
    chunks: Sequence[Mapping[str, Any]],
    doc_split: Mapping[str, str],
    cfg: BuildConfig,
    generator: Generator | None = None,
    rules: bool = True,
) -> tuple[list[Example], Counter[str]]:
    rng = random.Random(cfg.seed)
    rejected: Counter[str] = Counter()
    by_doc: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_doc[str(chunk["document_id"])].append(chunk)
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda c: int(c["ordinal"]))

    anchors: dict[str, list[str]] = {}
    for chunk in chunks:
        anchors[str(chunk["id"])] = [s.text for s in sentence_spans(str(chunk["text"])) if is_good_anchor(s.text)]

    # Names and borrowable sentences per split, from other documents only (see entity_swap).
    names_by_doc = {d: set(_clean_names([n for c in cs for n in names(str(c["text"]))])) for d, cs in by_doc.items()}
    docs_in_split: dict[str, list[str]] = defaultdict(list)
    for doc, split in doc_split.items():
        if doc in by_doc:
            docs_in_split[split].append(doc)
    for docs in docs_in_split.values():
        docs.sort()

    examples: list[Example] = []
    prompts: list[tuple[str, str]] = []
    pending: list[tuple[str, _Context]] = []

    for doc, doc_chunks in sorted(by_doc.items()):
        split = doc_split[doc]
        others = [d for d in docs_in_split[split] if d != doc]
        name_pool = sorted({n for d in others for n in names_by_doc[d]})
        for position, chunk in enumerate(doc_chunks):
            sentences = anchors[str(chunk["id"])]
            if not sentences:
                rejected["no_anchor"] += 1
                continue
            neighbours = [c for c in doc_chunks[max(0, position - 2) : position + 3] if c is not chunk]
            distractors = rng.sample(neighbours, k=min(len(neighbours), rng.randint(0, cfg.max_distractors)))
            passages = [_passage(chunk), *(_passage(c) for c in distractors)]
            rng.shuffle(passages)
            passage_text = _joined(passages)
            shown = {p["id"] for p in passages}
            context = (passages, doc, split, str(chunk["id"]))

            candidates: list[Candidate | None] = []
            for anchor in rng.sample(sentences, k=min(cfg.anchors_per_chunk, len(sentences))):
                if rules:
                    for name, rule in SENTENCE_RULES.items():
                        if name == "entity_swap":
                            candidates.append(rule(anchor, passage_text, rng, pool=name_pool))
                        else:
                            candidates.append(rule(anchor, passage_text, rng))
                if generator is not None:
                    prompts.append(build_generation_prompt(anchor, str(chunk["text"])))
                    pending.append((anchor, context))
            if rules:
                # Cross-document: a sentence from another document of the same split, about something else.
                candidates.append(
                    _borrowed(
                        rng,
                        others,
                        by_doc,
                        anchors,
                        passages,
                        cfg,
                        "cross_document",
                        "The passages are about something else and do not mention this.",
                    )
                )
                # Outside fact: true elsewhere in the same document, but not in the passages shown.
                same_doc = [c for c in doc_chunks if str(c["id"]) not in shown]
                candidates.append(
                    _borrowed_from_chunks(
                        rng,
                        same_doc,
                        anchors,
                        passages,
                        cfg,
                        "outside_fact",
                        "The passages do not mention this, even if it may be true elsewhere.",
                    )
                )
            for candidate in candidates:
                _keep(candidate, context, "rules", {}, examples, rejected)

    if generator is not None and prompts:
        replies = generator(prompts)
        for reply, (anchor, context) in zip(replies, pending, strict=True):
            data = parse_generation(reply)
            if data is None:
                rejected["model_unparsable"] += 1
                continue
            for candidate in validate_generated(data, anchor, _joined(context[0])):
                _keep(
                    candidate, context, generator.name, {"generation_version": GENERATION_VERSION}, examples, rejected
                )
    return _dedupe(examples), rejected


_Context = tuple[list[dict[str, str]], str, str, str]  # passages, document id, split, anchor chunk id


def _keep(
    candidate: Candidate | None,
    context: _Context,
    source: str,
    extra: dict[str, Any],
    examples: list[Example],
    rejected: Counter[str],
) -> None:
    if candidate is None:
        return
    passages, doc, split, chunk_id = context
    if not _verified(candidate, passages):
        rejected[f"unverified:{candidate.method}"] += 1
        return
    examples.append(
        Example(
            claim=candidate.claim,
            label=candidate.label,
            method=candidate.method,
            evidence=candidate.evidence,
            rationale=candidate.rationale,
            passages=passages,
            document_id=doc,
            split=split,
            source=source,
            anchor_chunk_id=chunk_id,
            extra=dict(extra),
        )
    )


def _borrowed(
    rng: random.Random,
    other_docs: Sequence[str],
    by_doc: Mapping[str, Sequence[Mapping[str, Any]]],
    anchors: Mapping[str, list[str]],
    passages: Sequence[Mapping[str, str]],
    cfg: BuildConfig,
    method: str,
    rationale: str,
) -> Candidate | None:
    if not other_docs:
        return None
    for _ in range(cfg.borrowed_per_chunk * 4):
        doc = rng.choice(list(other_docs))
        candidate = _borrowed_from_chunks(rng, by_doc[doc], anchors, passages, cfg, method, rationale)
        if candidate is not None:
            return candidate
    return None


def _borrowed_from_chunks(
    rng: random.Random,
    chunks: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, list[str]],
    passages: Sequence[Mapping[str, str]],
    cfg: BuildConfig,
    method: str,
    rationale: str,
) -> Candidate | None:
    # Sample a few chunks first, then their sentences: building every sentence of a book for each
    # anchor chunk made the build quadratic in document length (minutes on a 1,000-chunk book).
    if not chunks:
        return None
    picked = rng.sample(chunks, k=min(len(chunks), 8)) if isinstance(chunks, list) else list(chunks)[:8]
    pool = [s for c in picked for s in anchors[str(c["id"])]]
    rng.shuffle(pool)
    text = _joined(passages)
    for sentence in pool[:20]:
        if coverage(sentence, text) <= cfg.max_foreign_coverage:
            return Candidate(sentence, UNSUPPORTED, method, "", rationale)
    return None


def _dedupe(examples: list[Example]) -> list[Example]:
    seen: set[tuple[str, str, str]] = set()
    unique = []
    for ex in examples:
        key = (normalize(ex.claim), ex.label, ex.anchor_chunk_id)
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def recorded_examples(
    judgments: Iterable[Mapping[str, Any]], doc_split: Mapping[str, str]
) -> tuple[list[Example], Counter[str]]:
    """Human-reviewed judgments recorded by the backend. Only the human label is used as a target.

    The judge model's own label and rationale are never training targets (they are a closed
    model's output); the rationale is replaced with a neutral sentence per label.
    """
    jp = shared.judge_prompt()
    neutral = {
        SUPPORTED: "The passage states this.",
        PARTIAL: "The passage supports the core of the claim but not all of it.",
        UNSUPPORTED: "The passages do not support this claim.",
    }
    kept: list[Example] = []
    dropped: Counter[str] = Counter()
    for j in judgments:
        label = j.get("human_label")
        if label not in LABELS:
            dropped["recorded_unreviewed"] += 1
            continue
        passages = [
            {"id": str(p["id"]), "title": str(p.get("title", "")), "text": str(p["text"])}
            for p in j.get("passages", [])
        ]
        if not passages or not str(j.get("claim", "")).strip():
            dropped["recorded_incomplete"] += 1
            continue
        docs = {p["id"].rsplit(":", 1)[0] for p in passages}
        splits = {doc_split.get(d, "train") for d in docs}
        if len(splits) != 1:
            dropped["recorded_spans_splits"] += 1
            continue
        evidence = str(j.get("evidence", "")) if label != UNSUPPORTED else ""
        if label != UNSUPPORTED and not jp.evidence_is_verbatim(evidence, passages):
            dropped["recorded_evidence_not_verbatim"] += 1
            continue
        kept.append(
            Example(
                claim=str(j["claim"]).strip(),
                label=label,
                method="recorded",
                evidence=evidence,
                rationale=neutral[label],
                passages=passages,
                document_id=sorted(docs)[0],
                split=splits.pop(),
                source="recorded",
                anchor_chunk_id=passages[0]["id"],
            )
        )
    return kept, dropped


def balance(examples: Sequence[Example], seed: int) -> list[Example]:
    """Equal counts per label within each split, spread evenly across the methods of each label."""
    rng = random.Random(seed)
    out: list[Example] = []
    for split in SPLITS:
        by_label: dict[str, dict[str, list[Example]]] = {lab: defaultdict(list) for lab in LABELS}
        for ex in examples:
            if ex.split == split:
                by_label[ex.label][ex.method].append(ex)
        counts = {lab: sum(len(v) for v in methods.values()) for lab, methods in by_label.items()}
        target = min(counts.values())
        for methods in by_label.values():
            queues = []
            for method in sorted(methods):
                items = list(methods[method])
                rng.shuffle(items)
                queues.append(items)
            picked: list[Example] = []
            while len(picked) < target:
                for queue in queues:
                    if queue and len(picked) < target:
                        picked.append(queue.pop())
            out.extend(picked)
    return out


def to_row(ex: Example) -> dict[str, Any]:
    jp = shared.judge_prompt()
    system, user = jp.render_judge_prompt(ex.claim, ex.passages)
    target = json.dumps({"label": ex.label, "evidence": ex.evidence, "rationale": ex.rationale}, ensure_ascii=False)
    passage_ids = [p["id"] for p in ex.passages]
    digest = hashlib.sha1(f"{ex.split}|{ex.label}|{ex.method}|{ex.claim}|{','.join(passage_ids)}".encode()).hexdigest()
    return {
        "id": digest[:16],
        "prompt_version": jp.PROMPT_VERSION,
        "split": ex.split,
        "label": ex.label,
        "method": ex.method,
        "source": ex.source,
        "document_id": ex.document_id,
        "passage_ids": passage_ids,
        "claim": ex.claim,
        "evidence": ex.evidence,
        "passages": ex.passages,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        **ex.extra,
    }


def gold_sample(test_rows: Sequence[Mapping[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """A label-stratified sample of the test split for a person to label (label_gold.py)."""
    rng = random.Random(seed + 1)
    per_label = max(1, size // len(LABELS))
    sample: list[dict[str, Any]] = []
    for label in LABELS:
        rows = [r for r in test_rows if r["label"] == label]
        for row in rng.sample(rows, k=min(per_label, len(rows))):
            sample.append(
                {
                    "id": row["id"],
                    "prompt_version": row["prompt_version"],
                    "document_id": row["document_id"],
                    "claim": row["claim"],
                    "passages": row["passages"],
                    "constructed_label": row["label"],
                    "method": row["method"],
                    "gold_label": None,
                    "note": "",
                }
            )
    rng.shuffle(sample)
    return sample


def report(
    rows: Sequence[Mapping[str, Any]], doc_split: Mapping[str, str], rejected: Counter[str], cfg: BuildConfig
) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in SPLITS:
        split_rows = [r for r in rows if r["split"] == split]
        by_split[split] = {
            "rows": len(split_rows),
            "documents": sorted({d for d, s in doc_split.items() if s == split}),
            "labels": dict(Counter(r["label"] for r in split_rows)),
            "methods": dict(sorted(Counter(r["method"] for r in split_rows).items())),
            "sources": dict(Counter(r["source"] for r in split_rows)),
        }
    return {
        "prompt_version": shared.judge_prompt().PROMPT_VERSION,
        "config": {
            "seed": cfg.seed,
            "anchors_per_chunk": cfg.anchors_per_chunk,
            "max_distractors": cfg.max_distractors,
            "ratios": list(cfg.ratios),
            "max_foreign_coverage": cfg.max_foreign_coverage,
        },
        "splits": by_split,
        "rejected": dict(sorted(rejected.items())),
    }
