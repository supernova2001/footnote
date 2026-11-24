from __future__ import annotations

import json
from collections import Counter

from footnote_ft.dataset import (
    SPLITS,
    BuildConfig,
    assign_splits,
    balance,
    build_examples,
    gold_sample,
    recorded_examples,
    to_row,
)
from footnote_ft.io import read_jsonl, write_jsonl
from footnote_ft.text import normalize


def test_labels_are_balanced_in_every_split(rows):
    for split in SPLITS:
        counts = Counter(r["label"] for r in rows if r["split"] == split)
        assert set(counts) == {"supported", "partial", "unsupported"}, (split, counts)
        assert len(set(counts.values())) == 1, (split, counts)


def test_balance_spreads_each_label_across_methods(rows):
    train = [r for r in rows if r["split"] == "train"]
    partial = Counter(r["method"] for r in train if r["label"] == "partial")
    assert len(partial) >= 3 and max(partial.values()) - min(partial.values()) <= 2 + min(partial.values())


def test_splits_are_by_document_with_no_leakage(rows, doc_split):
    docs_by_split = {s: {r["document_id"] for r in rows if r["split"] == s} for s in SPLITS}
    assert not docs_by_split["train"] & docs_by_split["test"]
    assert not docs_by_split["train"] & docs_by_split["val"]
    assert not docs_by_split["val"] & docs_by_split["test"]
    for row in rows:
        # every passage shown comes from a document of the row's own split
        for pid in row["passage_ids"]:
            assert doc_split[pid.rsplit(":", 1)[0]] == row["split"]
    train_text = normalize(" ".join(p["text"] for r in rows if r["split"] == "train" for p in r["passages"]))
    for row in rows:
        if row["split"] == "test":
            assert normalize(row["claim"]) not in train_text, row["claim"]


def test_assign_splits_is_deterministic_and_fills_every_split():
    docs = [f"doc-{i}" for i in range(9)]
    a = assign_splits(docs, (0.8, 0.1, 0.1), seed=13)
    assert a == assign_splits(reversed(docs), (0.8, 0.1, 0.1), seed=13)
    assert set(a.values()) == set(SPLITS)
    assert set(assign_splits(["x", "y"], (0.8, 0.1, 0.1), seed=1).values()) == {"train"}


def test_build_is_reproducible(chunks, doc_split, rows):
    examples, _ = build_examples(chunks, doc_split, BuildConfig(seed=13))
    again = [to_row(ex) for ex in balance(examples, seed=13)]
    assert [r["id"] for r in again] == [r["id"] for r in rows]


def test_recorded_judgments_use_only_human_labels(doc_split):
    test_doc = next(d for d, s in doc_split.items() if s == "test")
    train_doc = next(d for d, s in doc_split.items() if s == "train")
    passage = {"id": f"{test_doc}:0000", "title": "T", "text": "The clinic opened in 1897 with two beds."}
    judgments = [
        {
            "claim": "The clinic opened in 1897.",
            "passages": [passage],
            "label": "supported",
            "evidence": "The clinic opened in 1897",
            "rationale": "Claude's words.",
            "human_label": "supported",
        },
        {
            "claim": "The clinic opened in 1897.",
            "passages": [passage],
            "label": "supported",
            "evidence": "x",
            "rationale": "r",
        },
        {
            "claim": "Mixed.",
            "passages": [passage, {"id": f"{train_doc}:0001", "title": "T", "text": "Other."}],
            "human_label": "unsupported",
        },
        {
            "claim": "It had two beds.",
            "passages": [passage],
            "evidence": "not in the passage",
            "human_label": "supported",
        },
    ]
    kept, dropped = recorded_examples(judgments, doc_split)
    assert len(kept) == 1
    assert kept[0].split == "test" and kept[0].label == "supported"
    assert kept[0].rationale != "Claude's words."
    assert dropped == {"recorded_unreviewed": 1, "recorded_spans_splits": 1, "recorded_evidence_not_verbatim": 1}


def test_gold_sample_is_stratified_and_unlabelled(rows):
    test_rows = [r for r in rows if r["split"] == "test"]
    gold = gold_sample(test_rows, size=6, seed=13)
    assert Counter(g["constructed_label"] for g in gold) == {"supported": 2, "partial": 2, "unsupported": 2}
    assert all(g["gold_label"] is None and g["passages"] for g in gold)


def test_cli_never_overwrites_human_labels(tmp_path, chunks):
    import build_dataset

    src = tmp_path / "chunks.jsonl"
    write_jsonl(src, chunks)
    out = tmp_path / "out"
    assert build_dataset.main(["--chunks", str(src), "--out", str(out), "--rules-only", "--gold-size", "6"]) == 0
    gold = list(read_jsonl(out / "test_gold.jsonl"))
    gold[0]["gold_label"] = "partial"
    write_jsonl(out / "test_gold.jsonl", gold)
    assert build_dataset.main(["--chunks", str(src), "--out", str(out), "--rules-only", "--gold-size", "6"]) == 0
    assert next(read_jsonl(out / "test_gold.jsonl"))["gold_label"] == "partial"
    assert (out / "test_gold.new.jsonl").exists()
    report = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
    assert report["prompt_version"] and report["splits"]["train"]["rows"] > 0
