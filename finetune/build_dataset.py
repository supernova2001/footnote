"""Build the grounding judge's train/val/test data from the corpus's chunks.

    # dry run, rules only (no model needed)
    python build_dataset.py --chunks data/chunks.jsonl --out data/out --rules-only

    # real run: rules + paraphrases from an open-weight model behind any OpenAI-compatible server
    python build_dataset.py --chunks data/chunks.jsonl --out data/out \
        --generator openai --base-url http://localhost:8001/v1 --model Qwen/Qwen2.5-7B-Instruct

    # real run on a Kaggle/Colab GPU with vLLM's offline engine
    python build_dataset.py --chunks data/chunks.jsonl --out data/out --generator vllm --model Qwen/Qwen2.5-7B-Instruct

Writes train.jsonl, val.jsonl, test.jsonl, test_gold.jsonl (for label_gold.py) and build_report.json.
An existing test_gold.jsonl that already has human labels is never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from footnote_ft.dataset import (
    SPLITS,
    BuildConfig,
    assign_splits,
    balance,
    build_examples,
    gold_sample,
    load_chunks,
    recorded_examples,
    report,
    to_row,
)
from footnote_ft.io import read_jsonl, write_json, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks", required=True, type=Path, help="chunks JSONL exported by the backend")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--judgments", type=Path, help="recorded judgments JSONL; only human-labelled rows are used")
    p.add_argument("--rules-only", action="store_true", help="rule-based claims only (dry runs, CI)")
    p.add_argument("--generator", choices=["openai", "vllm"], help="model-written claims from an open-weight model")
    p.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://localhost:8001/v1")
    p.add_argument("--model", help="generator model id")
    p.add_argument("--no-rules", action="store_true", help="with --generator: model-written claims only")
    p.add_argument("--thinking", action="store_true", help="leave a Qwen3 generator's thinking on (off by default)")
    p.add_argument("--vllm-dtype", default="auto", help='vLLM dtype; "half" on a T4, which has no bfloat16')
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--anchors-per-chunk", type=int, default=2)
    p.add_argument("--max-distractors", type=int, default=2)
    p.add_argument("--gold-size", type=int, default=200)
    p.add_argument("--max-chunks", type=int, help="use only the first N chunks (quick runs)")
    args = p.parse_args(argv)
    if not args.rules_only and not args.generator:
        p.error("choose --rules-only for a dry run, or --generator openai|vllm --model ... for real training data")
    if args.generator and not args.model:
        p.error("--generator needs --model")
    if args.generator == "openai" and not args.base_url:
        p.error("--generator openai needs --base-url")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = BuildConfig(
        seed=args.seed,
        anchors_per_chunk=args.anchors_per_chunk,
        max_distractors=args.max_distractors,
        gold_size=args.gold_size,
    )
    chunks = load_chunks(read_jsonl(args.chunks))
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    doc_split = assign_splits((str(c["document_id"]) for c in chunks), cfg.ratios, cfg.seed)

    generator = None
    template_kwargs = None if args.thinking else {"enable_thinking": False}
    if args.generator == "openai":
        from footnote_ft.generate import OpenAICompatibleGenerator

        generator = OpenAICompatibleGenerator(args.base_url, args.model, chat_template_kwargs=template_kwargs)
    elif args.generator == "vllm":
        from footnote_ft.generate import VLLMGenerator

        generator = VLLMGenerator(
            args.model,
            chat_template_kwargs=template_kwargs,
            dtype=args.vllm_dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    examples, rejected = build_examples(chunks, doc_split, cfg, generator=generator, rules=not args.no_rules)
    if args.judgments:
        recorded, dropped = recorded_examples(read_jsonl(args.judgments), doc_split)
        examples.extend(recorded)
        rejected.update(dropped)

    balanced = balance(examples, cfg.seed)
    rows = [to_row(ex) for ex in balanced]
    rejected.update({"unbalanced_surplus": len(examples) - len(balanced)})

    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(args.out / f"{split}.jsonl", (r for r in rows if r["split"] == split))

    gold_path = args.out / "test_gold.jsonl"
    gold = gold_sample([r for r in rows if r["split"] == "test"], cfg.gold_size, cfg.seed)
    if gold_path.exists() and any(r.get("gold_label") for r in read_jsonl(gold_path)):
        gold_path = args.out / "test_gold.new.jsonl"
        print(f"test_gold.jsonl already has human labels; wrote the new sample to {gold_path.name}", file=sys.stderr)
    write_jsonl(gold_path, gold)

    summary = report(rows, doc_split, rejected, cfg)
    summary["gold_sample"] = {
        "path": gold_path.name,
        "rows": len(gold),
        "labels": dict(Counter(g["constructed_label"] for g in gold)),
    }
    write_json(args.out / "build_report.json", summary)

    for split in SPLITS:
        s = summary["splits"][split]
        print(f"{split:5}  rows={s['rows']:5}  docs={len(s['documents']):3}  labels={s['labels']}")
    print(f"gold sample: {len(gold)} rows -> {gold_path}")
    if not rows:
        print(
            "no rows: every label needs at least one example per split (see build_report.json 'rejected')",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
