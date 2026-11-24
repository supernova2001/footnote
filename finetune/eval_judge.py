"""Score a judge on the test set: base model zero-shot, the fine-tuned adapter, or a served model.

    # base model, zero-shot
    python eval_judge.py --name base:qwen3-1.7b --model Qwen/Qwen3-1.7B
    # the fine-tuned adapter (same prompt, same template)
    python eval_judge.py --name qlora:qwen3-1.7b --model Qwen/Qwen3-1.7B --adapter outputs/qwen3-1.7b-judge/adapter
    # any OpenAI-compatible server, e.g. the GGUF served by Ollama
    python eval_judge.py --name local:footnote-judge --openai-base-url http://localhost:11434/v1 --openai-model footnote-judge

Gold labels: when --gold points at a test_gold.jsonl with human labels, those rows and labels are
the test set ("label_source": "human"); otherwise the constructed labels of --data are used.
Metrics come from the backend's judge_metrics.py, the same code that scores the Claude judge.
Writes results/<name>.json (full report + every reply) and updates results/judge_rows.json (the
`judge` rows of the API's EvalReport).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from footnote_ft import shared
from footnote_ft.formatting import render_prompt, system_user
from footnote_ft.io import read_jsonl, write_json


def load_cases(data: Path, gold: Path | None, limit: int | None) -> tuple[list[dict[str, Any]], str]:
    if gold and gold.exists():
        labelled = [r for r in read_jsonl(gold) if r.get("gold_label")]
        if labelled:
            rows = [{**r, "label": r["gold_label"]} for r in labelled]
            return (rows[:limit] if limit else rows), "human"
    rows = list(read_jsonl(data))
    return (rows[:limit] if limit else rows), "constructed"


class HFJudge:
    """Greedy generation with transformers; an optional LoRA adapter on top of the base model."""

    def __init__(
        self, model: str, adapter: str | None, load_in_4bit: bool, template_kwargs: dict[str, Any], max_new_tokens: int
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        cuda = torch.cuda.is_available()
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
        kwargs: dict[str, Any] = {"dtype": dtype}
        if cuda:
            kwargs["device_map"] = {"": 0}
        if load_in_4bit and cuda:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.template_kwargs = template_kwargs
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompts: list[tuple[str, str]]) -> list[str]:
        texts = [render_prompt(self.tokenizer, s, u, self.template_kwargs) for s, u in prompts]
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[:, batch["input_ids"].shape[1] :]
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in new_tokens]


class OpenAIJudge:
    """Any OpenAI-compatible chat endpoint (Ollama, llama.cpp, vLLM), standard library only."""

    def __init__(self, base_url: str, model: str, json_schema: bool, max_tokens: int) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.json_schema = json_schema
        self.max_tokens = max_tokens

    def _one(self, system: str, user: str) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.json_schema:
            schema = shared.judge_prompt().JUDGE_OUTPUT_SCHEMA
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": schema, "strict": True},
            }
        request = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode())
        return str(payload["choices"][0]["message"]["content"] or "")

    def __call__(self, prompts: list[tuple[str, str]]) -> list[str]:
        return [self._one(s, u) for s, u in prompts]


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help='judge name in the results, e.g. "qlora:qwen3-1.7b"')
    p.add_argument("--data", type=Path, default=Path("data/out/test.jsonl"))
    p.add_argument("--gold", type=Path, default=Path("data/out/test_gold.jsonl"))
    p.add_argument("--model", help="Hugging Face model id or path")
    p.add_argument("--adapter", help="LoRA adapter directory")
    p.add_argument("--load-in-4bit", action="store_true", help="evaluate a 4-bit base, as trained (CUDA only)")
    p.add_argument(
        "--enable-thinking", action="store_true", help="leave Qwen3 thinking on (off by default, as trained)"
    )
    p.add_argument("--openai-base-url")
    p.add_argument("--openai-model")
    p.add_argument("--json-schema", action="store_true", help="ask the server for schema-constrained JSON")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int)
    p.add_argument("--cost-per-1k", type=float, default=0.0, help="USD per 1,000 judgments, for the table")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    metrics = shared.judge_metrics()
    rows, label_source = load_cases(args.data, args.gold, args.limit)
    if not rows:
        raise SystemExit(f"no test rows in {args.data}")

    if args.openai_base_url:
        if not args.openai_model:
            p.error("--openai-base-url needs --openai-model")
        judge: Any = OpenAIJudge(args.openai_base_url, args.openai_model, args.json_schema, args.max_new_tokens)
        batch_size = 1
    elif args.model:
        template_kwargs = {} if args.enable_thinking else {"enable_thinking": False}
        judge = HFJudge(args.model, args.adapter, args.load_in_4bit, template_kwargs, args.max_new_tokens)
        batch_size = max(1, args.batch_size)
    else:
        p.error("give --model (with optional --adapter) or --openai-base-url")

    cases, predictions = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [system_user(r) for r in batch]
        began = time.perf_counter()
        replies = judge(prompts)
        per_item_ms = (time.perf_counter() - began) * 1000 / len(batch)
        for row, reply in zip(batch, replies, strict=True):
            reply = _THINK.sub("", reply).strip()
            cases.append(
                metrics.JudgeCase(
                    gold=row["label"], reply=reply, passages=row["passages"], latency_ms=per_item_ms, id=row["id"]
                )
            )
            predictions.append(
                {
                    "id": row["id"],
                    "gold": row["label"],
                    "method": row.get("method"),
                    "reply": reply,
                    "latency_ms": round(per_item_ms, 1),
                }
            )
        print(f"{min(start + batch_size, len(rows))}/{len(rows)}", end="\r", flush=True)
    print()

    report = metrics.score_judge(cases)
    row = metrics.eval_row(report, args.name, args.cost_per_1k)
    by_method: dict[str, dict[str, int]] = {}
    for pred, case in zip(predictions, cases, strict=True):
        stats = by_method.setdefault(str(pred["method"]), {"n": 0, "correct": 0})
        stats["n"] += 1
        stats["correct"] += int(metrics.predicted_label(case) == case.gold)

    result = {
        "name": args.name,
        "model": args.model or args.openai_model,
        "adapter": args.adapter,
        "label_source": label_source,
        "prompt_version": shared.judge_prompt().PROMPT_VERSION,
        "batch_size": batch_size,
        "latency_note": "per-item latency is batch wall time / batch size" if batch_size > 1 else "per-item wall time",
        "row": row,
        "report": report.to_dict(),
        "accuracy_by_method": by_method,
        "predictions": predictions,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", args.name)
    write_json(args.results_dir / f"{safe}.json", result)

    rows_path = args.results_dir / "judge_rows.json"
    existing = json.loads(rows_path.read_text(encoding="utf-8")) if rows_path.exists() else []
    existing = [r for r in existing if r.get("judge") != args.name] + [row]
    write_json(rows_path, existing)

    print(json.dumps(row, indent=2))
    print(
        f"accuracy={report.accuracy:.3f}  deployed_macro_f1={report.deployed_macro_f1:.3f}  "
        f"evidence_verbatim={report.evidence_verbatim_rate}  labels={label_source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
