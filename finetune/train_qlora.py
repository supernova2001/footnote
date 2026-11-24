"""QLoRA fine-tuning of a small open model as Footnote's grounding judge.

    python train_qlora.py --config configs/qwen3-1.7b.yaml               # Kaggle/Colab T4
    python train_qlora.py --config configs/smoke-cpu.yaml                # CPU smoke run, plain LoRA

On a CUDA GPU with bitsandbytes the base model is loaded in 4-bit NF4 with double quantization and
LoRA adapters are trained on top (QLoRA). Without a GPU the same script trains plain LoRA in fp32,
which is only for checking the pipeline end to end.

Writes <output_dir>/adapter (the LoRA weights + tokenizer) and <output_dir>/train_metrics.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import time
from pathlib import Path
from typing import Any

import yaml

from footnote_ft import shared
from footnote_ft.formatting import to_prompt_completion
from footnote_ft.io import read_jsonl, write_json


def load_config(path: Path, overrides: argparse.Namespace) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides.train:
        cfg["data"]["train"] = str(overrides.train)
    if overrides.val:
        cfg["data"]["val"] = str(overrides.val)
    if overrides.output_dir:
        cfg["output_dir"] = str(overrides.output_dir)
    if overrides.max_steps is not None:
        cfg["training"]["max_steps"] = overrides.max_steps
    if overrides.push_to_hub:
        cfg.setdefault("hub", {})["push"] = True
        cfg["hub"]["repo_id"] = overrides.push_to_hub
    return cfg


def build_datasets(cfg: dict[str, Any], tokenizer: Any) -> tuple[Any, Any, dict[str, int]]:
    """Prompt-completion datasets; rows too long for max_length are dropped, never truncated.

    Truncation would cut the verdict off the end of the sequence and train the model on a prompt
    with no answer, so overlong rows are counted and left out instead.
    """
    from datasets import Dataset

    template_kwargs = cfg.get("chat_template_kwargs") or {}
    max_length = int(cfg["data"]["max_length"])
    stats: dict[str, int] = {}

    def load(path: str | None, name: str) -> Any:
        if not path or not Path(path).exists():
            return None
        records, too_long = [], 0
        for row in read_jsonl(path):
            pc = to_prompt_completion(row, tokenizer, template_kwargs)
            length = len(tokenizer(pc["prompt"] + pc["completion"], add_special_tokens=False)["input_ids"]) + 1
            if length > max_length:
                too_long += 1
                continue
            records.append(pc)
        stats[f"{name}_rows"] = len(records)
        stats[f"{name}_dropped_too_long"] = too_long
        return Dataset.from_list(records) if records else None

    train = load(cfg["data"]["train"], "train")
    if train is None:
        raise SystemExit(f"no training rows in {cfg['data']['train']}: run build_dataset.py first")
    return train, load(cfg["data"].get("val"), "val"), stats


def _known_fields(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in values.items() if k in names}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--push-to-hub", metavar="REPO_ID", help="push the adapter to this Hugging Face repo")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args)

    import peft
    import torch
    import transformers
    import trl
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cuda = torch.cuda.is_available()
    quant = cfg.get("quantization") or {}
    use_4bit = bool(quant.get("load_in_4bit")) and cuda
    if quant.get("load_in_4bit") and not cuda:
        print("No CUDA GPU: 4-bit QLoRA needs one, so this run trains plain LoRA in fp32 (smoke run only).")
    if cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)),
            bnb_4bit_compute_dtype=dtype,
        )
    if cuda:
        model_kwargs["device_map"] = {"": 0}  # one GPU; Kaggle's second T4 is not used
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **model_kwargs)
    tcfg = cfg["training"]
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True))
        )

    train_ds, val_ds, data_stats = build_datasets(cfg, tokenizer)

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=lora.get("target_modules", "all-linear"),
        bias="none",
        task_type="CAUSAL_LM",
    )

    output_dir = Path(cfg["output_dir"])
    sft_values: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": float(tcfg.get("epochs", 2)),
        "max_steps": int(tcfg.get("max_steps", -1)),
        "per_device_train_batch_size": int(tcfg.get("per_device_batch_size", 4)),
        "per_device_eval_batch_size": int(tcfg.get("per_device_batch_size", 4)),
        "gradient_accumulation_steps": int(tcfg.get("gradient_accumulation", 4)),
        "learning_rate": float(tcfg.get("learning_rate", 2e-4)),
        "lr_scheduler_type": tcfg.get("lr_scheduler", "cosine"),
        "warmup_steps": float(tcfg.get("warmup_ratio", 0.05)),  # a float < 1 is a ratio of total steps
        "weight_decay": float(tcfg.get("weight_decay", 0.0)),
        "logging_steps": int(tcfg.get("logging_steps", 10)),
        "eval_strategy": "steps" if val_ds is not None else "no",
        "eval_steps": int(tcfg.get("eval_steps", 100)),
        "save_strategy": "steps",
        "save_steps": int(tcfg.get("save_steps", 200)),
        "save_total_limit": 2,
        "gradient_checkpointing": bool(tcfg.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "fp16": cuda and dtype == torch.float16,
        "bf16": cuda and dtype == torch.bfloat16,
        "max_length": int(cfg["data"]["max_length"]),
        "completion_only_loss": True,  # loss on the verdict only, not on the passages
        "packing": False,
        "seed": int(tcfg.get("seed", 13)),
        "report_to": tcfg.get("report_to", "none"),
        "dataloader_pin_memory": cuda,
        "optim": tcfg.get("optim", "paged_adamw_8bit" if use_4bit else "adamw_torch"),
    }
    sft_config = SFTConfig(**_known_fields(SFTConfig, sft_values))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.3f}%)")

    started = time.time()
    train_result = trainer.train()
    elapsed = time.time() - started
    eval_metrics = trainer.evaluate() if val_ds is not None else {}

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "model": cfg["model"],
        "prompt_version": shared.judge_prompt().PROMPT_VERSION,
        "quantization": "4bit-nf4-double" if use_4bit else "none",
        "device": torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu",
        "train_seconds": round(elapsed, 1),
        "train": train_result.metrics,
        "eval": eval_metrics,
        "data": data_stats,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "config": cfg,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
        },
        "log_history": trainer.state.log_history,
    }
    write_json(output_dir / "train_metrics.json", json.loads(json.dumps(metrics, default=str)))
    print(f"adapter -> {adapter_dir}\nmetrics -> {output_dir / 'train_metrics.json'}")

    hub = cfg.get("hub") or {}
    if hub.get("push") and hub.get("repo_id"):
        trainer.model.push_to_hub(hub["repo_id"], private=bool(hub.get("private", True)))
        tokenizer.push_to_hub(hub["repo_id"], private=bool(hub.get("private", True)))
        print(f"pushed adapter to https://huggingface.co/{hub['repo_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
