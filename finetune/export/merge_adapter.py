"""Merge the LoRA adapter into the base model's weights, ready for GGUF conversion.

    python export/merge_adapter.py --base Qwen/Qwen3-1.7B --adapter outputs/qwen3-1.7b-judge/adapter --out outputs/merged

The base is loaded in fp16 (not 4-bit: a merge needs real weights), the adapter applied and folded
in, and the result saved as safetensors with the adapter's tokenizer. Qwen3-1.7B needs about 4 GB of
RAM for this; no GPU is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = p.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=getattr(torch, args.dtype))
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(args.adapter)).save_pretrained(str(args.out))
    print(f"merged model -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
