"""Writes kaggle_train.ipynb. Edit the cells here, then run: python notebooks/build_notebook.py"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

CELLS = [
    new_markdown_cell(
        "# Footnote: fine-tune the grounding judge with QLoRA (Kaggle T4)\n\n"
        "Build data → train Qwen3-1.7B with QLoRA → compare base vs fine-tuned → export a GGUF for Ollama.\n\n"
        "**Before running:** Settings → Accelerator **GPU T4 x1**, Internet **on**. Add the corpus export "
        "(`chunks.jsonl`, written by the backend) as a Kaggle dataset and set `CHUNKS` below. "
        "Optional: add a Kaggle secret `HF_TOKEN` to push the adapter to the Hugging Face Hub.\n\n"
        "Rough timings on one T4: data generation with Qwen3-4B 15-30 min for ~2k anchors; training "
        "3 epochs on ~6k rows 1.5-3 h; each evaluation of 200-600 rows 5-15 min; GGUF export 10 min. "
        "These are estimates, not measurements: the cells print the real numbers."
    ),
    new_code_cell(
        "import os\n"
        'REPO_URL = "https://github.com/YOUR-USER/footnote.git"   # edit: your clone of the repo\n'
        'CHUNKS = "/kaggle/input/footnote-chunks/chunks.jsonl"    # edit: the backend export\n'
        'GENERATOR = "Qwen/Qwen3-4B"   # open-weight (Apache-2.0) model that writes paraphrases\n'
        'BASE = "Qwen/Qwen3-1.7B"      # the judge to fine-tune\n'
        "!nvidia-smi --query-gpu=name,memory.total --format=csv"
    ),
    new_code_cell(
        "!git clone --depth 1 $REPO_URL /kaggle/working/footnote\n"
        "%cd /kaggle/working/footnote/finetune\n"
        '!pip install -q -U "transformers>=4.51" "peft>=0.15" "trl>=0.19" "accelerate>=1.6" '
        '"datasets>=3.5" "bitsandbytes>=0.45" pyyaml\n'
        '!pip install -q "vllm>=0.9"'
    ),
    new_markdown_cell(
        "## 1. Build the dataset\n"
        "Rules plus model-written variants from an open-weight model; every label is checked against the "
        "passages, splits are by document. Runs as a subprocess so vLLM frees the GPU before training."
    ),
    new_code_cell(
        "!python build_dataset.py --chunks $CHUNKS --out data/out \\\n"
        "    --generator vllm --model $GENERATOR --vllm-dtype half --max-model-len 4096\n"
        "import json; print(json.dumps(json.load(open('data/out/build_report.json'))['splits'], indent=1)[:3000])"
    ),
    new_markdown_cell(
        "## 2. Train (QLoRA: 4-bit NF4 base + LoRA r=16 on all linear layers)\n"
        "Optional: label `data/out/test_gold.jsonl` by hand first (`python label_gold.py ...` on your "
        "machine) and upload it, so the evaluation below uses human labels."
    ),
    new_code_cell("!python train_qlora.py --config configs/qwen3-1.7b.yaml"),
    new_markdown_cell("## 3. Evaluate: base zero-shot vs fine-tuned, same prompt, same test set"),
    new_code_cell(
        "!python eval_judge.py --name base:qwen3-1.7b --model $BASE --load-in-4bit --batch-size 8\n"
        "!python eval_judge.py --name qlora:qwen3-1.7b --model $BASE --adapter outputs/qwen3-1.7b-judge/adapter "
        "--load-in-4bit --batch-size 8\n"
        "import pandas as pd\n"
        "pd.DataFrame(json.load(open('results/judge_rows.json')))"
    ),
    new_markdown_cell("## 4. Export: merge the adapter, convert to GGUF, quantize to Q4_K_M"),
    new_code_cell(
        "!python export/merge_adapter.py --base $BASE --adapter outputs/qwen3-1.7b-judge/adapter --out outputs/merged\n"
        "!bash export/to_gguf.sh outputs/merged outputs/gguf"
    ),
    new_markdown_cell(
        "## 5. (Optional) push the adapter to the Hugging Face Hub\n"
        "Fill in MODEL_CARD.md's results table from `results/` first."
    ),
    new_code_cell(
        "from kaggle_secrets import UserSecretsClient\n"
        "from huggingface_hub import login, upload_folder\n"
        'login(UserSecretsClient().get_secret("HF_TOKEN"))\n'
        'REPO_ID = "YOUR-USER/footnote-judge-qwen3-1.7b"   # edit\n'
        'upload_folder(repo_id=REPO_ID, folder_path="outputs/qwen3-1.7b-judge/adapter", repo_type="model")'
    ),
    new_markdown_cell(
        "Download `outputs/gguf/` (the Q4_K_M file and the Modelfile) and `results/`, then on your machine: "
        "`ollama create footnote-judge -f Modelfile` and point the backend's local judge at "
        "`http://localhost:11434/v1`."
    ),
]


def main() -> None:
    nb = new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["kaggle"] = {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}
    nbformat.validate(nb)
    target = Path(__file__).with_name("kaggle_train.ipynb")
    nbformat.write(nb, target)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
