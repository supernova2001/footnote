# Fine-tuning Footnote's grounding judge (QLoRA)

Footnote's agents never let a sentence through unchecked: every claim in an answer or a draft goes to
a **grounding judge** that reads the claim and the passages it cites and returns
`{"label": "supported" | "partial" | "unsupported", "evidence": "...", "rationale": "..."}`
(contract: [`docs/api-contract.md`](../docs/api-contract.md#the-grounding-judge-shared-by-the-agents-and-finetune)).
Claude does that job by default. This folder trains a small open model, **Qwen3-1.7B** (Apache-2.0),
to do the same job: it runs on a laptop CPU as a 1 GB GGUF and costs nothing per call.

```
corpus chunks ──► build_dataset.py ──► train / val / test (+ test_gold for hand labels)
                     │ rules + open-weight paraphrases, labels known by construction
                     ▼
               train_qlora.py  (4-bit NF4 base + LoRA r=16, loss on the verdict only)
                     ▼
               eval_judge.py   (base vs fine-tuned vs served, metrics shared with the backend)
                     ▼
               export/         (merge → GGUF → Q4_K_M → Ollama "footnote-judge" at :11434/v1)
```

## Design choices

**Why fine-tune the judge, not the writer.** The judge's task is narrow (three labels, a quote, a
sentence), has a single right answer per example, and can be measured exactly, so a 1.7B model can
plausibly match a frontier model on it and the claim is testable. It is also the call made most often:
one answer of eight sentences means eight judge calls. Writing well for a reader is open-ended,
is hard to score, and is where a large model earns its cost. Cheap, local, measurable checking of
expensive generation is the right split.

**Why labels by construction.** Each training claim is derived from a sentence of a real passage by a
transformation whose effect on the label is known: restate it (supported); add a date, place,
qualifier or more certainty (partial); negate it, change its number or name, or take a sentence from
somewhere else (unsupported). No model labels the data, so no closed model's output becomes a
training target, and every example is re-checked against the exact passages the judge will see
(`_verified` in `footnote_ft/dataset.py`): evidence must be a verbatim quote, and a partial or
unsupported claim must not itself occur in the passages. Paraphrases come from an open-weight model
(default Qwen3-4B, Apache-2.0), and a paraphrase is kept only if it adds no number or name the passage
lacks. Rules that cannot guarantee their label decline: `added_place`, for example, refuses any passage
that already locates something, because a second place there would contradict instead of go beyond.

**Why split by document.** Neighbouring passages of one source share names, dates and phrasing. A
random row split would put the same facts in train and test and measure memorisation. Whole documents
go to one split, and every negative that borrows text from another document borrows it from the same
split, so no test claim's text appears anywhere in training (`tests/test_dataset.py` checks this).

**One prompt, imported, not copied.** `footnote_ft/shared.py` loads the backend's
`judge_prompt.py` and `judge_metrics.py` by path. The fine-tuned model is trained, evaluated and served
on the byte-identical prompt the backend sends Claude, and both judges are scored by the same function.
Every row carries `prompt_version`; training and evaluation refuse rows built with an older prompt.

**Loss on the verdict only.** Rows become TRL prompt-completion pairs: the prompt is rendered with the
model's own chat template (`enable_thinking=False` for Qwen3, so the verdict follows an empty think
block), the completion is the JSON. The passages are context, not something to learn to generate.
Rows longer than `max_length` are dropped and counted, never truncated: truncation would cut off
the verdict.

## Run it

### Locally (no GPU): tests and a dry run

```bash
cd finetune
uv sync --group dev                         # light deps only
uv run pytest -q
uv run python -m footnote_ft.demo_corpus data/smoke/chunks.jsonl     # invented demo corpus
uv run python build_dataset.py --chunks data/smoke/chunks.jsonl --out data/smoke --rules-only
```

CPU smoke run of training and evaluation (downloads a 135M model; plain LoRA, 3 steps):

```bash
uv sync --extra train --group dev
uv run python train_qlora.py --config configs/smoke-cpu.yaml
uv run python eval_judge.py --name smoke --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --adapter outputs/smoke/adapter --data data/smoke/test.jsonl --gold data/smoke/none.jsonl \
    --results-dir outputs/smoke/results
```

### On a free Kaggle (or Colab) T4: the real run

Open [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) (cells are generated from
`notebooks/build_notebook.py`). It runs, in order:

```bash
python build_dataset.py --chunks chunks.jsonl --out data/out --generator vllm --model Qwen/Qwen3-4B --vllm-dtype half
python train_qlora.py --config configs/qwen3-1.7b.yaml
python eval_judge.py --name base:qwen3-1.7b  --model Qwen/Qwen3-1.7B --load-in-4bit --batch-size 8
python eval_judge.py --name qlora:qwen3-1.7b --model Qwen/Qwen3-1.7B --adapter outputs/qwen3-1.7b-judge/adapter --load-in-4bit --batch-size 8
python export/merge_adapter.py --base Qwen/Qwen3-1.7B --adapter outputs/qwen3-1.7b-judge/adapter --out outputs/merged
bash export/to_gguf.sh outputs/merged outputs/gguf
```

Any OpenAI-compatible server can replace vLLM's offline engine for generation
(`--generator openai --base-url http://host:port/v1 --model ...`). `configs/qwen3-4b.yaml` trains the
larger sibling.

### Hand-label the gold test set (recommended before reporting numbers)

```bash
python label_gold.py data/out/test_gold.jsonl
```

200 test claims, stratified by label, shown without their constructed label. Answers are saved after
each key press. At the end you get the agreement rate between your labels and the construction, which
is the measured noise of the dataset's labels. Once labels exist, `eval_judge.py` scores against them
(`"label_source": "human"` in the results), and `build_dataset.py` never overwrites them.

### Serve it

```bash
cd outputs/gguf && ollama create footnote-judge -f Modelfile
python export/check_served.py --base-url http://localhost:11434/v1 --model footnote-judge
python eval_judge.py --name local:footnote-judge --openai-base-url http://localhost:11434/v1 --openai-model footnote-judge
```

The backend's local judge calls that same OpenAI-compatible endpoint.

## Time and cost (estimates, not measurements)

| Step | Where | Rough time | Cost |
|---|---|---|---|
| Tests + rules-only build | laptop CPU | seconds | 0 |
| Model-written claims, ~2k anchors, Qwen3-4B on vLLM | Kaggle T4 | 15-30 min | 0 (Kaggle gives ~30 GPU h/week) |
| QLoRA, Qwen3-1.7B, ~6k rows × 3 epochs | Kaggle T4 | 1.5-3 h | 0 |
| Evaluation, 200-600 rows, per judge | Kaggle T4 | 5-15 min | 0 |
| GGUF export + quantize | Kaggle CPU | ~10 min | 0 |
| Inference, Q4_K_M on a laptop CPU | Ollama | roughly 1-4 s per verdict | 0 |

The scripts print and record the real numbers (`train_metrics.json`, `results/*.json`); replace this
table with them.

## Results

Not run yet. `eval_judge.py` writes `results/judge_rows.json` (the `judge` rows of the API's
`EvalReport`, served at `GET /api/evals`) and one detailed file per judge.

| judge | labels | n | macro-F1 | unsupported recall | JSON valid | p50 latency | cost / 1k |
|---|---|---|---|---|---|---|---|
| base:qwen3-1.7b (zero-shot) | | | | | | | |
| qlora:qwen3-1.7b | | | | | | | |
| local:footnote-judge (Q4_K_M, Ollama) | | | | | | | |
| claude (backend eval) | | | | | | | |

Metric definitions are in `backend/src/footnote/judge_metrics.py`: an unparsable reply counts as
wrong; "deployed" scores also treat a supported/partial verdict whose evidence is not a verbatim
quote as unsupported, as the agents do.

## Row format (what `footnote eval judge` reads)

- `train/val/test.jsonl`: `id`, `prompt_version`, `split`, `label` (the constructed gold label),
  `method` (how the claim was made: `verbatim`, `negation`, `model_paraphrase`, ... never null),
  `source` (`rules` | `model:<id>` | `recorded`), `document_id`, `passage_ids`, `claim`, `evidence`,
  `passages` (`[{id, title, text}]`), `messages` (system / user / assistant).
- `test_gold.jsonl`: `id`, `prompt_version`, `document_id`, `claim`, `passages`, `method`,
  `constructed_label`, `gold_label` (the human label, `null` until labelled), `note`. Score against
  `gold_label` where it is set; `eval_judge.py` uses only labelled rows.

## What the backend provides

- **`data/chunks.jsonl`**: one backend `Chunk` per line (`backend/src/footnote/domain.py`): `id`
  (`<document_id>:<ordinal:04d>`), `document_id`, `ordinal`, `title`, `section`, `text`, `char_start`,
  `char_end`, `scholar_ids`, `license_id`, `source_url`. Required here: `id`, `document_id`,
  `ordinal`, `title`, `text`.
- **Optional `data/judgments.jsonl`**: recorded judgments, one per line:
  `{"claim", "passages": [{"id", "title", "text"}], "label", "evidence", "rationale", "judge",
  "prompt_version", "human_label"}`. Only rows with a `human_label` are used, and only that label and
  the verbatim evidence; the recording model's own label and rationale are never training targets.

## Files

| Path | What |
|---|---|
| `build_dataset.py` | chunks → balanced, verified, by-document train/val/test + gold sample |
| `label_gold.py` | terminal labeller for the gold sample |
| `train_qlora.py`, `configs/*.yaml` | QLoRA training (plain LoRA on CPU for smoke runs) |
| `eval_judge.py` | score base / adapter / served judges with the shared metrics |
| `export/` | merge adapter, GGUF + Q4_K_M, Ollama Modelfile, served-model check |
| `notebooks/` | the Kaggle notebook and its generator |
| `footnote_ft/` | rules (`corruptions.py`), model-written claims (`generate.py`), assembly (`dataset.py`), prompt formatting, demo corpus |
| `MODEL_CARD.md`, `DATA_CARD.md` | cards for the Hub and the dataset |
