---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: peft
pipeline_tag: text-generation
language: [en]
tags: [qlora, lora, fact-checking, grounding, rag, claim-verification]
---

# footnote-judge-qwen3-1.7b

A LoRA adapter that turns **Qwen3-1.7B** into a grounding judge: given source passages and one claim,
it says whether the passages support the claim and quotes the evidence. It is the local judge of
Footnote, a RAG and agent project where every sentence an agent writes is checked before a reader
sees it.

> Template. Fill in every `TODO` from `train_metrics.json` and `results/` before publishing. No number
> on this card may be estimated.

## Output contract

```json
{"label": "supported" | "partial" | "unsupported", "evidence": "<shortest verbatim quote, or \"\">", "rationale": "<one sentence>"}
```

- **supported**: everything the claim says is stated in, or follows directly from, the passages.
- **partial**: the core is supported, but the claim adds a detail, number, scope, date or certainty
  the passages do not give.
- **unsupported**: the passages do not say it, say the opposite, or are about something else.

## Prompt

The system and user text come from Footnote's `judge_prompt.py` (version `TODO: PROMPT_VERSION`),
rendered with Qwen3's chat template and **thinking disabled** (`enable_thinking=False`). Use exactly
that prompt; the model was trained on nothing else.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "TODO/footnote-judge-qwen3-1.7b")
tok = AutoTokenizer.from_pretrained("TODO/footnote-judge-qwen3-1.7b")
system, user = render_judge_prompt(claim, passages)  # from Footnote's judge_prompt.py
text = tok.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
out = model.generate(**tok(text, return_tensors="pt").to(model.device), max_new_tokens=256, do_sample=False)
```

A merged Q4_K_M GGUF with an Ollama `Modelfile` is produced by `finetune/export/`.

## Training

| | |
|---|---|
| Method | QLoRA: 4-bit NF4 base with double quantization, LoRA on all linear projections |
| LoRA | r=16, alpha=32, dropout=0.05 |
| Optimiser | paged AdamW 8-bit, lr 2e-4, cosine, 5% warmup, effective batch 16 |
| Epochs | TODO |
| Loss | on the JSON verdict only (prompt-completion masking) |
| Hardware | TODO (e.g. 1× Tesla T4, fp16), TODO hours |
| Data | TODO rows train / TODO val; see the data card |

## Evaluation

Test set: TODO rows from documents never seen in training; labels: TODO (human-labelled gold sample
or constructed). Metrics from Footnote's `judge_metrics.py`.

| judge | macro-F1 | unsupported recall | JSON valid | evidence verbatim | p50 latency |
|---|---|---|---|---|---|
| Qwen3-1.7B zero-shot | TODO | TODO | TODO | TODO | TODO |
| this adapter | TODO | TODO | TODO | TODO | TODO |
| Claude judge (reference) | TODO | TODO | TODO | TODO | TODO |

## Limitations

- English only; trained on short passages (1-3 chunks) and single-sentence claims.
- Judges against the given passages only; it has no world knowledge by design, and a claim that is
  true but absent is "unsupported".
- Rationales in training are templated, so they are terse and formulaic; rely on the label and the
  evidence quote.
- Constructed negatives are easier than real hallucinations; check the human-labelled results above,
  not only the constructed test set.
- A supported/partial verdict whose evidence is not a verbatim quote should be treated as unsupported
  (Footnote's agents do).
