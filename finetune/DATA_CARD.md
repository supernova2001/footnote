# Data card: Footnote grounding-judge training set

## Purpose

Supervised fine-tuning of a small model to judge whether source passages support a claim
(see `MODEL_CARD.md`). One row = passages + one claim + the target verdict.

## Source text

Passages are chunks of Footnote's corpus, exported by the backend. Every chunk carries its licence
(`license_id`) and `source_url`. Corpus sources: Wikipedia biographies (CC BY-SA 4.0), US
public-domain works, and CC BY / CC0 open-access papers. **Distributing this dataset carries those
terms**: attribute CC BY sources, and share anything containing CC BY-SA text under CC BY-SA. Check
`license_id` per row before publishing a split. The repository's test fixtures use an invented corpus
(`footnote_ft/demo_corpus.py`) that has no licence question.

## How claims are made (labels known by construction)

| label | method | what it does |
|---|---|---|
| supported | `verbatim` | the anchor sentence itself |
| supported | `trim_clause` | the anchor minus a non-restrictive clause or parenthetical |
| supported | `model_paraphrase` | open-weight model's paraphrase; rejected if it adds a number or name the passage lacks |
| partial | `certainty_upgrade` | one hedge made certain ("suggests" → "proves", "some of" → "all of") |
| partial | `added_year` | a year the passage never gives, on a sentence with no date |
| partial | `added_place` | a place, only when the passage locates nothing (else it could contradict) |
| partial | `added_qualifier` | "unanimously", "immediately", ... in a passive verb |
| partial | `model_partial` | model adds one named detail; rejected if the passage contains it |
| unsupported | `negation` | add or remove a negation |
| unsupported | `number_swap` | change a number to one the passages lack |
| unsupported | `entity_swap` | swap a name for one from another document of the same split |
| unsupported | `cross_document` | a sentence from another document of the same split, low word overlap |
| unsupported | `outside_fact` | a true sentence from elsewhere in the same document, not among the passages shown |
| unsupported | `model_contradiction` | model-written contradiction of the anchor |
| any | `recorded` | a judgment recorded by the backend, **only if a person labelled it** |

Every candidate is re-verified against the exact passages in its row: supported and partial evidence
must be a verbatim quote of a passage; partial and unsupported claims must not occur in the passages.
Failures are counted in `build_report.json` (`rejected`).

The `evidence` target is the whole anchor sentence (the decisive quote); `rationale` targets are
templated per method.

## Splits and balance

- Split **by document** (deterministic hash of the document id, 80/10/10). Borrowed text (cross-document
  negatives, swapped names) comes from the same split only.
- Labels are balanced per split by downsampling; within a label, methods are sampled round-robin.
- `test_gold.jsonl`: a label-stratified sample of the test split for hand labelling
  (`label_gold.py`). The agreement rate between human and constructed labels is the measured label
  noise; report it with any results.

## Versions

- `prompt_version` on every row: the backend's `judge_prompt.PROMPT_VERSION`. A prompt change means a
  rebuild; training and evaluation refuse stale rows.
- `generation_version` on model-written rows: `footnote_ft/generate.py` `GENERATION_VERSION`.
- `source`: `rules`, `model:<generator id>` or `recorded`.

## Known limitations

- Rules-only builds (`--rules-only`, for dry runs) make every supported claim verbatim or trimmed; use
  model-written paraphrases for real training data.
- Rule-based corruptions are surface edits; real hallucinations are subtler. The human-labelled gold
  set is the honest test.
- English only; sentence splitting is rule-based and can miss unusual abbreviations.
- Claims written by a closed model (recorded judgments of Claude's answers) enter only as inputs with
  human labels; whether even that is acceptable under the provider's terms is the project owner's call,
  and recorded rows are off unless `--judgments` is passed.
