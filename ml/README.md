# Instruction → Action (Audelle ML project)

A standalone ML project that turns a natural-language audio-editing instruction
into a structured action:

| Field | Meaning | Example |
|-------|---------|---------|
| `ops` | one or more operations | `["gain", "remove_stem"]` |
| `targets` | what the operation acts on | `["drums", "vocals"]` |
| `amount` / `unit` | how much | `6.0` / `"dB"` |
| `span` | start/end seconds when stated | `[5.0, 20.0]` |

> **This folder is intentionally standalone.** It does **not** import the
> Audelle app (`server/`, `website/`, `mobile/`) and changing it cannot break
> Audelle. Audelle integration is a later, separate step.

---

## Why this project exists

The product (Audelle) already has a large rule-based prompt parser. That parser
is the *baseline* here, not the product. The ML contribution is learning the
mapping from text to action:

1. **Classification** — which operation(s) and which target(s)?
2. **Regression** — how much (+6 dB, a 2-second fade, 1.5× speed)?

The pretrained audio models (Demucs, Whisper, pyannote) stay the "hands" that
execute an action. This project trains the "brain" that understands it.

---

## Current status

| Piece | Status |
|-------|--------|
| Operation + target schema (12 ops, 9 targets) | done |
| Hand-labeled seed prompts | done (`data/seed_prompts.jsonl`) |
| Paraphrase expansion + grouped splits | done (`scripts/build_dataset.py`) |
| Rule-based baseline | done (`src/instruction_ml/rules.py`) |
| TF-IDF + logistic-regression baseline | done (`scripts/train_baseline.py`) |
| Evaluation (accuracy, F1, confusion, amount MAE) | done (`scripts/evaluate.py`) |
| Error analysis | done (`reports/`) |
| Small transformer (DistilBERT multi-task) | scaffolded; bert-tiny pipeline check only (not usable yet) |
| LLM paraphrasing helper (optional) | scaffolded (`scripts/expand_with_llm.py`) |
| Inference demo | done (`scripts/predict.py`) |
| Audelle integration | **out of scope for now** |

Results from the last local run are in [`RESULTS.md`](RESULTS.md). Re-run the
pipeline to refresh them. The headline number is the **held-out human-written**
set, not the paraphrased template split.

---

## Quickstart (Windows / PowerShell)

```powershell
cd ml
python -m pip install -r requirements.txt

# 1. Build the dataset (seed + paraphrases, grouped train/val/test splits)
python scripts/build_dataset.py

# 2. Train the TF-IDF + logistic-regression baseline
python scripts/train_baseline.py

# 3. Evaluate baseline + rule-based parser on the unseen test split
python scripts/evaluate.py --split test

# 3b. The honest benchmark: independent, hand-written prompts
python scripts/evaluate.py --split heldout

# 4. Try a prompt
python scripts/predict.py "make the drums louder by 6 db and remove the vocals"
```

Everything is pure Python + scikit-learn. No API key, no GPU, no network.

### Transformer (optional, needs PyTorch + internet once)

```powershell
# small and fast; good for a laptop / Colab CPU
python scripts/train_transformer.py --model-name prajjwal1/bert-tiny --epochs 8

# the real run (Colab / Kaggle / friend's GPU)
python scripts/train_transformer.py --model-name distilbert-base-uncased --epochs 4
```

Then evaluate the checkpoint:

```powershell
python scripts/evaluate.py --split test `
  --transformer-preds models/transformer/test_predictions.json
python scripts/evaluate.py --split heldout `
  --transformer-preds models/transformer/heldout_predictions.json
```

---

## Dataset

`data/seed_prompts.jsonl` — written and labeled by hand. Each line:

```json
{"id": "seed-0001", "text": "make the drums louder by 6 db",
 "ops": ["gain"], "targets": ["drums"], "amount": 6.0, "unit": "dB",
 "span": null, "notes": ""}
```

Labeling rules are in [`data/labeling_guide.md`](data/labeling_guide.md).
`scripts/build_dataset.py` expands each seed row into paraphrases and writes:

```
data/dataset.jsonl          # seed + template variants
data/splits/train.jsonl     # 70% of groups
data/splits/val.jsonl       # 15% of groups
data/splits/test.jsonl      # 15% of groups
data/dataset_stats.json
```

Splits are **grouped by seed id**, so no paraphrase of a test prompt appears in
training.

`data/heldout_prompts.jsonl` is a separate, independently written set of 80
prompts. It is never used for training or threshold tuning and is the headline
evaluation. The template test split is a fast regression check; because its
wording overlaps the seeds, it is optimistic.

## Models

| Model | What it does |
|-------|--------------|
| `rules.py` | deterministic baseline, adapted from the product parser for comparison |
| `model.py` | TF-IDF (word + char n-grams) → one-vs-rest logistic regression for ops and targets; ridge regression for amounts |
| `train_transformer.py` | DistilBERT with four heads (ops BCE, targets BCE, unit CE, amount L1) |

## Metrics

- Operation: subset accuracy, micro/macro F1, per-operation F1, confusion matrix
  on single-operation prompts.
- Target: subset accuracy, micro/macro F1, per-target F1.
- Amount: MAE per operation/unit on single-operation prompts.
- Whole action: exact match of ops + targets.

Reports are written to `reports/` as JSON + Markdown + an error CSV.

## Known limitations

- **No op→target pairing.** "make the drums louder and remove the vocals" is
  stored as two operations and two targets, not as `gain(drums)` +
  `remove_stem(vocals)`. The flat schema is deliberate for v1 but a per-clause
  or predicate-argument version is the natural next step.
- **Amounts are single-operation only.** A number in a multi-operation prompt
  is ambiguous, so it is left null and excluded from regression.
- **Span is rule-only.** The learned models do not predict timestamps yet;
  `rules.py` extracts them, and `span` is reported but not scored.
- **The template test split is optimistic.** Use `--split heldout` for the
  honest comparison.

---

## Layout

```
ml/
  data/                 seed prompts, labeling guide, generated dataset + splits
  src/instruction_ml/   schema, data, augmentation, rules, model, metrics
  scripts/              build / train / evaluate / predict / transformer / LLM helper
  tests/                fast pytest checks for labels, splits, baseline
  reports/              generated metrics and error analysis
  models/               baseline joblib (committed); transformer checkpoints (ignored)
```

## What is deliberately *not* here

- No Audelle imports, no audio processing, no Demucs/Whisper calls.
- No integration with the product website or mobile app.
- No claims of transformer results before the transformer is trained.

The last point matters for the evaluation: show only what actually exists.
