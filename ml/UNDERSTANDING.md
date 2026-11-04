# ML project - complete understanding

This file explains the whole project in plain words. Read it top to bottom before the evaluation.

## 1. What the project does

A user types an instruction like "make the drums louder by 6 db".

The project turns it into an action:
- ops: ["gain"]
- targets: ["drums"]
- amount: 6.0
- unit: "dB"
- span: null

Ops means what to do. Targets means what to do it to. Amount means how much. Span means start and end time in seconds.

Audelle audio code can do the edit. This project only understands the text.

## 2. Operations and targets

There are 12 operations:
remove_stem, isolate_stem, gain, fade, cut_segment, trim_segment, remove_speaker, denoise, normalize, speed, add_effect, remove_silence.

There are 9 targets:
full_mix, vocals, drums, bass, guitar, keys, other, speaker_1, speaker_2.

Rules for hard cases:
- cut deletes a part. Trim keeps a part.
- "remove the vocals" removes the stem. "remove the guitar solo" removes a time part of the guitar.
- "keep only speaker 1" is stored as remove_speaker with target speaker_2. The target is always the speaker who goes away.
- normalize makes loudness even over time. Gain applies a fixed dB change.
- Amount is only stored for single-operation rows. If a prompt has two operations, amount is null because it is not clear which operation the number belongs to.

File: `ml/src/instruction_ml/schema.py`

## 3. Dataset - what each row looks like

File: `ml/data/seed_prompts.jsonl`

One line example:
`make the drums louder by 6 db` with ops gain, targets drums, amount 6.0, unit dB.

Each row has: id, text, ops, targets, amount, unit, span, notes, source, group.

Source is seed, template, or human. Group links copies of one seed together.

Second file: `ml/data/heldout_prompts.jsonl` with 80 rows. These rows are never used for training. They use new wording like nuke, chuck, chop, ditch.

## 4. How 297 becomes 1482

Script: `ml/scripts/build_dataset.py`
Code: `ml/src/instruction_ml/augment.py`

Steps for each seed:
1. Start with seed text.
2. Try to swap one phrase. List is fixed. Examples: remove the to delete the, keep only to keep just, clean up to denoise, speed it up to make it faster, 6 db to 6db, 2 seconds to 2s.
3. Add polite words if the sentence starts with a verb. Front: please, can you, could you. Back: please, for me, on this track. No front part if the sentence starts with "just the piano" because that would make bad English.
4. Keep labels exactly the same. Same ops, same targets, same amount, same span.
5. Give the copy the same group as the seed, for example seed-0001-v1 has group seed-0001.
6. Drop the copy if the text already exists.
7. Make up to 4 copies per seed.

Result now: 297 seeds -> 1482 rows.

Then split by group:
- train 1032
- val 225
- test 225
- held-out 80 separate

Group split means no copy of a test prompt is in training. This is checked by code.

## 5. Rule baseline - what it does

File: `ml/src/instruction_ml/rules.py`

It is a set of word patterns. No learning.

For "remove the vocals": finds verb remove plus stem vocals, returns remove_stem.
For "set tempo to 128 bpm": finds bpm number, returns speed with 128 bpm.
For time: finds "from 0:05 to 0:20" and returns span [5.0, 20.0].

Good at numbers and speaker words. Bad at new verbs like nuke, chuck, ditch because those words are not in its lists.

## 6. Learned model - what it does

File: `ml/src/instruction_ml/model.py`

Name: TF-IDF plus logistic regression plus ridge regression.

### 6a. Text to numbers

Two TF-IDF vectorizers joined together:
- word 1-2 grams: boost, bass, boost bass
- char 3-5 grams: boo, oost, bas

TF-IDF gives high weight to rare useful words like hiss, bpm, speaker, and low weight to common words like the, please.

### 6b. Operation prediction

12 separate logistic regressions. Each asks one yes or no question. Is this gain. Is this remove_stem.

Settings: C 4.0, balanced class weight, solver liblinear, 2000 steps.

Same for targets: 9 separate logistic regressions.

Thresholds are picked on validation data. For each class try 0.20 to 0.70 and pick the threshold with best F1. Now remove_speaker needs 0.7, speed needs 0.2.

### 6c. Extra fixes at predict time

- Block remove_speaker unless the text has speaker, host, guest, interviewer, voice, or person. This stops "first 5 seconds" becoming a speaker.
- If two operations fire but the text has no and, then, also, plus, comma, keep only the top score.
- Concrete stem beats full_mix. Speaker targets are removed unless the operation is remove_speaker.
- If no target passes threshold and the operation usually applies to the whole mix, use full_mix.

### 6d. Amount prediction

Two steps:
1. Presence model: one logistic regression that asks does this prompt state a number.
2. If yes, ridge regression per op and unit: gain dB, fade seconds, speed bpm, speed factor, remove_silence seconds. Ridge alpha 1.0. Needs at least 6 examples or it is skipped.

Values are clamped: dB -24 to +24, factor 0.25 to 4.0, bpm 30 to 300.

Span is always null for this model. Only rules predict span.

## 7. How training runs

```
python scripts/build_dataset.py
python scripts/train_baseline.py
```

build_dataset validates seeds, expands to 1482, checks held-out has no overlap, writes splits and stats.

train_baseline fits TF-IDF on train only, fits 12 plus 9 plus presence plus ridge models, tunes thresholds on val only, saves to `ml/models/baseline.joblib`.

No test data is used in training. No held-out data is used in training or threshold tuning.

## 8. How scores are calculated

Script: `ml/scripts/evaluate.py`
Code: `ml/src/instruction_ml/metrics.py`

For operations, each prompt can have more than one label, so we use multi-label scores.

- Subset accuracy: share of rows where predicted ops exactly match gold ops. Strict.
- Micro F1: count all true positives, false positives, false negatives across all 12 ops, then compute F1. Good ops with many examples count more.
- Macro F1: compute F1 per operation, then average. Rare and common ops count the same.
- Per-class F1: F1 for one operation. Precision means of all times we predicted it, how many were right. Recall means of all times it was true, how many we found.
- Confusion matrix: for single-operation rows, rows are true op, columns are predicted op.
- Target scores: same as above but for 9 targets.
- Action exact match: share where both ops and targets match exactly.
- Amount MAE: mean absolute error. Example: true 6.0, predicted 3.69, error 2.31. Only counted if operation matches, amount exists, unit matches, and row has one operation. Median AE is also shown because one bad bpm error can shift the mean a lot.

Current held-out 80 rows:
rules op micro F1 0.709, TF-IDF 0.894
rules target micro F1 0.951, TF-IDF 0.927
rules exact 0.512, TF-IDF 0.787
rules amount MAE 0.533, TF-IDF 4.387

TF-IDF wins on operations. Rules win on exact numbers. Amount mean is high because of two bpm misses near 25 error each, median is near 1.0.

Template test 225 rows is less strict because wording overlaps training. Held-out is the number to quote.

## 9. Error types

From `ml/reports/errors_tfidf-logreg_heldout.csv`:

1. New slang verbs confuse rules. Nuke, chuck, ditch, lose have no rule. TF-IDF gets most from context words like vocals and drums.
2. Cut versus trim is hard. Keep 0:10 through 0:50 and leave only 30 seconds starting at 1:00 confuse both models.
3. Speaker polarity. Keep the host means remove speaker 2. The model learned this after more seeds, now 1.0 F1.
4. Numbers do not transfer. TF-IDF predicts near the mean for unseen bpm. This needs number features or a transformer.
5. Flat labels lose pairing. Make drums louder and remove vocals stores ops gain plus remove_stem and targets drums plus vocals, but does not store which target goes with which op. Next version should store pairs.

## 10. Transformer status

Script: `ml/scripts/train_transformer.py`

Plan: DistilBERT with four heads. Ops BCE, targets BCE, unit CE, amount L1.

Test with bert-tiny on CPU works end to end but scores are low. About 0.39 op micro F1 on template test. Too small, too little data, thresholds over-predict. Do not present as a result.

Real run needs DistilBERT base, 5 epochs, lr 3e-5, batch 16, on GPU, with 1000 plus checked prompts.

## 11. How to demo

```
python scripts/predict.py "boost the bass by 4 db"
python scripts/predict.py "nuke the vocals"
python scripts/predict.py --rules "keep only the vocals and add reverb"
python scripts/evaluate.py --split heldout
```

Reports: `ml/reports/report_heldout.md`, `ml/reports/errors_*_heldout.csv`, `ml/RESULTS.md`.

## 12. Limits to state openly

- No op to target pairing.
- Amount only for single-operation rows.
- Span only from rules.
- Template test shares vocabulary with training. Held-out is the honest number.
- No LLM generated the data. `expand_with_llm.py` was never run. Expansion is fixed templates in `augment.py`.
