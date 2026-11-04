# Plan

Derived from "Audelle ML Project: What I'm Actually Doing" (Nov 3, 2026).

## Scope

Train a model that reads an editing instruction and predicts:

- **operation** — one or more of 12 canonical operations
- **target** — the stem / speaker / mix it applies to
- **amount** — numeric value + unit when the instruction gives one
- **span** — start/end seconds when the instruction gives timestamps

Audelle's existing pretrained audio models remain the execution layer. This
project only trains the understanding layer. The product stays untouched.

## Operations (12)

| Operation | Meaning | Amount unit |
|-----------|---------|-------------|
| `remove_stem` | mute / delete a stem | — |
| `isolate_stem` | keep only a stem | — |
| `gain` | level change | dB |
| `fade` | fade in/out | seconds |
| `cut_segment` | delete a time region | seconds |
| `trim_segment` | keep only a time region | seconds |
| `remove_speaker` | drop a speaker (or keep only another) | — |
| `denoise` | remove noise / hiss / hum | — |
| `normalize` | even out loudness | — |
| `speed` | tempo / playback speed | factor or bpm |
| `add_effect` | reverb, delay, distortion, mood | — |
| `remove_silence` | cut pauses / silence | seconds (optional) |

Targets: `full_mix`, `vocals`, `drums`, `bass`, `guitar`, `keys`, `other`,
`speaker_1`, `speaker_2`.

## Milestones

| # | Milestone | Exit criterion | State |
|---|-----------|----------------|-------|
| 1 | Define operations + targets | schema validates, guide written | done |
| 2 | Hand-label 100+ prompts | ≥100 seed rows, all labels valid | done |
| 3 | Expand paraphrases | ~500+ rows, grouped splits | done |
| 4 | Rule-based baseline | runs on test split, metrics logged | done |
| 5 | Simple model | TF-IDF + logistic regression beats / loses to baseline honestly | done |
| 6 | Transformer | DistilBERT multi-task trained and evaluated | script ready; bert-tiny check only |
| 7 | Compare + error analysis | table + confusion matrix + failure examples | done for 1-5 |
| 8 | Demo | type a prompt, print structured action | done |
| 9 | Audelle integration | branch, after submission | not started |

## Evaluation script (for tomorrow)

1. Say what the project is, in one sentence.
2. Show the operation list (12) and 5 labeled rows.
3. Show the rule parser running as the baseline.
4. Show the dataset sizes + split sizes.
5. Show baseline vs simple model numbers on unseen prompts.
6. Show the plan: transformer → compare → error analysis → demo.

Do not claim transformer results that have not been produced yet.

## Open decisions

- [ ] Merge `add_effect` sub-types into separate labels? Keep one label until
      the dataset is large enough to support the split.
- [ ] Add `add_instrument` / `convert` / `classify` only after the core 12 are
      stable; more labels with thin data would hurt the F1.
- [ ] Decide whether multi-instruction prompts stay multi-label (current:
      yes; amount regression is restricted to single-operation prompts).
- [ ] Encode op→target pairing (per-clause extraction or predicate-argument
      structure); the current flat schema unions targets across operations.
- [ ] Add span/timestamp prediction to the learned models (rules only today).
- [ ] Human-verify every LLM-generated paraphrase before adding it to the test
      set; the test set must stay hand-labeled.
