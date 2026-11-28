# ML project plan — "Fix my recording"

Final scope. Adds a real ML project to Audelle **without changing the working
product**. Everything lives in `ml-project/`; the only product surface is one
audit page under `/features/clean`, which currently runs the signal-processing
baseline and says so.

## One line

Train small models from scratch that diagnose and repair a bad voice recording,
measure them against a hand-written baseline and the literature, and show the
repairs in Audelle.

## Guiding rules

- All ML work lives in `ml-project/`. No changes to `server/` or `website/`
  behaviour until a model passes its evaluation.
- The shipped product keeps its local DSP path. ML results are shown as a
  separate, honest comparison.
- CPU only. Torch threads capped at 4. One heavy job at a time. Training that
  needs a GPU runs on Colab or a borrowed GPU, never on the user's laptop in the
  background.
- Ask before downloading large data.

## The four models

| Model | Job | Trained on | Metric | Where |
|---|---|---|---|---|
| **Diagnoser** | Predict SNR (dB) and clipped fraction | Mixtures with exact labels | SNR error (dB), clipped-% error | CPU |
| **Denoiser** | Conv-GRU mask, noisy → clean | Mixtures built from clean speech | SI-SDR, STOI, PESQ | GPU |
| **Declipper** | Small U-Net that restores flat peaks | Clean speech, hard-clipped at known levels | SI-SDR vs cubic-spline baseline | GPU |
| **Router** | Decision tree over diagnoser outputs → repair chain | Labels from oracle search | Agreement with oracle | CPU, seconds |

Room echo is deliberately **out of scope**: there is no dereverberator, so the
diagnoser does not report echo. Diagnosing a fault you cannot repair is a demo
trick, not engineering.

## Why a learned router and not hand thresholds

A hand-written rule ("denoise below 8 dB") invites "why 8 dB?" and there is no
defensible answer. Instead:

1. Define a **finite** candidate set of pipelines, e.g.
   `denoise only`, `declip only`, `declip → denoise`, `denoise → declip`.
2. Run every candidate on every training clip and keep the best by SI-SDR
   (the oracle label).
3. Fit a decision tree on the diagnoser's outputs to predict that label.
4. Print the tree. Its splits are readable and reportable.

Ties and class imbalance must be reported: if one pipeline wins 90% of clips,
the tree is not learning routing, it is predicting the majority class.

## Data

- **Speech source:** VoiceBank-DEMAND clean utterances (CC BY 4.0, University of
  Edinburgh DataShare). 16 kHz.
- **Noise:** DEMAND recordings, mixed at **continuous** SNRs drawn from a
  distribution rather than the benchmark's four fixed levels, so the diagnoser
  is not trained on a handful of values.
- **Clipping:** applied in the pipeline at random levels; the clipped fraction
  is therefore known exactly.
- **Splits:** speaker-disjoint and noise-type-disjoint between train and test,
  matching the benchmark protocol. Never split by utterance.
- `manifest.json` per dataset: source URL, license, checksum, counts, split.

Total download target: small. The clean training set only, not the whole
48 kHz corpus.

## Experiments

1. **Baseline vs model** — no repair, spectral gating, declip by cubic
   interpolation, our models.
2. **Fixed pipeline vs learned router vs oracle** on held-out clips damaged with
   both noise and clipping.
3. **Order** — denoise-then-declip vs declip-then-denoise, and whether the
   router learns the better order.
4. **Diagnoser accuracy** — SNR error in dB and clipped-% error on unseen
   speakers and unseen noise types.
5. **Ablations** — model size, training data (25/50/100%), loss (L1 vs SI-SDR).

## Honest risks

- Synthetic damage is why the labels are exact, and also why the benchmark is
  not the real world. Every result is paired with real recordings of our own
  (before/after audio, no scores).
- The denoiser and declipper need a GPU. The diagnoser and router do not.
- Voice only. This is not a music restoration system.
- A 1M-parameter model trained from scratch will not beat pretrained
  state-of-the-art. The contribution is the training, the benchmark and the
  analysis, not a record score.
- More models is not more impressive. If time runs short, the diagnoser alone is
  a complete, defensible project.

## Build order

1. **Baseline in the product** — done: `/features/clean` measures noise,
   clipping and level and repairs with DSP. No ML claims.
2. **Diagnoser** — cheapest, exact labels, and it must beat the DSP estimate
   measured in step 1.
3. **Denoiser** — GPU, replaces spectral gating for the denoise step behind a
   flag.
4. **Declipper + router** — the comparison experiments.

## Demo in Audelle

`/features/clean` (shipped, local DSP) plus a research table stating what is
trained and what is not. Scores are added to that table only when a run
finishes, next to the baseline the model has to beat.

## Test gate before any merge

```
cd website && bun test src
python -m pytest server/tests -q
cd website && npx next build
cd website && npx eslint src
```
