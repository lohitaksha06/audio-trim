# Audelle — ML

This folder holds the machine-learning part of Audelle: **"Fix my recording"**.

The product page `/features/clean` already measures and repairs a recording with
signal processing. The ML work here trains small models that do the same tasks
better, and measures them against that baseline.

## Models

| Task | Model | Purpose |
| --- | --- | --- |
| Diagnosis | Diagnoser | Predict noise level (SNR) and clipped fraction |
| Noise | Denoiser (conv-GRU mask) | Remove stationary noise |
| Clipping | Declipper (small U-Net) | Restore flat-topped peaks |
| Routing | Router (decision tree) | Choose the repair chain from the diagnosis |

Room echo is out of scope: there is no dereverberator, so it is not diagnosed.

## Data

- **VoiceBank-DEMAND** — clean speech, 16 kHz, CC BY 4.0
  (<https://datashare.ed.ac.uk/handle/10283/2791>)
- **DEMAND** noise recordings — mixed in at continuous SNRs
- Clipping applied in the pipeline, so the clipped fraction is exact

See `datasets.md` for sizes, licenses and subsets.

## Scope

Models are trained **from scratch**, small (about 1M parameters), and measured
against a hand-written baseline. The denoiser and declipper need a GPU (Colab or
borrowed); the diagnoser and router train on CPU.

See `../docs/ML_PROJECT_PLAN.md` for the full plan and experiments.
