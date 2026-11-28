# ML project plan (final-project scope)

Add a real ML angle to Audelle without breaking the working product. The ML
lives in a **separate folder** (`ml-project/`); the Audelle UI is a demo that
shows the outputs.

## Guiding rule

**Do not break the working product.** Everything below is additive:

- `ml-project/` is a new, separate folder.
- New backend endpoints sit behind a feature flag and are off by default.
- The local DSP editor stays the default; ML outputs come from the backend only.
- No existing `server/` or `website/` behaviour changes until a model passes eval.

## The three models (one per job)

| Job | Model | Dataset | What it learns | Metric |
|---|---|---|---|---|
| Separate stems | Demucs `htdemucs` | MUSDB18-HQ | unmix a song into parts | SDR / SI-SDR |
| Clean audio | DeepFilterNet3 | DNS Challenge | remove noise, keep signal | PESQ / STOI |
| Recognize instruments | PANNs CNN14 | AudioSet / NSynth | name what's playing | precision / recall / F1 |

## Datasets are huge — use subsets

You do **not** download the full datasets. For a final project the models are
**pretrained** (no big training, no GPU), so the dataset is only needed for
**evaluation and demo**, not training.

- MUSDB18-HQ: 150 songs (~30 GB) → use **10–20 tracks** for the demo.
- DNS: large → use the small dev subset.
- NSynth / AudioSet tags: use a few hundred labelled notes/clips.

Each model keeps its own dataset; there is no mixing. A `manifest.json` per
dataset records source URL, license, checksum and split.

## One "aggregated" model = one wrapper (code, not ML)

A single orchestrator runs all three models and returns one combined result:

```
audio in → wrapper →
  { stems: {vocals, drums, bass, other},
    clean: <clean audio>,
    instruments: [{label, confidence}] }
```

Precision/recall/F1 only make sense for **classification** (PANNs). Separation
uses SDR/SI-SDR, cleaning uses PESQ/STOI. The wrapper reports all of them in one
metrics table so the demo shows evidence, not vibes.

## Roadmap

1. **Phase 0 — scaffold** `ml-project/` (README, folders, manifest schema,
   small-subset download scripts). No training, no heavy downloads.
2. **Phase 1 — pretrained baseline** wire Demucs + DeepFilterNet3 + PANNs into
   the backend behind a flag; add a `/features/ml` demo page.
3. **Phase 2 — evaluate** on the small subsets; show SDR/PESQ/precision-recall.
4. **Phase 3 — fine-tune only if a GPU appears** (Colab). Optional.

## Models / subagents

Planning is done here; implementation tasks go to `general` subagents using
`opencode/deepseek-v4.1-flash-free`.
