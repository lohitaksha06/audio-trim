# Audelle — Machine Learning Project Brief (for evaluation)

> One-line thesis: **Audelle is a supervised-ML audio-understanding system
> (classification + regression + clustering) with a DSP editing engine.**
> Generative synthesis exists only at the output stage and is procedural
> (NumPy oscillators), not a trained generative-AI model. Every product
> decision — what the audio *is*, what noise is on it, what to do next —
> comes from a trained discriminative model.

## 1. What part is majorly ML? (direct answer)

The **understanding pipeline** that runs on every upload
(`POST /api/ml/understand`, `server/routes/ml.py:90`). Before any edit is
offered, three trained classifiers + one regressor + one clustering stage
read the file:

| # | Head | Type | Classes / target | File |
|---|------|------|------------------|------|
| 1 | Noise-condition | **RandomForestClassifier** (200 trees) | clean/hiss/hum/crowd/clipped/muffled | `server/ml/audio_understanding/content_classifier.py` |
| 2 | Quality | **RandomForestRegressor** (150 trees) | effective SNR_dB → 0–100 score | same file |
| 3 | Groove/subgenre | **RandomForestClassifier** (150 trees) | 15 EDM feels | `server/ml/rhythm/analyzer.py:140` |
| 4 | Genre | **RandomForestClassifier** (200 trees) | 10 GTZAN genres | `server/ml/audio_understanding/genre_classifier.py:196` |
| 5 | Speakers | **KMeans + silhouette** (unsupervised) | SPK_0..N | `server/ml/diarization/diarization.py:60` |
| 6 | Structure | **cosine-similarity recurrence** | intro/verse/chorus/bridge/outro | `server/ml/audio_understanding/song_structure.py:94` |

Pretrained deep nets used as-is (not trained by us, CPU inference):
Demucs `htdemucs` (separation), Whisper `openai/whisper-tiny` (transcription),
optional CLAP `laion/clap-htsat-unfused` (instrument tagging).

**Not ML (say this explicitly):** prompt parsing
(`server/ml/prompt_engine.py:581`, regex), drum/synth rendering (NumPy DSP),
mix tips (thresholds). The prompt bench (`server/ml/eval/prompt_bench.py`)
*evaluates* the regex — it does not train it.

## 2. Which classifier, exactly? (so there is no ambiguity)

* **Noise/condition: `sklearn.ensemble.RandomForestClassifier`**,
  `n_estimators=200, random_state=42, n_jobs=-1`, 84-dim features,
  stratified 80/20 split, `classification_report` + confusion matrix.
* **Quality: `sklearn.ensemble.RandomForestRegressor`**,
  `n_estimators=150`, same split, MAE (dB) + R².
* **Groove: `RandomForestClassifier(150, max_depth=12, min_samples_leaf=2)`**,
  8-dim rhythm features, stratified 75/25 split.
* **Genre: `RandomForestClassifier(200)`**, 75-dim features, 80/20 split +
  5-fold CV.
* **Speakers: `sklearn.cluster.KMeans`**, k picked by
  `sklearn.metrics.silhouette_score`.

Why RandomForest everywhere: small tabular data, CPU-only, robust to
correlated spectral features, interpretable (`feature_importances_`),
serializes to ~1 MB joblibs, reproducible with fixed seeds.

## 3. Datasets — where from, how big, where stored

| Model | Source | Size | Storage |
|-------|--------|------|---------|
| Content/noise | **Real:** GTZAN subset, HuggingFace `storylinez/gtzan-music-genre-dataset`, `--clips-per-genre` (default 6) × 10 genres, first 10 s each. **Synthetic:** 6 labeled degradations per clip in RAM with per-clip sampled SNRs (hiss 8–18 dB, hum 50+100+150 Hz 5–15 dB, crowd/babble 3–12 dB, hard-clip, 1 kHz lowpass). Measured 2026-09-23, clip-grouped split (no leakage): classifier **acc 0.85** (12 held-out clips → 72 samples; muffled 1.00 / hiss 0.92 / clipped 0.96 / hum 0.86 / clean 0.67 / crowd 0.67 F1); regressor (clean/hiss/hum/crowd) **MAE 9.4 dB, R² 0.22** — modest baseline, see §7. | ~360 samples × 84 feats | Temp dir only, deleted after training |
| Groove | **Synthetic in RAM:** `render_loops()` 8 s loops, 15 feels × BPM/key/swing/humanize × polished/bare, ~164 clips. Optional real songs `--songs-dir data/songs/<label>/`. | ~164 × 8 feats | No dataset folder at all |
| Genre | **GTZAN subset** as above, 10 clips × 10 genres. | 100 × 75 feats | Temp dir only, deleted after |

Retrain commands (each prints accuracy / MAE to stdout):

```
python scripts/train_content_classifier.py [--clips-per-genre 6]
python scripts/train_groove_classifier.py [--songs-dir data/songs]
python scripts/train_genre_classifier.py [--clips-per-genre 10]
```

Artifacts: `server/ml/models/{content,groove,genre}_classifier.joblib`
(~1 MB each). All three load **lazily** and every caller has a
missing-artifact fallback, so the product never breaks.

## 4. Features (what the models actually see)

* Content/noise (84): MFCC20 mean/std (40) + chroma mean/std (24) +
  centroid/bandwidth/rolloff/ZCR/RMS mean/std (10) + tempo (1) + flatness
  mean/std (2) + hum-band 40–130 Hz ratio + hiss-band >6 kHz ratio +
  clip ratio + crest_dB (4) + hum-peak prominence + noise-floor ratio +
  spectral-contrast mean (3). The last three are the targeted noise
  discriminators (mains lines, pause-filling babble, peak/valley structure).
* Groove (8): tempo, fourfloor, halftime, swing, hats_density, sub_ratio,
  kick_rate, snare_rate — deliberately **no MFCC**, so synth training
  transfers to real songs.
* Genre (75): same base minus the 6 noise features.

## 5. How ML drives the product (ML is the hero, not decoration)

1. Upload → `/api/ml/understand` returns `content {condition, confidence,
   snr_db, quality_score, suggested_action}` + `genre` + `rhythm {groove}`.
2. UI shows "what kind of audio is this" (e.g. *crowd-noisy, SNR 11 dB,
   quality 28/100*) and a one-click fix prompt per condition
   (`CONDITION_ACTIONS`: hiss→denoise, clipped→normalize,
   muffled→brighten, crowd→isolate vocals…).
3. `add drums` with no style named uses the groove classifier to match the
   song's feel; named styles always override it.
4. Eval: `GET /api/eval/metrics` exposes all three model cards
   (classes, feature_size, test_accuracy, MAE/R², artifact bytes);
   `GET /api/eval/content|groove|genre` per model.

## 6. Generative vs discriminative (for the "is this just GenAI?" question)

* **Discriminative / learned:** the 4 classifiers + 1 regressor + KMeans
  above. They *decide* what the audio is. This is the graded ML.
* **Procedural synthesis:** drums/synths rendered from NumPy oscillators on
  the detected beat grid. No diffusion/transformer weights are trained for
  this; swapping timbres needs no retraining.
* **Optional LLM:** `USE_LLM=1` routes prompt parsing through an
  OpenAI-compatible endpoint; default is the offline regex engine. The LLM
  never touches audio.

## 7. Viva Q&A (30 seconds each)

* *Overfitting / leakage?* Fixed seeds; **clip-grouped splits**
  (`GroupShuffleSplit` — all 6 degradations of one clip stay on one side,
  so musical content can't leak); confusion matrices in training logs;
  forests capped on groove (`max_depth=12`); noise features are band-ratios,
  not memorized spectra.
* *Weak spots?* Clean↔crowd (5 clean→crowd) is the residual confusion —
  babble *is* music mixed with music; documented in the report with the
  confusion matrix. The SNR regressor is a modest baseline (MAE 9.4 dB,
  R² 0.22): clip-level spectral means carry condition strongly but level
  weakly — improvement path is frame-dynamics/level-aware features, stated
  as future work rather than hidden.
* *Why not deep learning?* <500 labeled clips per head; forests beat CNNs
  there, run on CPU in CI, and stay explainable.
* *Data leakage?* Groove synth shares **no code** with playback patterns;
  content uses clip-grouped splits (above), so no base clip appears on both
  sides.
* *What would you add with more time?* Calibrated probabilities
  (`CalibratedClassifierCV`), per-class precision/recall gates in the UI,
  and a learned section classifier to replace the cosine heuristic.

## 8. Safe ML extensions (additive, product never breaks)

1. Intent classifier (TF-IDF + LogReg on prompt bench) shadowing the regex.
2. Mood regressor (valence/arousal) replacing mood thresholds.
3. Calibrated confidence gates per head.
All three follow the established pattern: script in `scripts/train_*`,
joblib in `server/ml/models/`, lazy loader + `None` fallback, card in
`server/ml/eval/metrics.py`, endpoint in `server/routes/eval.py`, tests in
`server/tests/test_*.py`.
