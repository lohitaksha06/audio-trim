# Genre Classifier — Model Card (Audelle)

## Summary
Lightweight RandomForest over librosa features. Powers genre-aware suggestions in `POST /api/ml/understand`.

- **Artifact:** `server/ml/models/genre_classifier.joblib` (~ few KB)
- **Training:** `scripts/train_genre_classifier.py --clips-per-genre 10` (GTZAN subset, 100 clips, 10 genres)
- **Features (88):** 20 MFCC mean+std (40), 12 chroma mean+std (24), spectral centroid/bandwidth/rolloff mean+std (6), ZCR mean+std (2), RMS mean+std (2), tempo (1) — see `genre_classifier.py:_extract_features`
- **Model:** `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)` + `LabelEncoder`
- **Split:** 80/20 stratified, 5-fold CV

## Training
```bash
python scripts/train_genre_classifier.py --clips-per-genre 10
# optional: --clips-per-genre 30 for stronger baseline
python scripts/train_genre_classifier.py --keep-data
```

Reproducibility: `random_state=42` everywhere, pinned `librosa==0.11.0`, `numpy==2.0.0`, `joblib` artifact includes `feature_size` + `classes` so load fails loudly on mismatch.

## Evaluation
Run:
```bash
python -m pytest server/tests/test_eval.py -v
curl http://localhost:8000/api/eval/metrics
curl http://localhost:8000/api/eval/prompt-bench
```
- **Prompt bench:** 36 paraphrases across 15 intents; current regex engine: **accuracy 1.0** (see `server/ml/eval/prompt_bench.py`).
- **Genre:** test accuracy + CV printed by training script; production guard `confidence < 0.3 → None` so UI hides weak predictions.

## Limitations
- GTZAN subset is small; genre is inherently fuzzy — treat as suggestion, not ground truth.
- Heuristic fallback for instruments/mood is intentionally light; set `USE_PRETRAINED=1` + `INSTRUMENT_MODEL=laion/clap-htsat-unfused` to enable CLAP tagger (requires `transformers` + `torch`).

## Intended Use
UI suggestions (`genre_tuning_actions`) and paper baseline. Not for standalone genre research without retraining on full GTZAN/FMA.

## Versioning
Bump artifact by re-running training; commit new `.joblib` with git LFS if >50 MB (current is tiny). Metrics snapshot in `scripts/report_figs/results.json` for IEEE report.
