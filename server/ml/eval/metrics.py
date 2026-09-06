"""Model-level metrics exposed via API and CI.

- Genre classifier: artefact existence, classes, feature_size, CV note from training.
- Prompt bench: delegated to prompt_bench.evaluate_prompt_bench()
- System: shape echoes for /understand so the frontend can show confidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "genre_classifier.joblib"


def genre_model_metrics() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {"available": False, "reason": "no artifact at server/ml/models/genre_classifier.joblib"}
    try:
        art = joblib.load(MODEL_PATH)
        return {
            "available": True,
            "classes": art.get("classes", []),
            "feature_size": int(art.get("feature_size", 0)),
            "model_type": type(art.get("model")).__name__ if art.get("model") is not None else "unknown",
            "cv_note": "train via scripts/train_genre_classifier.py — 5-fold CV logged to stdout; re-run to refresh",
            "artifact_bytes": MODEL_PATH.stat().st_size,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def system_metrics() -> dict[str, Any]:
    from server.ml.eval.prompt_bench import evaluate_prompt_bench

    return {
        "genre": genre_model_metrics(),
        "prompt_bench": evaluate_prompt_bench(),
    }
