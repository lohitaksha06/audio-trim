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
CONTENT_PATH = Path(__file__).resolve().parent.parent / "models" / "content_classifier.joblib"
GROOVE_PATH = Path(__file__).resolve().parent.parent / "models" / "groove_classifier.joblib"


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


def _artifact_card(path: Path, kind: str) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "reason": f"no artifact at {path.name}"}
    try:
        art = joblib.load(path)
        model = art.get("clf", art.get("model"))
        card: dict[str, Any] = {
            "available": True,
            "classes": art.get("classes", []),
            "feature_size": int(art.get("feature_size", 0)),
            "model_type": type(model).__name__ if model is not None else "unknown",
            "artifact_bytes": path.stat().st_size,
            "kind": kind,
        }
        for k in ("test_accuracy", "mae_db", "r2"):
            if k in art:
                card[k] = art[k]
        return card
    except Exception as e:
        return {"available": False, "reason": str(e)}


def content_model_metrics() -> dict[str, Any]:
    card = _artifact_card(CONTENT_PATH, "classification+regression")
    if card.get("available"):
        card["cv_note"] = "train via scripts/train_content_classifier.py — stratified 80/20 split logged to stdout; re-run to refresh"
    return card


def groove_model_metrics() -> dict[str, Any]:
    card = _artifact_card(GROOVE_PATH, "classification")
    if card.get("available"):
        card["cv_note"] = "train via scripts/train_groove_classifier.py — stratified 75/25 split logged to stdout; re-run to refresh"
    return card


def system_metrics() -> dict[str, Any]:
    from server.ml.eval.prompt_bench import evaluate_prompt_bench

    return {
        "genre": genre_model_metrics(),
        "content": content_model_metrics(),
        "groove": groove_model_metrics(),
        "prompt_bench": evaluate_prompt_bench(),
    }
