"""High-level rhythm analysis: tempo, beats, feel, and auto-groove.

``analyze_rhythm`` is the single entry point. It runs the pulse-geometry grid
estimator, adds swing/hats/spectral descriptors, and — when the groove
classifier artifact (``scripts/train_groove_classifier.py``) is available —
guesses the electronic subgenre so "add drums" can match the song with no
prompt hints. Prompt-named grooves always win; this only fills the default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np

from server.ml.rhythm.features import (
    _ANALYSIS_SR,
    _band_env,
    _pick_peaks,
    _regular_grid,
    estimate_kick_snare_grids,
    prep_audio,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "groove_classifier.joblib"

# classifier label -> drum-pattern groove (names in audio_operations._drum_pattern)
LABEL_TO_GROOVE: dict[str, str] = {
    "house": "house",
    "techno": "techno",
    "trance": "trance",
    "bigroom": "big_room",
    "dubstep": "dubstep",
    "trap": "trap",
    "dnb": "dnb",
    "phonk": "phonk",
    "hiphop": "half_time",
    "pop": "default",
    "disco": "house",
    "garage": "garage",
    "amapiano": "amapiano",
    "afro_house": "afro_house",
    "jungle": "jungle",
    "grime": "grime",
}

AUTO_GROOVE_MIN_CONFIDENCE = 0.45

_ARTIFACT: dict[str, Any] | None = None
_ARTIFACT_LOADED = False

_CACHE: dict[tuple, dict] = {}


def _load_artifact() -> dict[str, Any] | None:
    global _ARTIFACT, _ARTIFACT_LOADED
    if _ARTIFACT_LOADED:
        return _ARTIFACT
    _ARTIFACT_LOADED = True
    if not MODEL_PATH.exists():
        return None
    try:
        _ARTIFACT = joblib.load(MODEL_PATH)
    except Exception:
        _ARTIFACT = None
    return _ARTIFACT


def groove_model_available() -> bool:
    return _load_artifact() is not None


def swing_score(y: np.ndarray, sr: int, beats: list[float], tempo: float) -> float:
    """0 = straight 8ths, 1 = hard triplet swing. From hat-band onset timing."""
    try:
        y, sr = prep_audio(y, sr)
        hat_env, fr = _band_env(y, sr, 6000.0, None)
        hats = _pick_peaks(hat_env, fr, min_dist_sec=0.06, thresh_k=0.6)
        if len(hats) < 8 or tempo <= 0:
            return 0.0
        step = 60.0 / tempo
        offs: list[float] = []
        for b in beats:
            # offbeat 8th window: expect hats near b + step/2
            near = hats[(hats >= b + step * 0.3) & (hats <= b + step * 0.85)]
            for h in near:
                offs.append((float(h) - b) / step)
        if len(offs) < 4:
            return 0.0
        med = float(np.median(offs))
        return float(max(0.0, min(1.0, (med - 0.5) / 0.17)))
    except Exception:
        return 0.0


def hats_density(y: np.ndarray, sr: int) -> float:
    """Hat-band onsets per second (fast rollers vs sparse halftime)."""
    try:
        y, sr = prep_audio(y, sr)
        hat_env, fr = _band_env(y, sr, 6000.0, None)
        hats = _pick_peaks(hat_env, fr, min_dist_sec=0.05, thresh_k=0.7)
        dur = len(y) / sr
        return float(len(hats) / max(dur, 0.5))
    except Exception:
        return 0.0


def rhythm_feature_vector(y: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """Timbre-invariant feel vector: only tempo/pulse structure, no MFCC or
    spectral tone features — so a model trained on synth loops still reads
    real songs (and vice versa)."""
    from server.ml.rhythm.features import _band_env, detect_hits

    y, sr = prep_audio(y, sr)
    grid = estimate_kick_snare_grids(y, sr)
    tempo = grid["tempo"]
    beats = grid["beats"]
    sw = swing_score(y, sr, beats, tempo)
    hd = hats_density(y, sr)

    hits = detect_hits(y, sr)
    dur = max(len(y) / sr, 0.5)
    kick_rate = float(len(hits["kicks"]) / dur)
    snare_rate = float(len(hits["snares"]) / dur)
    sub_env = hits["sub_env"]
    sub_ratio = float(np.mean(sub_env) / (np.mean(np.abs(y)) + 1e-9))

    feats = [tempo, grid["fourfloor"], grid["halftime"], sw, hd, sub_ratio,
             kick_rate, snare_rate]
    info = {"tempo": tempo, "fourfloor": grid["fourfloor"],
            "halftime": grid["halftime"], "swing": sw, "hats_density": hd,
            "kick_rate": kick_rate, "snare_rate": snare_rate,
            "method": grid["method"]}
    return np.asarray(feats, dtype=np.float32), info


def predict_groove_label(y: np.ndarray, sr: int) -> dict[str, Any] | None:
    """Classify the electronic subgenre feel. None when model unavailable."""
    artifact = _load_artifact()
    if artifact is None:
        return None
    try:
        feats, info = rhythm_feature_vector(y, sr)
        if feats.shape[0] != int(artifact["feature_size"]):
            return None
        model = artifact["model"]
        probs = model.predict_proba(feats.reshape(1, -1))[0]
        classes = list(artifact["classes"])
        idx = int(np.argmax(probs))
        return {"label": classes[idx], "confidence": float(probs[idx]),
                "probs": {c: float(p) for c, p in zip(classes, probs)},
                "features": info}
    except Exception:
        return None


def heuristic_groove(y: np.ndarray, sr: int, grid: dict, swing: float,
                     hats: float) -> tuple[str, float]:
    """Rule fallback when the classifier is missing or unsure."""
    tempo = grid["tempo"]
    four = grid["fourfloor"]
    half = grid["halftime"]
    try:
        from server.ml.rhythm.features import detect_hits
        _hits = detect_hits(y, sr)
        _dur = max(len(_hits["y"]) / _hits["sr"], 0.5)
        kick_rate = len(_hits["kicks"]) / _dur
    except Exception:
        kick_rate = 0.0
    if 107.0 <= tempo <= 117.0 and hats >= 5.0 and swing < 0.35:
        return ("amapiano", 0.55)
    if half >= 0.5:
        if tempo >= 165:
            return ("dnb", 0.55)
        if swing >= 0.4 and 125.0 <= tempo <= 152.0:
            return ("phonk", 0.55)
        if tempo >= 135:
            # triplet-hat trap vs wobble dubstep: hats decide
            return ("trap" if hats >= 3.0 else "dubstep", 0.55)
        return ("trap", 0.5)
    if tempo >= 160:
        # chopped jungle break vs steady dnb roller: kick density decides
        return ("jungle" if kick_rate >= 3.5 else "dnb", 0.55)
    if swing >= 0.4 and 128.0 <= tempo <= 142.0 and four < 0.7:
        return ("garage", 0.55)
    if 0.4 <= four < 0.7 and 118.0 <= tempo <= 126.0 and 0.15 <= swing <= 0.5:
        return ("afro_house", 0.5)
    if four >= 0.55:
        return ("house" if tempo < 135 else "big_room", 0.6)
    if tempo >= 165:
        return ("dnb", 0.6)
    if tempo >= 128 and hats >= 4.0:
        return ("techno", 0.5)
    if swing >= 0.5 and tempo < 115:
        return ("half_time", 0.55)
    if swing >= 0.45 and 115 <= tempo <= 150:
        return ("phonk", 0.5)
    return ("default", 0.4)


def auto_groove(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Pick the drum-pattern groove matching the song. Always returns one."""
    y, sr = prep_audio(y, sr)
    grid = estimate_kick_snare_grids(y, sr)
    sw = swing_score(y, sr, grid["beats"], grid["tempo"])
    hd = hats_density(y, sr)
    pred = predict_groove_label(y, sr)
    if pred and pred["confidence"] >= AUTO_GROOVE_MIN_CONFIDENCE:
        groove = LABEL_TO_GROOVE.get(pred["label"], "default")
        return {"groove": groove, "confidence": round(pred["confidence"], 3),
                "source": "classifier", "label": pred["label"],
                "tempo": grid["tempo"], "swing": sw}
    groove, conf = heuristic_groove(y, sr, grid, sw, hd)
    return {"groove": groove, "confidence": conf, "source": "heuristic",
            "label": None, "tempo": grid["tempo"], "swing": sw}


def analyze_rhythm(y: np.ndarray, sr: int, full_duration: float | None = None) -> dict[str, Any]:
    """Full rhythm read: tempo, beat grid over the whole track, feel, groove.

    ``full_duration`` extends the grid past the analysis window (the estimator
    only listens to the first 60 s); defaults to the audio length.
    """
    y_arr = np.asarray(y, dtype=np.float32)
    mono = np.mean(y_arr, axis=0) if y_arr.ndim > 1 else y_arr
    grid = estimate_kick_snare_grids(mono, sr)
    dur = float(full_duration if full_duration else len(mono) / sr)
    step = 60.0 / grid["tempo"] if grid["tempo"] > 0 else 0.5
    beats = _regular_grid(grid["anchor"] % step, step, dur)
    sw = swing_score(mono, sr, grid["beats"], grid["tempo"])
    hd = hats_density(mono, sr)
    auto = auto_groove(mono, sr)
    return {"tempo_bpm": round(float(grid["tempo"]), 1),
            "beats": [round(float(b), 4) for b in beats],
            "beat_count": len(beats),
            "anchor_sec": round(float(grid["anchor"]), 3),
            "four_on_floor": round(float(grid["fourfloor"]), 3),
            "halftime": round(float(grid["halftime"]), 3),
            "swing": round(float(sw), 3),
            "hats_per_sec": round(float(hd), 2),
            "method": grid["method"],
            "auto_groove": auto["groove"],
            "groove_confidence": auto["confidence"],
            "groove_source": auto["source"]}


def cache_key(y: np.ndarray, sr: int) -> tuple:
    a = np.asarray(y)
    head = np.ascontiguousarray(a[..., :65536]).tobytes()
    return (a.shape, sr, hash(head))
