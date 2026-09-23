"""Audio-content understanding: noise-condition classification + SNR regression.

Two supervised models trained by `scripts/train_content_classifier.py`
(real GTZAN music + synthetic hiss/hum/crowd/clip/muffle degradations):
  - clf: RandomForestClassifier over {clean,hiss,hum,crowd,clipped,muffled}
  - reg: RandomForestRegressor predicting effective SNR_dB (quality)

This is the "what kind of audio is this / what noise is on it" head. It is
purely discriminative (classification + regression) — the generative/DSP
editing pipeline is untouched. Missing artifact -> predict_content() returns
None and every caller falls back gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "content_classifier.joblib"

CONDITION_ACTIONS: dict[str, str] = {
    "clean": "Mix sounds healthy",
    "hiss": "Enhance vocals and denoise",
    "hum": "Remove background noise",
    "crowd": "Keep only the vocals",
    "clipped": "Normalize the volume",
    "muffled": "Make it brighter",
}

_ARTIFACT: dict[str, Any] | None = None
_LOADED = False


def _load() -> dict[str, Any] | None:
    global _ARTIFACT, _LOADED
    if _LOADED:
        return _ARTIFACT
    _LOADED = True
    if not MODEL_PATH.exists():
        return None
    try:
        _ARTIFACT = joblib.load(MODEL_PATH)
    except Exception:
        _ARTIFACT = None
    return _ARTIFACT


def _extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """84-dim vector. MUST mirror scripts/train_content_classifier.extract_features."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(84, dtype=np.float32)
    feats: list[float] = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma, axis=1))
    feats.extend(np.std(chroma, axis=1))
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats.extend([float(np.mean(cent)), float(np.std(cent))])
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats.extend([float(np.mean(bw)), float(np.std(bw))])
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats.extend([float(np.mean(roll)), float(np.std(roll))])
    zcr = librosa.feature.zero_crossing_rate(y)
    feats.extend([float(np.mean(zcr)), float(np.std(zcr))])
    rms = librosa.feature.rms(y=y)
    feats.extend([float(np.mean(rms)), float(np.std(rms))])
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.append(float(np.atleast_1d(tempo)[0]))
    except Exception:
        feats.append(120.0)
    flat = librosa.feature.spectral_flatness(y=y)
    feats.extend([float(np.mean(flat)), float(np.std(flat))])
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    total = float(np.sum(S ** 2)) + 1e-12
    hum_band = (freqs >= 40) & (freqs <= 130)
    hiss_band = freqs >= 6000
    feats.append(float(np.sum(S[hum_band] ** 2) / total))
    feats.append(float(np.sum(S[hiss_band] ** 2) / total))
    feats.append(float(np.mean(np.abs(y) >= 0.99)))
    peak = float(np.max(np.abs(y))) + 1e-12
    rms_v = float(np.sqrt(np.mean(y ** 2))) + 1e-12
    feats.append(float(20 * np.log10(peak / rms_v)))
    hum_lines = ((freqs >= 48) & (freqs <= 52)) | ((freqs >= 97) & (freqs <= 103)) | ((freqs >= 147) & (freqs <= 153))
    hum_sur = (freqs >= 40) & (freqs <= 160) & (~hum_lines)
    hum_peak = float(np.sum(S[hum_lines] ** 2) + 1e-12) / (float(np.sum(S[hum_sur] ** 2)) + 1e-12)
    feats.append(float(hum_peak))
    frame_rms = librosa.feature.rms(y=y)[0] + 1e-12
    feats.append(float(np.percentile(frame_rms, 10) / (np.median(frame_rms) + 1e-12)))
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats.append(float(np.mean(contrast)))
    except Exception:
        feats.append(0.0)
    return np.asarray(feats, dtype=np.float32)


def predict_content(audio_path: str) -> dict[str, Any] | None:
    """Classify noise condition + regress SNR. None if model unavailable/weak."""
    art = _load()
    if art is None:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30.0)
        if y.size == 0:
            return None
        feats = _extract_features(y, sr).reshape(1, -1)
        if feats.shape[1] != int(art["feature_size"]):
            return None
        probs = art["clf"].predict_proba(feats)[0]
        classes = list(art["classes"])
        idx = int(np.argmax(probs))
        snr = float(art["reg"].predict(feats)[0])
        snr = max(-5.0, min(45.0, snr))
        quality = round(max(0.0, min(100.0, snr / 40.0 * 100.0)), 1)
        return {
            "condition": classes[idx],
            "confidence": round(float(probs[idx]), 3),
            "probs": {c: round(float(p), 3) for c, p in zip(classes, probs)},
            "snr_db": round(snr, 1),
            "quality_score": quality,
            "suggested_action": CONDITION_ACTIONS.get(classes[idx], ""),
        }
    except Exception:
        return None


def model_available() -> bool:
    return _load() is not None
