"""Music genre prediction from a lightweight trained classifier.

Loads the artifact trained by `scripts/train_genre_classifier.py` (a random
forest over librosa features) and predicts the dominant genre of an audio file.
Also maps the predicted genre to a set of suggested tuning actions so the UI can
offer users concrete, genre-appropriate edits.

The classifier is optional: if no artifact exists, predictions return None and
callers fall back to existing heuristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import librosa
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "genre_classifier.joblib"

GENRE_TUNING_ACTIONS: dict[str, list[str]] = {
    "blues": [
        "Isolate the vocals",
        "Warm up the tone with more low-mid presence",
        "Add a slow, soulful reverb to the guitar",
    ],
    "classical": [
        "Give it a concert-hall reverb",
        "Make the dynamics wider and more dramatic",
        "Isolate the strings section",
    ],
    "country": [
        "Brighten the acoustic guitar",
        "Bring the vocals forward",
        "Add a light slapback echo to the voice",
    ],
    "disco": [
        "Give it a four-on-the-floor groove feel",
        "Boost the bass and pump the sidechain",
        "Isolate the drums and tighten them",
    ],
    "hiphop": [
        "Punch up the kick and 808 bass",
        "Make the beat hit harder",
        "Keep the vocals, remove the beat for an acapella",
    ],
    "jazz": [
        "Add a smoky, intimate room feel",
        "Warm up the brass",
        "Isolate the upright bass",
    ],
    "metal": [
        "Gain up the guitars",
        "Tighten the kick drums",
        "Cut the mud below 100 Hz",
    ],
    "pop": [
        "Make it louder and more radio-ready",
        "Punch up the chorus",
        "Isolate the vocals for a clean lead",
    ],
    "reggae": [
        "Give the bass that deep, rubbery groove",
        "Add spring reverb to the rhythm guitar",
        "Sit the vocals back in the mix",
    ],
    "rock": [
        "Make the drums punchy",
        "Bring up the electric guitar",
        "Add a room sound for live feel",
    ],
    "house": [
        "Give it a four-on-the-floor groove feel",
        "Pump the sidechain on the bass",
        "Add a warm house synth stab",
    ],
    "techno": [
        "Drive the kick four-on-the-floor",
        "Add a dark rolling techno bass",
        "Make the drums punchy",
    ],
    "trance": [
        "Add an uplifting trance arp",
        "Give the chorus more energy",
        "Add reverb for that big-room wash",
    ],
    "trap": [
        "Add an 808 slide bass",
        "Make the hi-hats roll faster",
        "Punch up the kick and 808 bass",
    ],
    "dubstep": [
        "Add a dubstep wobble bass",
        "Make the drop hit harder",
        "Convert to dubstep style",
    ],
    "dnb": [
        "Add a fast drum-and-bass break",
        "Layer a reese bass underneath",
        "Make the drums punchy",
    ],
    "phonk": [
        "Add a phonk cowbell lead",
        "Give the bass that deep 808 groove",
        "Slow it down to a drift tempo",
    ],
    "synthwave": [
        "Add a retro synthwave pad",
        "Add a square-wave lead",
        "Give it an 80s gated-drum feel",
    ],
    "edm": [
        "Add a big-room EDM stab",
        "Add a supersaw lead",
        "Convert to big-room festival style",
    ],
    "garage": [
        "Add a shuffled 2-step garage beat",
        "Swing the hi-hats",
        "Add a warpy sub bass",
    ],
    "amapiano": [
        "Add a log-drum bassline",
        "Drive the shaker groove",
        "Soften the kick four-on-the-floor",
    ],
    "afro_house": [
        "Add syncopated conga percussion",
        "Deepen the four-on-the-floor groove",
        "Add warm chord chops",
    ],
    "jungle": [
        "Chop up the break",
        "Add a fast jungle break",
        "Layer a heavy sub underneath",
    ],
    "grime": [
        "Make the beat half-time and stark",
        "Add an eski square lead",
        "Boom the kick on the one",
    ],
}

_ARTIFACT: dict[str, Any] | None = None
_ARTIFACT_LOADED = False


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


def _extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    features: list[float] = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([float(np.mean(spectral_centroid)), float(np.std(spectral_centroid))])

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth))])

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend([float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff))])

    zero_crossing = librosa.feature.zero_crossing_rate(y)
    features.extend([float(np.mean(zero_crossing)), float(np.std(zero_crossing))])

    rms = librosa.feature.rms(y=y)
    features.extend([float(np.mean(rms)), float(np.std(rms))])

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(np.atleast_1d(tempo)[0]))

    return np.asarray(features, dtype=np.float32)


def predict_genre(audio_path: str) -> dict[str, Any] | None:
    """Return the predicted genre + confidence for an audio file, or None if the
    model is unavailable or the prediction is too weak."""
    artifact = _load_artifact()
    if artifact is None:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        if y.ndim > 1:
            y = y[0]
        if y.size == 0:
            return None
        feats = _extract_features(y, sr).reshape(1, -1)
        if feats.shape[1] != int(artifact["feature_size"]):
            return None
        model = artifact["model"]
        probs = model.predict_proba(feats)[0]
        classes = artifact["classes"]
        idx = int(np.argmax(probs))
        genre = classes[idx]
        confidence = float(probs[idx])
        if confidence < 0.3:
            return None
        return {
            "genre": genre,
            "confidence": round(confidence, 3),
            "suggested_actions": GENRE_TUNING_ACTIONS.get(genre, []),
        }
    except Exception:
        return None


def genre_tuning_actions(genre: str | None) -> list[str]:
    if not genre:
        return []
    return GENRE_TUNING_ACTIONS.get(genre, [])


def model_available() -> bool:
    return _load_artifact() is not None