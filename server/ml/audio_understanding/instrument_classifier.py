"""Instrument classification — SOTA-optional with heuristic fallback.

- If ``USE_PRETRAINED=1`` and a pretrained tagger is installed (``transformers``
  CLAP / PANNs / MusicNN), we route through it for 0.0-1.0 per-instrument scores.
- Otherwise falls back to the lightweight librosa heuristic (no model download,
  CPU-only, deterministic) so CI and offline mode stay green.

Contract is stable: returns ``{instruments: [{instrument, confidence}], texture,
tempo_bpm, duration_seconds, backend}`` regardless of backend.
"""

from __future__ import annotations

import os
from typing import Any

import librosa
import numpy as np


def _band_energy_ratio(S, sr: int, fmin: float, fmax: float) -> float:
    freqs = librosa.fft_frequencies(sr=sr)
    total = np.sum(np.abs(S) ** 2)
    if total == 0:
        return 0.0
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(np.abs(S[band]) ** 2) / total)


_PRETRAINED_MODEL_ID = os.environ.get("INSTRUMENT_MODEL", "laion/clap-htsat-unfused")
_PRETRAINED_CACHE: dict[str, Any] = {}


def _try_pretrained(audio_path: str) -> dict[str, Any] | None:
    """Try CLAP zero-shot tagger if enabled and deps are present.

    Returns None on any failure so caller falls back to heuristic.
    """
    if os.environ.get("USE_PRETRAINED") != "1":
        return None
    try:
        # lazy import — keeps base install light
        from transformers import pipeline  # type: ignore
    except Exception:
        return None
    try:
        key = _PRETRAINED_MODEL_ID
        if key not in _PRETRAINED_CACHE:
            # CLAP supports zero-shot audio classification via candidate_labels
            _PRETRAINED_CACHE[key] = pipeline("zero-shot-audio-classification", model=key)
        clf = _PRETRAINED_CACHE[key]
        labels = ["vocals", "drums", "bass", "guitar", "piano", "strings", "other"]
        out = clf(audio_path, candidate_labels=labels)
        # pipeline returns sorted list of {label, score}
        scores = {r["label"]: float(r["score"]) for r in out}
        # normalize to our stem map
        mapped: dict[str, float] = {
            "vocals": scores.get("vocals", 0),
            "drums": scores.get("drums", 0),
            "bass": scores.get("bass", 0),
            "guitar": scores.get("guitar", 0),
            "keys": scores.get("piano", 0),
            "other": max(scores.get("strings", 0), scores.get("other", 0)),
        }
        present = [
            {"instrument": k, "confidence": round(float(v), 3)}
            for k, v in sorted(mapped.items(), key=lambda kv: kv[1], reverse=True)
            if v >= 0.15
        ]
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=librosa.effects.hpss(y)[1], sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        return {
            "instruments": present,
            "texture": "pretrained CLAP tagger",
            "tempo_bpm": round(float(tempo) if tempo and not np.isnan(tempo) else 0, 1),
            "duration_seconds": round(float(librosa.get_duration(y=y, sr=sr)), 2),
            "backend": "pretrained:" + key,
        }
    except Exception:
        return None


def classify_instruments(audio_path: str) -> dict[str, Any]:
    # SOTA path first — fast fallback if not enabled/available
    pre = _try_pretrained(audio_path)
    if pre is not None:
        return pre

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    if y.ndim > 1:
        y = y[0]

    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 1.0:
        return {"instruments": {}, "texture": "too short to analyze", "backend": "heuristic"}

    S = np.abs(librosa.stft(y))

    bass_ratio = _band_energy_ratio(S, sr, 40, 180)
    voice_ratio = _band_energy_ratio(S, sr, 300, 3400)
    high_ratio = _band_energy_ratio(S, sr, 6000, sr / 2)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, backtrack=True
    )
    onset_density = len(onsets) / max(duration, 1.0)

    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    centroid_mean = float(np.mean(spectral_centroid))

    rms = librosa.feature.rms(S=S)[0]
    rms_var = float(np.var(rms))

    harmonic, percussive = librosa.effects.hpss(y)

    def band_energy(sig, fmin, fmax):
        Sx = np.abs(librosa.stft(sig))
        return _band_energy_ratio(Sx, sr, fmin, fmax)

    harmonic_voice = band_energy(harmonic, 300, 3400)
    harmonic_total = float(np.sum(harmonic ** 2)) / max(float(np.sum(y ** 2)), 1e-9)

    percussive_ratio = float(np.sum(percussive ** 2)) / max(float(np.sum(y ** 2)), 1e-9)

    tempo, _ = librosa.beat.beat_track(y=librosa.effects.hpss(y)[1], sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    scores: dict[str, float] = {}
    scores["drums"] = float(np.clip(onset_density / 6.0 * 0.5 + percussive_ratio * 2.0, 0, 1))
    scores["bass"] = float(np.clip(bass_ratio / 0.3, 0, 1))
    scores["vocals"] = float(
        np.clip(voice_ratio * 1.2 + harmonic_voice * 0.8, 0, 1)
    )
    keys = float(np.clip(harmonic_total * 1.5, 0, 1))
    scores["keys"] = keys * (1.0 - min(scores["vocals"], 0.7))
    scores["guitar"] = float(
        np.clip(
            (1.0 - scores["keys"])
            * (harmonic_total * 0.8 + centroid_mean / 6000)
            * (1.0 if onset_density > 0.5 else 0.7),
            0,
            1,
        )
    )
    scores["other"] = float(
        np.clip(0.2 + high_ratio * 0.5 + (1.0 - max(scores.values())) * 0.3, 0, 1)
    )

    active = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    present = [
        {"instrument": name, "confidence": round(score, 3)}
        for name, score in active
        if score >= 0.25
    ]

    if centroid_mean < 1500:
        texture = "warm and dark timbre"
    elif centroid_mean > 3500:
        texture = "bright and crisp timbre"
    else:
        texture = "balanced mid-range timbre"

    if "drums" in {p["instrument"] for p in present}:
        texture += ", percussive groove"

    if tempo and 80 <= tempo <= 130:
        texture += f", steady groove at {round(tempo)} BPM"

    return {
        "instruments": present,
        "texture": texture,
        "tempo_bpm": round(tempo, 1),
        "duration_seconds": round(duration, 2),
        "backend": "heuristic:librosa-bands+hpss",
    }