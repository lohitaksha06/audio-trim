"""Instrument classification from acoustic features.

Lightweight, dependency-light classifier. Produces per-instrument presence
scores (0.0-1.0) using librosa-derived spectral/rhythmic features, plus a
texture description. Designed to run on CPU without downloading a model.
"""

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


def classify_instruments(audio_path: str) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    if y.ndim > 1:
        y = y[0]

    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 1.0:
        return {"instruments": {}, "texture": "too short to analyze"}

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
    }