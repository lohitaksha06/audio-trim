"""Mood / energy curve across time.

Computes per-window measures (energy, tension, brightness, loudness) and
summarizes the overall mood with a keyword description. All derived from
librosa low-level features so it runs fast on CPU with no model downloads.
"""

from typing import Any

import librosa
import numpy as np


def compute_mood_curve(
    audio_path: str, hop_seconds: float = 0.5
) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    hop = int(hop_seconds * sr)

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    spec_flat = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    n = min(rms.shape[0], sc.shape[0], spec_flat.shape[0], flux.shape[0])
    times = np.arange(n) * hop_seconds

    energy = rms[:n]
    brightness = sc[:n] / (sr / 2)
    tension = flux[:n] / (flux[:n].max() + 1e-9)
    calmness = 1.0 - np.clip(
        (spec_flat[:n] - spec_flat[:n].min()) / (np.ptp(spec_flat[:n]) + 1e-9), 0, 1
    )

    loudness_db = 20 * np.log10(energy + 1e-9)
    loudness_db = np.clip(loudness_db - loudness_db.min(), 0, None)

    curve = [
        {
            "t": round(float(t), 3),
            "energy": round(float(e), 4),
            "tension": round(float(tn), 4),
            "brightness": round(float(br), 4),
            "loudness": round(float(lv), 2),
        }
        for t, e, tn, br, lv in zip(
            times[::5], energy[::5], tension[::5], brightness[::5], loudness_db[::5]
        )
    ]

    return {
        "curve": curve,
        "sample_rate": sr,
        "hop_seconds": hop_seconds,
        "duration_seconds": round(duration, 2),
    }


def describe_mood(audio_path: str) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y)[0]
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    mean_energy = float(np.mean(rms))
    centroid = float(np.mean(sc))
    tension = float(np.mean(np.abs(librosa.feature.delta(rms))))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

    mood = "neutral"
    rules: list[str] = []
    if centroid < 1700 and mean_energy < 0.08:
        mood, rules = "dark", ["warm, low-brightness timbre"]
    elif centroid > 3600 and mean_energy < 0.08:
        mood, rules = "calm", ["soft, bright timbre"]
    elif mean_energy > 0.25 and tension > 0.02:
        mood, rules = "energetic", ["high energy with dynamic variation"]
    elif mean_energy < 0.05:
        mood, rules = "calm", ["gentle, low dynamics"]
    else:
        rules = ["balanced dynamics"]

    if spectral_flatness < 0.08:
        rules.append("clearly pitched / tonal")

    return {
        "mood": mood,
        "energy_mean": round(mean_energy, 4),
        "brightness_mean": round(centroid, 1),
        "tension_mean": round(tension, 4),
        "spectral_flatness": round(spectral_flatness, 4),
        "description": " ".join(rules),
        "duration_seconds": round(duration, 2),
    }