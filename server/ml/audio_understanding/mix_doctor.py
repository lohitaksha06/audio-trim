"""Mix Doctor — honest, measurement-based optimization tips.

Analyzes loudness, clipping, dynamics, spectral balance, stereo width and
silence, then returns concrete findings each with a one-click ``fix_prompt``
the user can paste into the Editor (or tap Apply in Mix Lab).

No ML model needed: pure librosa/numpy DSP so it runs fast on CPU and in CI.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np


def _band_ratio(S: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    freqs = librosa.fft_frequencies(sr=sr)
    total = float(np.sum(np.abs(S) ** 2)) + 1e-12
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(np.abs(S[band]) ** 2) / total)


def analyze_mix(audio_path: str) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    mono = y.mean(axis=0) if y.shape[0] > 1 else y[0]
    dur = len(mono) / sr if sr else 0.0

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    crest_db = round(float(20 * np.log10(peak / (rms + 1e-9))), 1) if rms > 0 else 0.0
    clipped = float(np.mean(np.abs(mono) >= 0.999)) if mono.size else 0.0

    S = np.abs(librosa.stft(mono)) if mono.size else np.zeros((1, 1))
    bass = _band_ratio(S, sr, 20, 250)
    lowmid = _band_ratio(S, sr, 250, 2000)
    presence = _band_ratio(S, sr, 2000, 6000)
    air = _band_ratio(S, sr, 6000, sr / 2)

    # dynamics: quiet-loud spread across 0.5s windows
    hop = max(1, int(sr * 0.5))
    wins = [mono[i: i + hop] for i in range(0, len(mono), hop) if len(mono[i: i + hop]) > hop // 2]
    win_rms = np.array([float(np.sqrt(np.mean(w ** 2))) + 1e-9 for w in wins]) if wins else np.array([1e-9])
    dyn_db = round(float(20 * np.log10(np.percentile(win_rms, 95) / np.percentile(win_rms, 10))), 1)

    # stereo width: mid/side energy ratio
    width = 0.0
    if y.shape[0] > 1:
        n = min(y.shape[1], len(mono))
        side = (y[0, :n] - y[1, :n]) / 2.0
        width = float(np.sqrt(np.mean(side ** 2)) / (rms + 1e-9))

    # leading/trailing silence (> -40dB)
    intervals = librosa.effects.split(mono, top_db=40) if mono.size else np.zeros((0, 2))
    trim_hint = 0.0
    if len(intervals) > 0:
        trim_hint = round(float(intervals[0][0]) / sr, 2)

    tips: list[dict[str, Any]] = []

    def add(severity: str, title: str, detail: str, fix_prompt: str) -> None:
        tips.append({"severity": severity, "title": title, "detail": detail, "fix_prompt": fix_prompt})

    if clipped > 0.005:
        add("high", "Clipping detected",
            f"{clipped * 100:.1f}% of samples hit 0 dBFS — distortion on loud parts.",
            "Normalize the volume")
    elif peak > 0.98:
        add("medium", "Hot master",
            f"Peak {peak:.2f} dBFS leaves no headroom for export encoding.",
            "Set gain to -2dB")
    if rms < 0.03:
        add("medium", "Quiet mix",
            f"RMS {rms:.3f} is low — the track will sound buried next to references.",
            "Normalize the volume")
    if crest_db > 22:
        add("medium", "Spiky dynamics",
            f"Crest factor {crest_db} dB — big gap between peaks and body; glue it.",
            "Make the drums punchy")
    if dyn_db < 3:
        add("low", "Flat dynamics",
            f"Only {dyn_db} dB between quiet and loud sections — the song never breathes.",
            "Make the chorus more energetic")
    if bass < 0.08:
        add("medium", "Thin low end",
            f"Only {bass * 100:.0f}% of energy below 250 Hz — kick/bass need weight.",
            "Make the bass louder")
    if bass > 0.55:
        add("medium", "Muddy low end",
            f"{bass * 100:.0f}% of energy is boom below 250 Hz — it masks the vocals.",
            "Make the vocals clearer")
    if presence < 0.08:
        add("medium", "Buried vocals",
            f"Presence band (2-6 kHz) is only {presence * 100:.0f}% — voices won't cut through.",
            "Make voices clearer and remove background noise")
    if air < 0.02:
        add("low", "Dull top end",
            "Almost no air above 6 kHz — cymbals and vocal shimmer are missing.",
            "Make it brighter")
    if air > 0.35:
        add("low", "Harsh highs",
            f"{air * 100:.0f}% of energy is hiss above 6 kHz — likely noise or harsh cymbals.",
            "Enhance vocals and denoise")
    if y.shape[0] > 1 and width < 0.05:
        add("low", "Narrow stereo",
            "Left and right are nearly identical — the mix could feel wider.",
            "Add reverb")
    if trim_hint > 1.0:
        add("low", "Dead air at the start",
            f"First {trim_hint:.1f}s is near-silence — trim it for a tighter intro.",
            f"Trim from 0:{trim_hint:04.1f} to end")
    if dur > 5 and not tips:
        add("good", "Mix sounds healthy",
            f"Peak {peak:.2f}, RMS {rms:.3f}, crest {crest_db} dB, dynamics {dyn_db} dB — no red flags.",
            "Convert this song into a house music style")

    summary = {
        "peak": round(peak, 3),
        "rms": round(rms, 4),
        "crest_db": crest_db,
        "dynamics_db": dyn_db,
        "clipped_pct": round(clipped * 100, 2),
        "bands": {"bass": round(bass, 3), "lowmid": round(lowmid, 3),
                  "presence": round(presence, 3), "air": round(air, 3)},
        "stereo_width": round(width, 3),
        "channels": int(y.shape[0]),
        "duration_seconds": round(float(dur), 2),
        "sample_rate": int(sr),
    }
    score = 100
    for t in tips:
        score -= {"high": 20, "medium": 10, "low": 5, "good": 0}[t["severity"]]
    return {"summary": summary, "score": max(0, score), "tips": tips}
