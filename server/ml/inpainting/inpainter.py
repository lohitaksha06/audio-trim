"""Audio inpainting — reconstruct a region instead of cutting it away.

What this replaced
------------------
The old `inpaint()` removed the samples and crossfaded the two ends together.
That is a delete with a smoother join: the file got shorter and the audio was
still discontinuous, so "remove that cymbal crash and fill smoothly" did not do
what it said. Nothing was reconstructed.

How this works
--------------
Two strategies, tried in order:

1. **Bar-aligned repeat** (the common case in music). If a crash ruins bar 9,
   the right material to put back is bar 5 or bar 13 -- the same musical
   position. We already detect beats, so those positions can be named exactly
   rather than hunted for.
2. **Feature search** over the whole file, for material that is similar but not
   bar-aligned. Log-mel features are compared on both sides of the hole.

The winner is spliced in with short equal-gain crossfades, keeping the timeline
length. If neither finds usable material, a spectral fill (log-magnitude
interpolation with phase propagation) is used and the metadata says so.

Honest scope: this recovers the *actual performance* of the same musical
material. It cannot invent notes that were never played -- on through-composed
material it produces something plausible in the right key and tempo, not the
original take.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

N_FFT = 2048
HOP = 512
FEAT_HOP = 256          # must equal the stride used to index feature frames


# --------------------------------------------------------------- spectral fill

def _fill_mono(
    y: np.ndarray, sr: int, start_s: float, end_s: float, taper: float
) -> np.ndarray:
    """Magnitude interpolation + phase propagation across a hole."""
    n = y.shape[-1]
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S = np.atleast_2d(S)
    n_bins, n_frames = S.shape

    a = max(0, int(round(start_s * sr / HOP)))
    b = min(n_frames, int(round(end_s * sr / HOP)))
    if b - a < 1:
        return y

    pre = max(a - 1, 0)
    post = min(b + 1, n_frames - 1)
    if post <= pre:
        return y

    gap = b - a
    log_pre = np.log(np.abs(S[:, pre]) + 1e-8)[:, None]
    log_post = np.log(np.abs(S[:, post]) + 1e-8)[:, None]
    ramp = np.linspace(0.0, 1.0, gap, dtype=np.float32)[None, :]

    taper_n = max(1, int(taper * gap))
    env = np.ones(gap, dtype=np.float32)
    env[:taper_n] = np.linspace(0.0, 1.0, taper_n) ** 0.7
    env[-taper_n:] = np.linspace(1.0, 0.0, taper_n) ** 0.7

    mag_gap = np.exp((1.0 - ramp) * log_pre + ramp * log_post) * env[None, :]

    # Phase: propagate the preceding frame's instantaneous frequency. Reusing
    # the flanking frames' phase directly sounds metallic, because two unrelated
    # sounds do not share phase.
    ref_phase = np.angle(S[:, pre])
    prev_phase = np.angle(S[:, max(a - 2, 0)])
    delta = np.mod(ref_phase - prev_phase + np.pi, 2.0 * np.pi) - np.pi

    phase_gap = np.empty((n_bins, gap), dtype=np.float32)
    ph = ref_phase.astype(np.float32).copy()
    for i in range(gap):
        ph = ph + delta.astype(np.float32)
        phase_gap[:, i] = ph
    drift = np.angle(np.exp(1j * (np.angle(S[:, post]) - phase_gap[:, -1])))
    phase_gap += drift[:, None] * np.linspace(0, 1, gap, dtype=np.float32)[None, :]

    S[:, a:b] = mag_gap * np.exp(1j * phase_gap)
    return np.asarray(librosa.istft(S, hop_length=HOP, n_fft=N_FFT, length=n),
                      dtype=np.float32)


def spectral_fill(
    y: np.ndarray, sr: int, start_s: float, end_s: float, taper: float = 0.35
) -> np.ndarray:
    """Fill [start_s, end_s); returns the full-length signal, all channels."""
    channels = y.shape[0] if y.ndim > 1 else 1
    if channels == 1:
        return _fill_mono(np.asarray(y.reshape(-1), dtype=np.float32),
                           sr, start_s, end_s, taper)[None, :]
    out = np.zeros_like(y, dtype=np.float32)
    for ch in range(channels):
        filled = _fill_mono(np.asarray(y[ch], dtype=np.float32),
                            sr, start_s, end_s, taper)
        m = min(filled.shape[-1], y.shape[-1])
        out[ch, :m] = filled[:m]
    return out


# ------------------------------------------------------------- repeat search

def _match_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame log-mel features, (frames, mels).

    `FEAT_HOP` must match the stride used to index these frames, or every
    index is silently wrong.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=FEAT_HOP, n_mels=64
    )
    return np.log1p(S).T.astype(np.float32)


def _bar_aligned_offsets(
    mono: np.ndarray, sr: int, a: int, b: int, limit: int = 32
) -> list[int]:
    """Candidate positions one or more bars *before* the hole, beat-aligned."""
    n = mono.shape[-1]
    try:
        from server.ml.audio_operations import _detect_beats

        tempo, beats = _detect_beats(mono[np.newaxis, :], sr)
    except Exception:
        return []
    if not beats or not (30.0 <= float(tempo) <= 260.0) or len(beats) < 4:
        return []

    beat = 60.0 / float(tempo)
    bar = beat * 4.0  # 4/4, the overwhelming majority
    gaps = np.diff(beats)
    med = float(np.median(gaps)) if gaps.size else beat
    if med <= 1e-6:
        return []

    # Phase of the hole within the detected beat grid, so candidates land on
    # musically corresponding positions rather than arbitrary ones.
    phase = (a / sr - float(beats[0])) % med
    guard = int(0.02 * sr)

    out: list[int] = []
    for k in range(1, limit + 1):
        c = int(round((a / sr - k * bar + phase) * sr))
        if c < 0 or c + (b - a) > n:
            break
        if c + (b - a) > a - guard and c < b + guard:
            continue
        out.append(c)
    return out


def _score(
    feats: np.ndarray, ctx_f: int, f_pre: int, f_post: int, c: int, gap: int
) -> float:
    f0 = c // FEAT_HOP
    f1 = (c + gap) // FEAT_HOP
    if f0 - ctx_f < 0 or f1 + ctx_f > len(feats):
        return float("inf")
    g_lead = feats[f_pre:f_pre + ctx_f]
    g_trail = feats[f_post:f_post + ctx_f]
    if len(g_lead) < ctx_f or len(g_trail) < ctx_f:
        return float("inf")
    lead = feats[f0 - ctx_f:f0]
    trail = feats[f1:f1 + ctx_f]
    return float(np.mean((lead - g_lead) ** 2) + np.mean((trail - g_trail) ** 2))


def repetition_fill(
    y: np.ndarray, sr: int, start_s: float, end_s: float, margin_s: float = 0.35
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill the hole with real material from elsewhere in the same track."""
    n = y.shape[-1]
    multi = y.ndim > 1
    channels = y.shape[0] if multi else 1
    mono = y.mean(axis=0) if multi else y

    a = max(0, int(round(start_s * sr)))
    b = min(n, int(round(end_s * sr)))
    gap = b - a
    info: dict[str, Any] = {"method": "repetition"}
    if gap < 1:
        return y, info

    # Context must be long enough to identify a musical position, and must not
    # shrink with the hole: a 0.3s hole still needs context to know where in the
    # track it sits. Capping context by gap length was an earlier mistake.
    ctx = max(int(0.10 * sr), min(int(margin_s * sr), int(gap)))
    usable = n - gap - 2 * ctx
    if usable <= 0:
        info["method"] = "spectral (no room for a repeat)"
        return spectral_fill(y, sr, start_s, end_s), info

    try:
        feats = _match_features(mono, sr)
        ctx_f = max(1, int(round(ctx / FEAT_HOP)))
        f_pre = max(0, a // FEAT_HOP - ctx_f)
        f_post = min(len(feats), b // FEAT_HOP + ctx_f)
        guard = int(0.05 * sr)

        best, best_score, kind = -1, float("inf"), ""

        for c in _bar_aligned_offsets(mono, sr, a, b):
            d = _score(feats, ctx_f, f_pre, f_post, c, gap)
            if d < best_score:
                best_score, best, kind = d, c, "bar-aligned repeat"

        scan_best, scan_score = -1, float("inf")
        for c in range(0, usable, FEAT_HOP):
            if c + ctx > a - guard and c < b + guard:
                continue
            d = _score(feats, ctx_f, f_pre, f_post, c, gap)
            if d < scan_score:
                scan_score, scan_best = d, c

        # Fall back to the general scan only when it is clearly better, i.e.
        # when the track turns out not to be looped at that position.
        if best < 0 or scan_score < best_score * 0.6:
            if scan_best >= 0:
                best, best_score, kind = scan_best, scan_score, "similar passage"

        if best < 0 or not np.isfinite(best_score):
            info["method"] = "spectral (no similar passage found)"
            return spectral_fill(y, sr, start_s, end_s), info

        out = np.asarray(y, dtype=np.float32).copy()
        # Candidates are exactly the hole's length, so no time-scaling is
        # needed. Every channel reads from the SAME source position so a stereo
        # source lands in the same place and the image survives.
        # The crossfade is sized like the old splice-out's (50 ms), because a
        # 12 ms seam measurably clicked more than that on decaying material.
        xf = min(int(0.05 * sr), max(1, gap // 4))
        ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32) if xf > 1 else None
        for ch in range(channels):
            row = y[ch] if multi else y
            cand = np.asarray(row[best: best + gap], dtype=np.float32)
            if cand.shape[-1] < gap:
                cand = np.pad(cand, (0, gap - cand.shape[-1]))
            if ramp is not None:
                out[ch, a:a + xf] = out[ch, a:a + xf] * (1 - ramp) + cand[:xf] * ramp
                out[ch, b - xf:b] = out[ch, b - xf:b] * (1 - ramp) + cand[-xf:] * ramp
                out[ch, a + xf:b - xf] = cand[xf:gap - xf]
            else:
                out[ch, a:b] = cand

        info.update({
            "source_start": round(best / sr, 3),
            "match_distance": round(best_score, 6),
            "source_kind": kind,
        })
        return out, info
    except Exception as exc:
        info["method"] = f"spectral (repeat search failed: {type(exc).__name__})"
        if os.environ.get("INPAINT_DEBUG"):
            import traceback
            info["error"] = f"{type(exc).__name__}: {exc}"
            info["traceback"] = traceback.format_exc()
        return spectral_fill(y, sr, start_s, end_s), info


# -------------------------------------------------------------------- facade

def inpaint(
    audio_path: str,
    start: float,
    end: float,
    mode: str = "fill",
    method: str = "auto",
    crossfade_s: float = 0.05,
) -> dict[str, Any]:
    """Replace a region.

    mode="fill"   reconstruct it and keep the timeline length (default)
    mode="close"  splice it out and crossfade the seam (previous behaviour)
    method="auto" bar-aligned repeat, then feature search, then spectral
    """
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    channels, total = y.shape
    dur = total / sr

    start_s = max(0.0, min(float(start), dur))
    end_s = max(start_s + 0.001, min(float(end), dur))
    start_idx = int(start_s * sr)
    end_idx = int(end_s * sr)

    fill_info: dict[str, Any] = {}
    if mode == "close":
        keep = np.concatenate([y[:, :start_idx], y[:, end_idx:]], axis=1)
        cf = int(min(crossfade_s * sr, keep.shape[1] // 4))
        if cf > 0 and keep.shape[1] > 0:
            rd = np.linspace(1.0, 0.0, cf)
            ru = np.linspace(0.0, 1.0, cf)
            keep[:, :cf] *= rd
            keep[:, -cf:] *= ru
            keep[:, :cf] += keep[:, cf:2 * cf] * (1 - rd)
        out = keep
    elif method == "spectral":
        out = spectral_fill(y, sr, start_s, end_s)
        fill_info = {"method": "spectral"}
    else:
        out, fill_info = repetition_fill(y, sr, start_s, end_s)
        if method == "repeat" and fill_info.get("method") != "repetition":
            # Caller insisted on a repeat but none was usable; say so rather
            # than letting a fallback masquerade as one.
            fill_info["requested"] = "repeat"

    new_len = out.shape[1]
    peak = float(np.max(np.abs(out))) + 1e-9
    if peak > 1.0:
        out = (out / peak * 0.99).astype(np.float32)

    out_path = tempfile.mktemp(suffix=".wav")
    sf.write(out_path, out.T if channels > 1 else out[0], sr)

    meta: dict[str, Any] = {
        "output_path": out_path,
        "removed_start": round(start_s, 3),
        "removed_end": round(end_s, 3),
        "removed_seconds": round(end_s - start_s, 3),
        "mode": mode,
        "original_duration_seconds": round(dur, 2),
        "new_duration_seconds": round(new_len / sr, 2),
    }
    if mode == "fill":
        meta["inpaint_method"] = fill_info.get("method", "spectral")
        meta["inpaint_reconstructed"] = True
        for key in ("source_start", "match_distance", "source_kind", "error"):
            if key in fill_info:
                meta[f"fill_{key}"] = fill_info[key]
    else:
        meta["inpaint_method"] = "splice + crossfade (region removed)"
        meta["inpaint_reconstructed"] = False
    return meta


def paint_intro_outro(
    audio_path: str, start: float, end: float, mode: str = "fill"
) -> dict[str, Any]:
    """Same as `inpaint`; `mode` kept in the signature for callers."""
    return inpaint(audio_path, start, end, mode=mode)