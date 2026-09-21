"""Rhythm analysis for beat-synced editing: tempo, beat grid, feel scores.

Why this exists: ``librosa.beat.beat_track`` alone locks onto half/double time
on electronic music (trap read as 75 BPM instead of 150, house drifting off the
four-on-the-floor). This module estimates the grid the way dance music is built:

1. find the sub-bass kick pulses directly (sine drops are unmistakable),
2. decide straight-vs-halftime from where the snare lands relative to kicks,
3. anchor beat 0 on the first kick and extend a regular grid over the track.

Also exposes feel scores (four-on-the-floor, halftime, swing) used both by the
auto-groove classifier and by the supervised calibration in
``scripts/train_groove_classifier.py``.
"""

from __future__ import annotations

import numpy as np
import librosa

_ANALYSIS_SR = 22050
_ANALYSIS_HOP = 512
_MAX_ANALYSIS_SECONDS = 60.0


def prep_audio(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Mono, resampled, capped — the analysis working copy."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=0).astype(np.float32)
    if sr != _ANALYSIS_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_ANALYSIS_SR)
        sr = _ANALYSIS_SR
    n = int(_MAX_ANALYSIS_SECONDS * sr)
    if len(y) > n:
        y = y[:n]
    return y, sr


def _band_env(y: np.ndarray, sr: int, fmin: float, fmax: float | None) -> tuple[np.ndarray, float]:
    """Per-frame energy envelope of a frequency band. Returns (env, frame_rate)."""
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=_ANALYSIS_HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax
    env = np.sum(S[mask], axis=0).astype(np.float32)
    frame_rate = sr / _ANALYSIS_HOP
    return env, frame_rate


def _pick_peaks(env: np.ndarray, frame_rate: float, min_dist_sec: float = 0.15,
                thresh_k: float = 1.0) -> np.ndarray:
    """Peak-pick an envelope. Returns peak times in seconds."""
    if len(env) < 3:
        return np.zeros(0, dtype=np.float32)
    med = float(np.median(env))
    std = float(np.std(env))
    thr = med + thresh_k * std
    cand = np.where((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > thr))[0] + 1
    if len(cand) == 0:
        return np.zeros(0, dtype=np.float32)
    # sub-frame refinement: quadratic interpolation around each peak kills
    # frame-quantization bias (23 ms at hop 512) that otherwise detunes tempo
    # by ~2% and drifts the grid on long tracks.
    refined = []
    for idx in (int(c) for c in cand):
        i = int(idx)
        if 1 <= i < len(env) - 1:
            a, b, cc = float(env[i - 1]), float(env[i]), float(env[i + 1])
            denom = a - 2 * b + cc
            off = 0.5 * (a - cc) / denom if abs(denom) > 1e-9 else 0.0
            off = max(-0.5, min(0.5, off))
        else:
            off = 0.0
        refined.append(off)
    min_dist = int(round(min_dist_sec * frame_rate))
    # greedy loudest-first non-max suppression (integer bins), then emit the
    # sub-frame-refined times
    offset_of = {int(c): off for c, off in zip((int(c) for c in cand), refined)}
    order = np.argsort(env[cand])[::-1]
    kept: list[int] = []
    blocked = np.zeros(len(env), dtype=bool)
    for idx in (int(c) for c in cand[order]):
        if blocked[idx]:
            continue
        kept.append(idx)
        lo, hi = max(0, idx - min_dist), min(len(env), idx + min_dist + 1)
        blocked[lo:hi] = True
    kept.sort()
    return np.asarray([float(i) + offset_of.get(i, 0.0) for i in kept],
                      dtype=np.float32) / frame_rate


def _grid_align(env: np.ndarray, frame_rate: float, ticks: np.ndarray,
                tol_sec: float = 0.07) -> float:
    """Fraction of grid ticks with envelope energy nearby (0..1)."""
    if len(ticks) == 0 or len(env) == 0:
        return 0.0
    med = float(np.median(env))
    std = float(np.std(env))
    thr = med + 0.75 * std
    tol = int(round(tol_sec * frame_rate))
    hits = 0
    for t in ticks:
        i = int(round(float(t) * frame_rate))
        lo, hi = max(0, i - tol), min(len(env), i + tol + 1)
        if lo < hi and float(np.max(env[lo:hi])) > thr:
            hits += 1
    return hits / len(ticks)


def _regular_grid(anchor: float, step: float, dur: float) -> np.ndarray:
    if step <= 0:
        return np.zeros(0, dtype=np.float32)
    start = anchor % step
    n = int(np.floor((dur - start) / step)) + 1
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    return (start + step * np.arange(n)).astype(np.float32)


def _filter_by_ratio(times: np.ndarray, num_env: np.ndarray, den_env: np.ndarray,
                     fr: float, thresh: float) -> np.ndarray:
    """Keep peak times where the numerator band dominates.

    Uses the best ratio over a ±2-frame window: sharp attacks are broadband
    for a single frame (kick click, snare crack), but the body that follows
    is band-dominant — hats never are.
    """
    if len(times) == 0:
        return times
    keep = []
    ratio = num_env / (den_env + 1e-9)
    for t in times:
        i = int(round(float(t) * fr))
        lo, hi = max(0, i - 1), min(len(ratio), i + 3)
        if lo < hi and float(np.max(ratio[lo:hi])) > thresh:
            keep.append(t)
    return np.asarray(keep, dtype=np.float32)


def detect_hits(y: np.ndarray, sr: int) -> dict:
    """Shared hit detection: filtered kick/snare times + band envelopes.

    Used identically by the grid estimator and the classifier features so
    training and serving see the same world.
    """
    y, sr = prep_audio(y, sr)
    sub_env, fr = _band_env(y, sr, 30.0, 150.0)
    snare_env, _ = _band_env(y, sr, 1500.0, 6000.0)
    hat_env, _ = _band_env(y, sr, 6000.0, None)
    body_env, _ = _band_env(y, sr, 200.0, 900.0)
    mid_env, _ = _band_env(y, sr, 150.0, 600.0)

    kicks = _pick_peaks(sub_env, fr, min_dist_sec=0.18, thresh_k=0.8)
    kicks = _filter_by_ratio(kicks, sub_env, mid_env, fr, 1.2)
    # snare candidates straight from the body band: hats are near-silent
    # there, so no ratio filter is needed (kick knock passes through, which
    # the grid probes tolerate by design).
    snares = _pick_peaks(body_env, fr, min_dist_sec=0.25, thresh_k=1.0)
    return {"y": y, "sr": sr, "kicks": kicks, "snares": snares,
            "sub_env": sub_env, "snare_env": snare_env, "hat_env": hat_env,
            "body_env": body_env, "fr": fr, "dur": len(y) / sr}


def estimate_kick_snare_grids(y: np.ndarray, sr: int) -> dict:
    """Estimate beat + feel from kick/snare pulse structure.

    Returns dict with: tempo (quarter-note BPM), beats (grid), anchor,
    fourfloor (0..1), halftime (0..1), straight (bool), method (str).
    """
    y, sr = prep_audio(y, sr)
    dur = len(y) / sr
    if dur < 1.0 or float(np.max(np.abs(y))) < 1e-6:
        step = 0.5
        return {"tempo": 120.0, "beats": _regular_grid(0.0, step, max(dur, 1.0)),
                "anchor": 0.0, "fourfloor": 0.0, "halftime": 0.0,
                "straight": True, "method": "silent_fallback"}

    hits = detect_hits(y, sr)
    kicks, snares = hits["kicks"], hits["snares"]
    sub_env, body_env, hat_env, fr = (hits["sub_env"], hits["body_env"],
                                      hits["hat_env"], hits["fr"])

    tracker_tempo = _quick_tracker_tempo(y, sr)
    result = _straight_or_half_grid(kicks, snares, sub_env, body_env, hat_env,
                                    fr, dur, tracker_tempo)
    if result is not None:
        return result
    return _tracker_grid(y, sr, sub_env, body_env, hat_env, fr, dur)


def _quick_tracker_tempo(y: np.ndarray, sr: int) -> float:
    """Cheap tempo hint from the beat tracker (a hypothesis, not the answer)."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        if np.isfinite(tempo) and tempo > 0:
            return tempo
    except Exception:
        pass
    return 120.0


def _steady_ioi(times: np.ndarray, lo: float = 0.2, hi: float = 2.5) -> tuple[float, float] | None:
    """Median IOI + CV for peak times. None when too few/erratic."""
    if len(times) < 3:
        return None
    iois = np.diff(times)
    iois = iois[(iois >= lo) & (iois <= hi)]
    if len(iois) < 2:
        return None
    med = float(np.median(iois))
    cv = float(np.std(iois) / (med + 1e-9))
    return med, cv


def _align_downbeat(kicks: np.ndarray, body_env: np.ndarray,
                    fr: float, step: float, dur: float) -> tuple[float, bool, float]:
    """Find the downbeat phase so beats[0] is beat 1 of a bar.

    Candidates come from kick phases (kicks land on beats in dance music);
    each is scored by snare-backbeat probes. Returns (anchor, is_halftime,
    score) where anchor in [0, step) starts the grid on a downbeat.
    """
    cands = {0.0}
    for p in (kicks[:12] if len(kicks) else []):
        cands.add(round(float(p) % step, 4))
    best = (0.0, False, -1.0)
    for d in sorted(cands):
        ticks = _regular_grid(d, step, dur)
        if len(ticks) < 4:
            continue
        straight = (_grid_align(body_env, fr, ticks[1::4], tol_sec=0.08)
                    + _grid_align(body_env, fr, ticks[3::4], tol_sec=0.08)) / 2
        half = _grid_align(body_env, fr, ticks[2::4], tol_sec=0.08)
        # halftime needs a clear margin: kick knock bleeds into every probe,
        # so ties/near-ties mean straight backbeats, not halftime.
        if half > straight + 0.1 and half > best[2]:
            best = (d, True, float(half))
        elif straight >= half and straight > best[2]:
            best = (d, False, float(straight))
    return best
def _candidate_steps(kicks: np.ndarray, snares: np.ndarray,
                     tracker_tempo: float) -> list[tuple[float, float]]:
    """Grid-step hypotheses (seconds) + source bonus.

    Direct pulse evidence (kick/snare clocks) outranks the tracker guess;
    the kick clock contributes M/4 too so sparse halftime kicks still
    hypothesize the fast grid (trap 150 from 1.6 s kick gaps).
    """
    cands: list[tuple[float, float]] = []
    kc = _steady_ioi(kicks, lo=0.2, hi=2.5)
    if kc is not None and kc[1] < 0.35:
        for div in (1, 2, 4):
            cands.append((kc[0] / div, 0.15))
    sc = _steady_ioi(snares, lo=0.25, hi=2.5)
    if sc is not None and sc[1] < 0.4:
        cands.append((sc[0] / 2, 0.15))
        cands.append((sc[0] / 4, 0.15))
    if 60.0 <= tracker_tempo <= 200.0:
        s = 60.0 / tracker_tempo
        cands += [(s, 0.0), (s / 2, 0.0), (s * 2, 0.0)]
    return [(c, b) for c, b in cands if 60.0 <= 60.0 / c <= 200.0]


def _score_step(step: float, anchor: float, dur: float, sub_env: np.ndarray,
                body_env: np.ndarray, hat_env: np.ndarray, fr: float) -> tuple[float, bool]:
    """Support for a quarter-note grid. Returns (score, is_halftime_geometry)."""
    ticks = _regular_grid(anchor, step, dur)
    if len(ticks) < 2:
        return -1.0, False
    sub = _grid_align(sub_env, fr, ticks)
    snr_a = _grid_align(body_env, fr, ticks[1::2], tol_sec=0.08)  # 2 & 4
    snr_b = _grid_align(body_env, fr, ticks[2::4], tol_sec=0.08)  # halftime 3
    # halftime geometry needs a clear margin: kick knock bleeds into both
    halftime_geom = snr_b > snr_a + 0.15
    snr = max(snr_a, snr_b)
    off = _regular_grid(anchor + step / 2, step, dur)
    hats = _grid_align(hat_env, fr, off, tol_sec=0.06)
    tempo = 60.0 / step
    prior = 0.15 if 90.0 <= tempo <= 180.0 else 0.0
    return sub + 0.7 * snr + 0.4 * hats + prior, halftime_geom


def _straight_or_half_grid(kicks: np.ndarray, snares: np.ndarray,
                           sub_env: np.ndarray, body_env: np.ndarray,
                           hat_env: np.ndarray, fr: float, dur: float,
                           tracker_tempo: float) -> dict | None:
    """Pick the quarter-note grid maximizing kick+snare+hat support.

    Octave ambiguity (75 vs 150 with identical kick/snare placement) is broken
    by hat-subdivision density: dense hats + halftime geometry = fast grid.
    """
    steps = _candidate_steps(kicks, snares, tracker_tempo)
    if not steps:
        return None
    anchor0 = float(kicks[0]) if len(kicks) else 0.0
    scored = []
    for step, bonus in steps:
        # dedupe near-identical hypotheses
        if any(abs(step - s) / s < 0.03 for _, s, *_ in scored):
            continue
        score, half_geom = _score_step(step, anchor0, dur, sub_env, body_env,
                                       hat_env, fr)
        scored.append((score + bonus, step, half_geom))
    if not scored:
        return None
    scored.sort(reverse=True)
    _, step, half_geom = scored[0]
    # de-quantize the step: frame-rate peak picking quantizes IOIs, which
    # drifts the grid on long tracks — refit from the full kick span.
    if len(kicks) >= 4:
        span = float(kicks[-1] - kicks[0])
        k = max(1, int(round(span / step)))
        refined = span / k
        if 60.0 <= 60.0 / refined <= 200.0:
            step = refined
    tempo = 60.0 / step
    # hat-density octave break: sparse-kick halftime with busy hats is the
    # fast grid (trap 150, not 75); sparse hats keep the slow grid.
    hat_ticks = _pick_peaks(hat_env, fr, min_dist_sec=0.05, thresh_k=0.7)
    hat_rate = len(hat_ticks) / max(dur, 0.5)
    if half_geom and hat_rate > 2.5 and tempo < 128.0 and tempo * 2 <= 200.0:
        step = step / 2
        tempo = tempo * 2
    anchor, half_geom, _ = _align_downbeat(kicks, body_env, fr, step, dur)
    ticks = _regular_grid(anchor, step, dur)
    four = _grid_align(sub_env, fr, ticks) if not half_geom else 0.0
    half = _grid_align(body_env, fr, ticks[2::4], tol_sec=0.08) if half_geom else 0.0
    return {"tempo": float(tempo), "beats": ticks, "anchor": anchor,
            "fourfloor": float(four), "halftime": float(half),
            "straight": bool(not half_geom),
            "method": "halftime" if half_geom else "kick_pulse"}


def _tracker_grid(y: np.ndarray, sr: int, sub_env: np.ndarray,
                  body_env: np.ndarray, hat_env: np.ndarray,
                  fr: float, dur: float) -> dict:
    """Fallback: beat tracker + octave pick by pulse alignment."""
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_ANALYSIS_HOP)
        tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                       hop_length=_ANALYSIS_HOP, win_length=384)
        agg = np.mean(tg, axis=1)
        tempi = librosa.feature.tempo_frequencies(len(agg), hop_length=_ANALYSIS_HOP, sr=sr)
        valid = (tempi >= 60) & (tempi <= 200)
        hint = 120.0
        if np.any(valid):
            agg_v = np.where(valid, agg, -1)
            hint = float(tempi[int(np.argmax(agg_v))])
        tempo0, beats0 = librosa.beat.beat_track(y=y, sr=sr, units="time",
                                                 start_bpm=hint, tightness=100)
        tempo0 = float(np.atleast_1d(tempo0)[0])
        beats0 = [float(b) for b in np.atleast_1d(beats0)]
    except Exception:
        step = 0.5
        return {"tempo": 120.0, "beats": _regular_grid(0.0, step, dur),
                "anchor": 0.0, "fourfloor": 0.0, "halftime": 0.0,
                "straight": True, "method": "tracker_fallback"}

    if not np.isfinite(tempo0) or tempo0 <= 0 or len(beats0) < 2:
        step = 0.5
        return {"tempo": 120.0, "beats": _regular_grid(0.0, step, dur),
                "anchor": 0.0, "fourfloor": 0.0, "halftime": 0.0,
                "straight": True, "method": "tracker_fallback"}

    ibis = np.diff(sorted(beats0))
    ibis = ibis[(ibis > 0.2) & (ibis < 1.5)]
    med_step = float(np.median(ibis)) if len(ibis) else 60.0 / tempo0
    base = 60.0 / med_step if med_step > 0 else tempo0

    # octave candidates: pick the grid with the best kick+snare support
    anchor = float(beats0[0])
    best = None
    for mult in (0.5, 1.0, 2.0):
        cand_tempo = base * mult
        if not 60.0 <= cand_tempo <= 200.0:
            continue
        step = 60.0 / cand_tempo
        ticks = _regular_grid(anchor, step, dur)
        sub = _grid_align(sub_env, fr, ticks)
        # snare support on the halftime backbeat (every 2nd tick, offset 1)
        half_ticks = ticks[1::2]
        snr = _grid_align(body_env, fr, half_ticks, tol_sec=0.08)
        score = sub + 0.6 * snr
        if best is None or score > best[0]:
            best = (score, cand_tempo, ticks, sub, snr)
    _, tempo, ticks, sub, snr = best
    four = sub if tempo >= 100 else 0.0
    half = snr if tempo >= 120 and sub < 0.5 else 0.0
    return {"tempo": float(tempo), "beats": [float(t) for t in ticks],
            "anchor": float(anchor), "fourfloor": float(four),
            "halftime": float(half), "straight": bool(sub >= 0.45),
            "method": "tracker_octave"}
