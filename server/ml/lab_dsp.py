"""Lightweight Lab DSP — no heavy models.

Applies EQ, filters, pitch, speed, gain via STFT + librosa.
Designed to be cheap: mono or stereo, 16-48kHz, <300ms per call.
Falls back to simple gains if scipy unavailable.
"""

import numpy as np
import librosa
import soundfile as sf
import tempfile

try:
    from scipy.signal import iirpeak, lfilter  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _load(path: str):
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    return y, sr


def _save(y: np.ndarray, sr: int) -> str:
    out = tempfile.mktemp(suffix=".wav")
    sf.write(out, y.T if y.shape[0] > 1 else y[0], sr)
    return out


def apply_lab_tweaks(
    audio_path: str,
    gains_db: dict | None = None,
    lowpass_hz: float | None = None,
    highpass_hz: float | None = None,
    pitch_semitones: float = 0.0,
    speed: float = 1.0,
    gain_db: float = 0.0,
    reverb_mix: float = 0.0,
) -> str:
    """Apply direct controls and return output wav path.

    gains_db: e.g. {"60": -2, "230": 1.5, "910": 0, "4000": 2, "14000": -1} (5-band)
    lowpass/highpass: cutoff in Hz or None
    pitch_semitones: -12..+12
    speed: 0.5..2.0 (time stretch)
    gain_db: -12..+12
    reverb_mix: 0..1
    """
    y, sr = _load(audio_path)
    n_ch = y.shape[0]

    # per-channel processing
    out_ch = []
    for ch in range(n_ch):
        y_ch = y[ch].copy()

        # speed — time stretch first (changes length)
        if speed != 1.0 and speed > 0:
            # librosa expects rate>0 where >1 = faster (shorter)
            y_ch = librosa.effects.time_stretch(y_ch, rate=float(speed))

        # pitch — after stretch to keep duration correct
        if abs(pitch_semitones) > 0.01:
            y_ch = librosa.effects.pitch_shift(y_ch, sr=sr, n_steps=float(pitch_semitones))

        # EQ / filters via STFT gain curve (cheap, phase-preserving via ISTFT)
        if gains_db or lowpass_hz or highpass_hz:
            D = librosa.stft(y_ch, n_fft=2048, hop_length=512)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            gain_curve = np.ones(len(freqs), dtype=np.float32)

            if gains_db:
                # interpolate 5-band gains onto freq axis (log scale)
                band_freqs = np.array([60, 230, 910, 4000, 14000], dtype=float)
                band_gains = np.array([gains_db.get(str(int(f)), gains_db.get(str(f), 0.0)) if isinstance(gains_db, dict) else 0.0 for f in band_freqs], dtype=float)
                # convert dB to linear
                band_linear = 10 ** (band_gains / 20.0)
                # log-interpolate
                log_f = np.log10(np.maximum(freqs, 1.0))
                log_b = np.log10(band_freqs)
                interp = np.interp(log_f, log_b, band_linear, left=band_linear[0], right=band_linear[-1])
                gain_curve *= interp

            if lowpass_hz:
                # smooth roll-off above cutoff
                roll = np.clip(1.0 - (freqs - lowpass_hz) / (lowpass_hz * 0.5 + 1e-9), 0, 1)
                roll[freqs <= lowpass_hz] = 1.0
                gain_curve *= roll
            if highpass_hz:
                roll = np.clip((freqs - highpass_hz * 0.5) / (highpass_hz * 0.5 + 1e-9), 0, 1)
                roll[freqs >= highpass_hz] = 1.0
                roll[freqs < highpass_hz * 0.5] = 0.0
                # smooth
                gain_curve *= roll

            D = D * gain_curve[:, np.newaxis]
            y_ch = librosa.istft(D, length=len(y_ch))

        # global gain
        if abs(gain_db) > 0.01:
            y_ch = y_ch * (10 ** (gain_db / 20.0))

        # light reverb (delay tap)
        if reverb_mix > 0.01:
            delay = int(0.045 * sr)
            rev = np.zeros_like(y_ch)
            if len(y_ch) > delay:
                rev[delay:] = y_ch[:-delay] * 0.35 * reverb_mix
                # second tap
                d2 = int(0.11 * sr)
                if len(y_ch) > d2:
                    rev[d2:] += y_ch[:-d2] * 0.18 * reverb_mix
            y_ch = y_ch * (1 - 0.35 * reverb_mix) + rev

        out_ch.append(y_ch)

    # pad to same length
    max_len = max(len(c) for c in out_ch)
    y_out = np.zeros((n_ch, max_len), dtype=np.float32)
    for i, c in enumerate(out_ch):
        y_out[i, : len(c)] = c

    # soft limiter
    peak = np.max(np.abs(y_out))
    if peak > 0.99:
        y_out = y_out / peak * 0.89

    return _save(y_out, sr)


def suggest_lab_preset(understand: dict) -> dict:
    """Return AI-suggested knob positions from existing understand result.

    No extra inference — just rules on mood/instruments/genre.
    This is what the Lab animates as 'AI is tweaking...' for the audience.
    """
    mood = (understand.get("mood") or {}).get("mood", "neutral") if isinstance(understand, dict) else "neutral"
    # genre may be nested
    genre = ""
    try:
        g = understand.get("genre")
        if isinstance(g, dict):
            genre = (g.get("genre") or "").lower()
    except Exception:
        pass

    preset = {
        "gains_db": {"60": 0, "230": 0, "910": 0, "4000": 0, "14000": 0},
        "lowpass_hz": None,
        "highpass_hz": None,
        "pitch_semitones": 0,
        "speed": 1.0,
        "gain_db": 0,
        "reverb_mix": 0.0,
        "reason": "Balanced — no strong correction needed.",
    }

    if mood == "dark":
        preset["gains_db"] = {"60": 2.0, "230": 1.0, "910": 0, "4000": -1.5, "14000": -2.0}
        preset["reverb_mix"] = 0.18
        preset["reason"] = "Dark mood — lifted lows, softened highs, light hall."
    elif mood == "energetic":
        preset["gains_db"] = {"60": 1.2, "230": 0, "910": 0.6, "4000": 1.4, "14000": 0.8}
        preset["gain_db"] = 1.0
        preset["reason"] = "Energetic — presence boost at 4kHz + gentle loudness."
    elif mood == "calm":
        preset["gains_db"] = {"60": -0.5, "230": 0, "910": 0, "4000": -0.8, "14000": 0.5}
        preset["highpass_hz"] = 80
        preset["reverb_mix"] = 0.22
        preset["reason"] = "Calm — airy highs, rumble cut at 80Hz."
    elif mood == "bright":
        preset["gains_db"] = {"60": -1.0, "230": -0.4, "910": 0.3, "4000": 1.8, "14000": 1.2}
        preset["reason"] = "Bright — de-mudded lows, shimmer on top."

    if "podcast" in genre or "speech" in genre:
        preset["highpass_hz"] = 90
        preset["gains_db"]["4000"] = float(preset["gains_db"]["4000"]) + 1.0  # clarity
        preset["reason"] += " · Speech preset: 90Hz HPF + presence."

    return preset
