"""Speech denoising and presence enhancement.

Measurement drove both designs here (see tests/test_voice.py):

* The previous path ran Demucs — a music-trained separator — over speech and
  made SNR *worse* by 5-7 dB, because it hallucinates stems on non-music input.
* A first attempt at spectral subtraction used a per-bin compressor that crushed
  tonal peaks: a single bin with high over-drive ratio took ~45 dB of gain
  reduction and nulled the signal. Compression is therefore done on **frame**
  level, which cannot nuke one bin.

Denoising uses a per-bin Wiener gain (smooth, no musical-noise pumping) with a
spectral floor, and bypasses entirely when the input is already clean. Presence
shaping uses adaptive EQ plus frame-level downward compression.
"""

from __future__ import annotations

import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

N_FFT = 1024
HOP = 256


def _stft(y: np.ndarray, sr: int):
    return librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT)


def _istft(S: np.ndarray, sr: int, length: int):
    return librosa.istft(S, hop_length=HOP, win_length=N_FFT, length=length)


def _channels(y: np.ndarray):
    return (True, [y]) if y.ndim == 1 else (False, list(y))


def _rebuild(chans: list[np.ndarray], single: bool) -> np.ndarray:
    return chans[0] if single else np.stack(chans, axis=0)


def broadband_snr_db(y: np.ndarray, sr: int) -> float:
    """Crude file-level SNR: loud frames vs quiet frames.

    If the quiet frames are already close to the loud ones the file is clean and
    denoising would only damage it.
    """
    ch = y[0] if y.ndim > 1 else y
    if ch.size < sr:
        return 99.0
    rms = librosa.feature.rms(y=ch, frame_length=N_FFT, hop_length=HOP)[0]
    if rms.size < 4:
        return 99.0
    loud = float(np.percentile(rms, 90)) + 1e-12
    quiet = float(np.percentile(rms, 10)) + 1e-12
    return float(20.0 * np.log10(loud / quiet))


def estimate_noise_profile(mag: np.ndarray, percentile: float = 12.0) -> np.ndarray:
    """Per-bin noise magnitude from the quietest frames (minimum statistics)."""
    if mag.shape[1] == 0:
        return np.zeros(mag.shape[0], dtype=np.float32)
    return np.percentile(mag, percentile, axis=1).astype(np.float32)


def denoise_speech(
    y: np.ndarray,
    sr: int,
    strength: float = 2.0,
    percentile: float = 12.0,
    clean_snr_db: float = 20.0,
) -> np.ndarray:
    """Remove stationary broadband/tonal noise from speech.

    ``strength`` 1..3 scales the Wiener over-subtraction factor. Returns the
    input untouched when ``broadband_snr_db`` exceeds ``clean_snr_db``.
    """
    if librosa is None or y.size == 0:
        return y
    if broadband_snr_db(y, sr) >= clean_snr_db:
        return y

    strength = float(np.clip(strength, 1.0, 3.0))
    alpha = 0.8 + (strength - 1.0) * 0.9      # Wiener over-subtraction
    floor = 0.10 - (strength - 1.0) * 0.03   # spectral floor

    single, chans = _channels(y)
    out: list[np.ndarray] = []

    for ch in chans:
        S = _stft(ch.astype(np.float32), sr)
        mag = np.abs(S)
        noise = np.maximum(estimate_noise_profile(mag, percentile)[:, None], 1e-10)

        power = mag ** 2
        # Wiener gain: 1 where signal >> noise, ~alpha*noise^2/|S|^2 elsewhere.
        gain = power / (power + alpha * noise ** 2 + 1e-20)
        gain = np.clip(gain, floor, 1.0)

        # Smooth across time so the gain does not flicker (musical noise).
        win = max(3, int(0.02 * sr / HOP))
        kernel = np.hanning(win)
        kernel /= kernel.sum()
        gain = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, gain)
        gain = np.clip(gain, floor, 1.0)

        rec = _istft(S * gain, sr, length=len(ch))
        peak = float(np.max(np.abs(rec)))
        if peak > 1.0:
            rec = rec / peak * 0.99
        out.append(rec.astype(np.float32))

    return _rebuild(out, single)


def frame_compress_db(
    frame_db: np.ndarray,
    ratio: float = 2.5,
    dynamic_db: float = 18.0,
    max_reduction_db: float = 12.0,
) -> np.ndarray:
    """Downward gain reduction in dB for each frame.

    Frame-level (not per-bin) so a single loud partial cannot be nulled, and the
    threshold is derived from the file's own level so it adapts to loudness.
    """
    ref = float(np.percentile(frame_db, 95))
    threshold = ref - dynamic_db
    over = frame_db - threshold
    reduction = np.where(over > 0.0, over * (1.0 - 1.0 / ratio), 0.0)
    return -np.clip(reduction, 0.0, max_reduction_db)


def speech_dominant_mask(
    mag: np.ndarray,
    percentile: float = 12.0,
    speech_percentile: float = 90.0,
) -> np.ndarray:
    """Per-bin indicator of where speech actually dominates the noise floor.

    A blanket 2-5 kHz presence lift makes broadband hiss *worse* (measured:
    -2.8 dB SNR), because the hiss lives in exactly that band. Boosting only the
    bins whose speech-to-noise margin is above the file's median keeps the
    intelligibility benefit and leaves noise-dominated bins alone.
    """
    noise = np.percentile(mag, percentile, axis=1) + 1e-12
    speech = np.percentile(mag, speech_percentile, axis=1) + 1e-12
    snr_db = 20.0 * np.log10(speech / noise)
    margin = float(np.median(snr_db))
    return np.clip((snr_db - margin) / 6.0, 0.0, 1.0).astype(np.float32)


def boost_speech_presence(
    y: np.ndarray,
    sr: int,
    presence_db: float = 4.0,
    compress: bool = True,
    makeup_db: float = 0.0,
    ratio: float = 2.5,
    dynamic_db: float = 10.0,
    max_reduction_db: float = 6.0,
    clean_snr_db: float = 20.0,
) -> np.ndarray:
    """Noise-aware presence EQ + gentle frame-level levelling.

    Skipped entirely on already-clean input: shaping a clean recording only
    risks raising its noise floor. ``makeup_db`` defaults to 0 because the
    compressor already restores loudness from the loudest frames.
    """
    if librosa is None or y.size == 0:
        return y
    if broadband_snr_db(y, sr) >= clean_snr_db:
        return y
    single, chans = _channels(y)
    out: list[np.ndarray] = []

    for ch in chans:
        S = _stft(ch.astype(np.float32), sr)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        gain = np.ones(len(freqs), dtype=np.float32)

        gain[freqs < 85] = 0.05                    # rumble

        # Presence lift, gated on speech dominance so hiss is not amplified.
        dominant = speech_dominant_mask(np.abs(S))
        band = (freqs >= 1500) & (freqs <= 6000)
        lift_db = np.zeros(len(freqs), dtype=np.float32)
        lift_db[band] = presence_db * dominant[band]
        gain *= 10.0 ** (lift_db / 20.0)

        # Residual high-frequency noise: cut rather than lift.
        residual_noise = np.percentile(np.abs(S), 12, axis=1)
        resid_db = 20.0 * np.log10(residual_noise + 1e-12)
        speech_ref = float(np.percentile(resid_db, 95))
        hiss_band = (freqs > 7000) & (resid_db < speech_ref - 10.0)
        gain[hiss_band] *= 0.7

        low = (freqs >= 120) & (freqs < 320)
        gain[low] *= 1.25                           # vocal warmth

        S2 = S * gain[:, None]

        if compress:
            frame_power = np.sqrt(np.mean(np.abs(S2) ** 2, axis=0)) + 1e-12
            frame_db = 20.0 * np.log10(frame_power)
            gain_db = frame_compress_db(
                frame_db, ratio=ratio, dynamic_db=dynamic_db, max_reduction_db=max_reduction_db
            )
            # Taper to unity at the edges to avoid boundary artefacts.
            taper = np.ones_like(gain_db)
            k = max(1, gain_db.size // 32)
            taper[:k] = np.linspace(0.0, 1.0, k)
            taper[-k:] = np.linspace(1.0, 0.0, k)
            gain_db = gain_db * taper

            # Loudness-preserving makeup: give back exactly what the loudest
            # frames lost. Without this the compressor just makes the voice
            # quieter (measured: -5 dB of speech level).
            loudest = int(round(0.95 * (gain_db.size - 1)))
            makeup_db += -float(gain_db[loudest])

            S2 = S2 * (10.0 ** (gain_db / 20.0))[None, :]

        S2 = S2 * 10.0 ** (makeup_db / 20.0)
        rec = _istft(S2, sr, length=len(ch))
        peak = float(np.max(np.abs(rec)))
        if peak > 0.99:
            rec = rec / peak * 0.99
        out.append(rec.astype(np.float32))

    return _rebuild(out, single)


def enhance_voice_speech(
    y: np.ndarray,
    sr: int,
    strength: float = 2.0,
    presence_db: float = 4.0,
    makeup_db: float = 2.0,
    compress: bool = False,
) -> tuple[np.ndarray, dict]:
    """Full speech treatment: denoise, then shape for audibility.

    Compression is **off** by default: measured against real noisy speech it
    traded speech level for noise floor and came out net-negative (-2.3 dB),
    because lifting quiet frames lifts the noise too. Denoise plus a flat
    makeup gain measured +1.0 to +6.2 dB instead.
    """
    before = broadband_snr_db(y, sr)
    d = denoise_speech(y, sr, strength=strength)
    out = boost_speech_presence(
        d, sr, presence_db=presence_db, makeup_db=makeup_db, compress=compress
    )
    after = broadband_snr_db(out, sr)
    return out, {
        "method": "spectral_denoise",
        "snr_before_db": round(before, 1),
        "snr_after_db": round(after, 1),
        "snr_gain_db": round(after - before, 1),
        "strength": strength,
    }