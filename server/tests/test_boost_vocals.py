"""Regression coverage for "boost the vocals" on noisy material.

A presence shelf across 1-6 kHz is SNR-neutral *inside that band* by
construction: it scales speech and noise alike. The audible problem is that on
a noisy recording the band is mostly hiss, so the shelf makes the hiss louder.
The boost path therefore denoises first, and these tests pin both halves of
that decision:

  - noisy input  -> denoise runs, and beats the bare shelf
  - clean input  -> denoise self-gates and the plain shelf is used

The fixture matters. An earlier version of this measurement used tones below
520 Hz, where the presence band contains no speech at all -- any shelf then
looks harmful because it can only lift noise. This fixture carries glottal
harmonics and fricative energy up to 6 kHz, which is what a presence boost is
actually for.
"""

import numpy as np
import soundfile as sf

from server.ml.audio_operations import _boost_target, execute_plan
from server.ml.prompt_engine import regex_plan_from_prompt

SR = 22050


def _speech_like(duration: float = 3.0, seed: int = 11) -> np.ndarray:
    """Harmonics with a -12 dB/oct tilt to 6 kHz, three formants, syllable rate."""
    from scipy.signal import butter, iirpeak, lfilter, sosfilt

    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * duration)) / SR

    f0 = 120.0 * (1.0 + 0.02 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    y = np.zeros_like(t)
    for k in range(1, int(6000 / 120)):
        y += (1.0 / (k**1.2)) * np.sin(k * phase)

    for centre, q, gain in ((700, 6, 3.0), (1200, 8, 2.2), (2600, 10, 1.6)):
        b, a = iirpeak(centre / (SR / 2), q)
        r = lfilter(b, a, y)
        peak = np.max(np.abs(r)) + 1e-9
        y = y + gain * r / peak * np.max(np.abs(y))

    env = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 4 * t - np.pi / 2), 0, 1) ** 1.5
    env[(t % 1.6) > 1.35] *= 0.05
    y = y * env

    fricative = sosfilt(
        butter(4, [3000 / (SR / 2), 7000 / (SR / 2)], btype="band", output="sos"),
        rng.standard_normal(len(t)),
    )
    burst = (np.sin(2 * np.pi * 2.5 * t) > 0.85).astype(float) * env
    y = y + 0.25 * fricative / (np.max(np.abs(fricative)) + 1e-9) * burst

    return (y / np.max(np.abs(y)) * 0.7).astype(np.float32)


def _add_noise(clean: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2) / 10 ** (snr_db / 10))
    return np.clip(clean + noise, -1, 1).astype(np.float32)


def _tsnr(out: np.ndarray, clean: np.ndarray) -> float:
    n = min(len(out), len(clean))
    err = out[:n] - clean[:n]
    return 10 * np.log10((np.mean(out[:n] ** 2) + 1e-12) / (np.mean(err**2) + 1e-12))


def test_noisy_vocal_boost_denoises_first(tmp_path):
    clean = _speech_like()
    mix = _add_noise(clean, snr_db=6)
    path = tmp_path / "noisy.wav"
    sf.write(path, mix, SR)

    result = execute_plan(str(path), regex_plan_from_prompt("boost the vocals"))

    assert result["intent"] == "boost"
    assert result["metadata"].get("denoised_before_boost") is True

    out, _ = sf.read(result["output_path"], dtype="float32")
    bare = _boost_target(mix[np.newaxis, :].copy(), SR, "vocals")[0]

    # The claim being pinned: denoise-then-shelf beats the bare shelf.
    assert _tsnr(out, clean) > _tsnr(bare, clean)


def test_clean_audio_skips_the_denoiser(tone_path):
    # A pure two-tone signal clears the denoiser's own clean-audio gate, so the
    # boost must not pay for a denoise pass it does not need.
    result = execute_plan(tone_path, regex_plan_from_prompt("boost the vocals"))

    assert result["intent"] == "boost"
    assert "denoised_before_boost" not in result["metadata"]
