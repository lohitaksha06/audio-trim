"""Voice enhancement must measurably help — these numbers come from real runs.

The previous implementation routed speech through Demucs (a music-trained
separator) and measured -5 to -7 dB SNR. These tests assert the measured
direction and magnitude so the regression cannot come back.

Metrics follow broadcast practice rather than sample-wise SNR, because EQ and
compression change the waveform and sample-wise comparison punishes them:
  - noise floor  (15th-percentile frames) should drop
  - speech level (60th+ percentile frames) should hold or rise
  - SNR gain     = d(speech) - d(noise)
"""

import numpy as np
import pytest
import soundfile as sf

from server.ml.speech.voice import (
    boost_speech_presence,
    broadband_snr_db,
    denoise_speech,
    enhance_voice_speech,
    frame_compress_db,
)

SR = 22050


def _burst(duration, rng, amp=0.3):
    n = int(duration * SR)
    t = np.arange(n) / SR
    noise = np.convolve(rng.standard_normal(n) * 0.5, np.ones(24) / 24, mode="same")
    return (amp * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * noise).astype(np.float32)


@pytest.fixture
def speech():
    """Three utterances with natural short gaps."""
    rng = np.random.default_rng(1)
    return np.concatenate([_burst(1.2, rng), np.zeros(int(0.15 * SR), np.float32),
                           _burst(1.2, rng), np.zeros(int(0.2 * SR), np.float32),
                           _burst(1.2, rng)])


@pytest.fixture
def noisy(speech):
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(speech)).astype(np.float32)
    sp = float(np.mean(speech * speech))
    npow = float(np.mean(noise * noise))
    scale = float(np.sqrt(sp / (npow * 1.0)))  # 0 dB SNR
    return np.clip(speech + noise * scale, -1, 1).astype(np.float32)


def levels(y):
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    loud = rms[rms > np.percentile(rms, 60)]
    quiet = rms[rms <= np.percentile(rms, 15)]
    return (20 * np.log10(np.mean(loud) + 1e-12),
            20 * np.log10(np.mean(quiet) + 1e-12) if quiet.size else float("-inf"))


def gains(sig, out):
    s_in, n_in = levels(sig)
    s_out, n_out = levels(out)
    return (s_out - s_in) - (n_out - n_in), (s_out - s_in)


class TestDenoise:
    def test_reduces_noise_floor(self, noisy):
        out = denoise_speech(noisy, SR, strength=2.0)
        _, n_in = levels(noisy)
        _, n_out = levels(out)
        assert n_out < n_in - 1.0, f"noise floor did not drop ({n_in:.1f} -> {n_out:.1f})"

    def test_does_not_mute_speech(self, noisy):
        out = denoise_speech(noisy, SR, strength=2.0)
        g, d_speech = gains(noisy, out)
        assert d_speech > -3.0, f"speech attenuated {d_speech:.1f} dB"

    def test_improves_speech_to_noise(self, noisy):
        out = denoise_speech(noisy, SR, strength=2.0)
        g, _ = gains(noisy, out)
        assert g > 0.5, f"expected SNR gain, got {g:+.2f} dB"

    def test_clean_input_untouched(self, speech):
        out = denoise_speech(speech, SR, strength=3.0)
        assert broadband_snr_db(speech, SR) >= 20.0
        assert np.allclose(out, speech), "clean audio must be returned untouched"


class TestEnhance:
    @pytest.mark.parametrize("strength", [1.0, 2.0, 3.0])
    def test_snr_gain_positive_at_every_strength(self, noisy, strength):
        out, meta = enhance_voice_speech(noisy, SR, strength=strength)
        g, _ = gains(noisy, out)
        assert g > 0.5, f"strength={strength}: SNR gain {g:+.2f} dB"

    def test_makes_voice_audible_not_quieter(self, noisy):
        """Regression: compression used to trade speech level away (-5 dB)."""
        out, _ = enhance_voice_speech(noisy, SR)
        _, d_speech = gains(noisy, out)
        assert d_speech > -1.0, f"speech got quieter by {d_speech:.1f} dB"

    def test_reports_measured_snr(self, noisy):
        _, meta = enhance_voice_speech(noisy, SR)
        assert meta["method"] == "spectral_denoise"
        assert meta["snr_gain_db"] > 0.5, meta

    def test_clean_audio_untouched_end_to_end(self, speech):
        out, _ = enhance_voice_speech(speech, SR)
        assert np.allclose(out, speech), "clean audio must pass through unchanged"

    def test_output_is_finite_and_bounded(self, noisy):
        out, _ = enhance_voice_speech(noisy, SR)
        assert np.isfinite(out).all()
        assert np.max(np.abs(out)) <= 1.0 + 1e-6


class TestShape:
    def test_frame_compress_reduces_loud_frames_only(self):
        frame_db = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        red = frame_compress_db(frame_db, ratio=3.0, dynamic_db=20.0, max_reduction_db=12.0)
        assert red[-1] < 0, "loudest frame must be reduced"
        assert red[0] == 0, "quietest frame must be untouched"
        assert red[-1] >= -12.0, "reduction must stay capped"

    def test_compressor_cannot_null_a_single_bin(self):
        """Regression: a per-bin compressor cut a pure tone by ~45 dB.

        Frame-level compression must keep the overall level close to the input.
        """
        t = np.arange(int(2.0 * SR)) / SR
        tone = 0.3 * np.sin(2 * np.pi * 220 * t)
        out = boost_speech_presence(tone, SR, compress=True, clean_snr_db=0.0)
        def db(x):
            return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
        assert abs(db(out) - db(tone)) < 8.0, (
            f"compressor changed level by {db(out) - db(tone):.1f} dB")
        assert np.isfinite(out).all() and np.max(np.abs(out)) > 0.05

    def test_stereo_shape_is_preserved(self, noisy):
        stereo = np.stack([noisy, noisy * 0.9])
        out = denoise_speech(stereo, SR)
        assert out.ndim == 2 and out.shape[0] == 2


class TestRoundTrip:
    def test_writes_readable_wav(self, noisy, tmp_path):
        out, _ = enhance_voice_speech(noisy, SR)
        p = tmp_path / "out.wav"
        sf.write(str(p), out, SR)
        back, sr = sf.read(str(p), dtype="float32")
        assert sr == SR and back.shape == out.shape
        assert np.isfinite(back).all()