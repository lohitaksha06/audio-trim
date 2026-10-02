"""Disfluency detection: short hesitations are found, long pauses are preserved."""

import numpy as np
import pytest
import soundfile as sf

from server.ml.transcription.fillers import detect_disfluencies, remove_spans

SR = 22050


def _syllable_burst(duration: float, rng: np.random.Generator, amp: float = 0.35) -> np.ndarray:
    """Speech-like burst: syllable-rate amplitude modulation over band noise."""
    n = int(duration * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n) * 0.5
    # crude formant-ish shaping so it is not pure white noise
    shaped = np.convolve(noise, np.ones(24) / 24, mode="same")
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    return (amp * env * shaped).astype(np.float32)


@pytest.fixture
def speech_with_gaps(tmp_path):
    """Utterances separated by realistic pauses: two hesitations to cut, one
    short inter-word gap to preserve, and a long beat to preserve."""
    rng = np.random.default_rng(0)
    parts = [
        _syllable_burst(1.0, rng),
        np.zeros(int(0.15 * SR), dtype=np.float32),   # normal short gap: KEEP
        _syllable_burst(1.0, rng),
        np.zeros(int(0.45 * SR), dtype=np.float32),   # hesitation: remove
        _syllable_burst(1.0, rng),
        np.zeros(int(0.50 * SR), dtype=np.float32),   # hesitation: remove
        _syllable_burst(1.0, rng),
        np.zeros(int(1.50 * SR), dtype=np.float32),   # long pause: keep
        _syllable_burst(1.0, rng),
    ]
    y = np.concatenate(parts)[np.newaxis, :]
    path = tmp_path / "gaps.wav"
    sf.write(str(path), y[0], SR)
    return str(path), y


def _load(path):
    y, sr = sf.read(path, dtype="float32")
    return y[np.newaxis, :], sr


def test_detects_hesitations(speech_with_gaps):
    path, _ = speech_with_gaps
    y, sr = _load(path)
    spans = detect_disfluencies(y, sr)
    assert spans, "expected hesitation spans"

    covered = [(s.start, s.end) for s in spans]
    # Gaps at 2.15-2.60s and 3.60-4.10s.
    assert any(abs(s - 2.15) < 0.1 and abs(e - 2.60) < 0.1 for s, e in covered), covered
    assert any(abs(s - 3.60) < 0.1 and abs(e - 4.10) < 0.1 for s, e in covered), covered


def test_normal_short_gap_is_preserved(speech_with_gaps):
    """Regression: a 0.12s floor removed ordinary inter-word pauses and cut 45%
    of real speech. The default must leave a 0.15s gap alone."""
    path, _ = speech_with_gaps
    y, sr = _load(path)
    spans = detect_disfluencies(y, sr)
    touched = [(s.start, s.end) for s in spans]
    assert not any(abs(s - 1.0) < 0.12 and abs(e - 1.15) < 0.12 for s, e in touched), (
        f"normal 0.15s gap was cut; spans={[(round(a,2), round(b,2)) for a, b in touched]}"
    )


def test_preserves_long_pause(speech_with_gaps):
    path, _ = speech_with_gaps
    y, sr = _load(path)
    spans = detect_disfluencies(y, sr)
    # The 1.50s beat starts at 5.60s and must survive.
    assert not any(abs(s - 5.60) < 0.12 for s, _ in [(x.start, x.end) for x in spans]), (
        f"long pause was cut; spans={[(round(x.start,2), round(x.end,2)) for x in spans]}"
    )


def test_removal_shortens_audio_by_expected_amount(speech_with_gaps):
    path, _ = speech_with_gaps
    y, sr = _load(path)
    before = y.shape[-1] / sr
    spans = detect_disfluencies(y, sr)
    out = remove_spans(y, sr, spans)
    after = out.shape[-1] / sr

    removed = before - after
    expected = sum(s.duration for s in spans)
    assert removed > 0
    assert abs(removed - expected) < 0.4, f"removed {removed:.2f}s vs expected {expected:.2f}s"
    # Only the two hesitations (0.45 + 0.50) should go.
    assert 0.6 < removed < 1.3, f"unexpected amount removed: {removed:.2f}s"


def test_output_shape_and_no_clipping(speech_with_gaps):
    path, _ = speech_with_gaps
    y, sr = _load(path)
    out = remove_spans(y, sr, detect_disfluencies(y, sr))
    assert out.ndim == 2 and out.shape[0] == y.shape[0]
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_silence_only_input_is_left_alone():
    y = np.zeros((1, SR), dtype=np.float32)
    assert detect_disfluencies(y, SR) == []


def test_max_pause_setting_widens_removal():
    rng = np.random.default_rng(3)
    y = np.concatenate(
        [
            _syllable_burst(1.0, rng),
            np.zeros(int(1.0 * SR), dtype=np.float32),
            _syllable_burst(1.0, rng),
        ]
    )[np.newaxis, :]

    default_spans = detect_disfluencies(y, SR, max_pause=0.9)
    wide_spans = detect_disfluencies(y, SR, max_pause=1.5)

    assert not default_spans, "1.0s gap should be out of range by default"
    assert wide_spans, "1.0s gap should be removable when max_pause=1.5"


def test_min_pause_setting_narrows_removal():
    """`min_pause` raises the floor for both clean pauses and filled blabs.

    At the default a 0.30s low-energy gap counts as a filled hesitation; raising
    the floor above it must leave the gap alone.
    """
    rng = np.random.default_rng(5)
    y = np.concatenate(
        [
            _syllable_burst(1.0, rng),
            np.zeros(int(0.30 * SR), dtype=np.float32),
            _syllable_burst(1.0, rng),
        ]
    )[np.newaxis, :]

    default_spans = detect_disfluencies(y, SR)
    assert default_spans, "0.30s filled hesitation should be caught by default"
    assert not detect_disfluencies(y, SR, min_pause=0.9), (
        "raising min_pause above the gap length must preserve it"
    )