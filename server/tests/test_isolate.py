"""Instrument and vocal isolation.

Locks in the two things that were broken:
  - piano/guitar were impossible: `htdemucs` only emits drums/bass/other/vocals,
    so they came back as everything that is not drums/bass/voice;
  - the vocal clean-up never ran. It called `denoise_speech(aggressive=...)`,
    which is not a parameter, and a bare `except` swallowed the TypeError --
    so the "cleanup" silently returned the untouched stem.
"""

import numpy as np
import pytest
import soundfile as sf

from server.ml.source_separation.isolate import (
    _MODEL_FOR_TARGET,
    available_targets,
    clean_vocal,
    isolate,
    resolve_target,
)

SR = 22050


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A short real-ish music clip: bass, chords, hats, a vocal-like lead."""
    d = tmp_path_factory.mktemp("iso")
    sr = SR
    dur = 6.0
    n = int(dur * sr)
    rng = np.random.default_rng(0)
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float32)

    # Sustained pad (piano-ish) and bass.
    for f, a in ((220.0, 0.28), (261.6, 0.22), (329.6, 0.18), (98.0, 0.3)):
        y += a * np.sin(2 * np.pi * f * t)

    # Lead with vibrato and harmonics, so there is something "vocal".
    f0 = 392.0 * (1 + 0.01 * np.sin(2 * np.pi * 5.0 * t))
    ph = 2 * np.pi * np.cumsum(f0) / sr
    lead = np.zeros(n, dtype=np.float32)
    for k, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)):
        lead += a * np.sin(k * ph)
    lead *= (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2)
    y += 0.35 * lead

    # Percussion so the drums stem is not empty.
    for b in range(12):
        at = int(b * 0.5 * sr)
        ln = int(0.08 * sr)
        seg = rng.standard_normal(ln).astype(np.float32) * np.exp(
            -np.arange(ln) / (0.008 * sr)
        )
        y[at:at + ln] += 0.3 * seg

    y = (y / (np.max(np.abs(y)) + 1e-9) * 0.85).astype(np.float32)
    p = d / "clip.wav"
    sf.write(str(p), y, sr)
    return str(p)


class TestTargetResolution:
    @pytest.mark.parametrize("word,expected", [
        ("vocals", "vocals"), ("vocal", "vocals"), ("voice", "vocals"),
        ("singing", "vocals"), ("drums", "drums"), ("drum", "drums"),
        ("bass", "bass"), ("bassline", "bass"), ("808", "bass"),
        ("piano", "piano"), ("keys", "piano"), ("keyboard", "piano"),
        ("guitar", "guitar"), ("acoustic guitar", "guitar"),
        ("other", "other"), ("accompaniment", "other"),
    ])
    def test_known_words(self, word, expected):
        assert resolve_target(word) == expected

    def test_unrepresentable_is_rejected_not_guessed(self):
        """Same lesson as the `phonk`/`honk` bug: do not force-fit."""
        for w in ("zither", "theremin", "kazoo", "wobble", ""):
            assert resolve_target(w) is None, w

    def test_piano_and_guitar_use_the_six_stem_model(self):
        """htdemucs/htdemucs_ft cannot produce these at all."""
        assert _MODEL_FOR_TARGET["piano"] == "htdemucs_6s"
        assert _MODEL_FOR_TARGET["guitar"] == "htdemucs_6s"

    def test_main_stems_use_the_ensemble_model(self):
        for t in ("vocals", "drums", "bass"):
            assert _MODEL_FOR_TARGET[t] == "htdemucs_ft"


class TestIsolate:
    def test_vocals_isolated_and_non_empty(self, clip):
        meta = isolate(clip, "vocals")
        assert meta.get("path"), meta.get("note")
        y, sr = sf.read(meta["path"], dtype="float32")
        assert sr == meta["isolate_sr"]
        assert y.size > 0
        assert meta["stem_rms"] > 0.0
        assert meta["isolate_model"] == "htdemucs_ft"

    def test_vocal_cleanup_actually_runs(self, clip):
        """Regression: the cleanup raised TypeError and was swallowed."""
        meta = isolate(clip, "vocals")
        assert "vocal_cleanup_error" not in meta, meta.get("vocal_cleanup_error")
        assert meta.get("vocal_denoised") is True
        assert meta.get("vocal_presence_lifted") is True

    def test_piano_isolated_from_the_six_stem_model(self, clip):
        """The capability that did not exist before."""
        meta = isolate(clip, "piano")
        assert meta.get("path"), meta.get("note")
        assert meta["isolated_stem"] == "piano"
        assert meta["isolate_model"] == "htdemucs_6s"
        y, _ = sf.read(meta["path"], dtype="float32")
        assert y.size > 0

    def test_guitar_isolated_from_the_six_stem_model(self, clip):
        meta = isolate(clip, "guitar")
        assert meta.get("path"), meta.get("note")
        assert meta["isolated_stem"] == "guitar"
        assert meta["isolate_model"] == "htdemucs_6s"

    def test_output_is_bounded(self, clip):
        for t in ("vocals", "drums", "bass"):
            meta = isolate(clip, t)
            if not meta.get("path"):
                continue
            y, _ = sf.read(meta["path"], dtype="float32")
            assert np.all(np.isfinite(y)), t
            assert np.max(np.abs(y)) <= 1.0, t

    def test_unknown_target_explains_itself(self, clip):
        meta = isolate(clip, "theremin")
        assert meta.get("path") is None
        assert "not a stem we can isolate" in meta["note"]
        assert meta["available"]

    def test_every_advertised_target_is_resolvable(self):
        for label in available_targets():
            assert resolve_target(label.split()[0]) is not None, label


class TestCleanVocal:
    def test_changes_the_signal(self):
        rng = np.random.default_rng(1)
        sr = SR
        t = np.arange(sr * 2) / sr
        clean = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        noisy = (clean + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
        out, meta = clean_vocal(noisy[np.newaxis, :], sr)
        assert "vocal_cleanup_error" not in meta, meta.get("vocal_cleanup_error")
        assert out.shape[-1] == noisy.shape[-1]
        assert not np.allclose(out, noisy), "cleanup was a no-op"

    def test_reports_what_it_did(self):
        sr = SR
        y = np.zeros((1, sr), dtype=np.float32)
        _, meta = clean_vocal(y, sr)
        assert meta["vocal_source"]
        assert "vocal_rms_before" in meta and "vocal_rms_after" in meta

    def test_silence_does_not_crash(self):
        out, meta = clean_vocal(np.zeros((1, SR), dtype=np.float32), SR)
        assert np.all(np.isfinite(out))
        assert out.shape[-1] == SR
        _ = meta