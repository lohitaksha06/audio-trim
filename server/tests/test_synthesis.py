"""Trained timbre model + real-sample drum engine.

These lock in what is actually real:
  - the timbre model exists, loads, and separates families measurably;
  - unknown instruments return None instead of being force-fitted (the
    'phonk'/'honk' substring trap in a new place);
  - drum hits come from the recorded bank, and stay tempo-locked.
"""

import numpy as np
import pytest

from server.ml.synthesis import drums, timbre


class TestTimbreAvailability:
    def test_model_is_built(self):
        assert timbre.available(), (
            "timbre model missing - run: python -m scripts.train_timbre_model"
        )

    def test_banks_are_real(self):
        assert timbre.available()
        assert drums.available(), (
            "drum bank missing - run: python -m scripts.build_drum_bank"
        )


class TestFamilyMapping:
    def test_exact_family(self):
        assert timbre._nearest_family("guitar") == "guitar/acoustic"

    def test_alias_maps_to_a_trained_family(self):
        assert timbre._nearest_family("piano") == "keyboard/acoustic"
        assert timbre._nearest_family("violin") == "string/acoustic"

    def test_unknown_instrument_is_not_force_fitted(self):
        """Regression: fuzzy matching mapped 'acid' onto vocal/acoustic."""
        for word in ("acid", "reese", "supersaw", "808", "wobble", "zzzz"):
            assert timbre._nearest_family(word) is None, word

    def test_unknown_raises_instead_of_guessing(self):
        with pytest.raises(KeyError):
            timbre.predict("zzzz", 60)


class TestPredictionsDiffer:
    def test_bass_is_darker_than_mallet(self):
        """A trained model should express real timbral differences."""
        bass = timbre.predict("bass", 45)
        mallet = timbre.predict("mallet", 84)
        assert bass["centroid"] < mallet["centroid"], (
            f"bass centroid {bass['centroid']:.3f} should be darker than "
            f"mallet {mallet['centroid']:.3f}"
        )

    def test_families_do_not_collapse_to_one_vector(self):
        vecs = [timbre.predict(f, 60)["bands"] for f in ("guitar", "flute", "bass", "organ")]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                assert not np.allclose(vecs[i], vecs[j]), f"{i} and {j} identical"

    def test_pitch_changes_the_prediction(self):
        low = timbre.predict("guitar", 45)["bands"]
        high = timbre.predict("guitar", 76)["bands"]
        assert not np.allclose(low, high)


class TestRenderNote:
    def test_renders_audio_of_requested_length(self):
        y, meta = timbre.render_note("guitar", 60, duration=0.5, sr=22050, seed=1)
        assert 0.45 * 22050 <= len(y) <= 0.6 * 22050
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) > 0.05, "rendered note is silent"

    def test_higher_pitch_has_higher_centroid(self):
        import librosa

        lo, _ = timbre.render_note("flute", 55, 0.6, sr=22050, seed=3)
        hi, _ = timbre.render_note("flute", 79, 0.6, sr=22050, seed=3)
        c_lo = float(np.mean(librosa.feature.spectral_centroid(y=lo, sr=22050)))
        c_hi = float(np.mean(librosa.feature.spectral_centroid(y=hi, sr=22050)))
        assert c_hi > c_lo, f"centroid did not rise with pitch ({c_lo:.0f} -> {c_hi:.0f})"

    def test_velocity_changes_level(self):
        soft, _ = timbre.render_note("keyboard", 60, 0.4, velocity=40, sr=22050, seed=2)
        loud, _ = timbre.render_note("keyboard", 60, 0.4, velocity=127, sr=22050, seed=2)
        assert np.sqrt(np.mean(loud ** 2)) > np.sqrt(np.mean(soft ** 2))

    def test_same_seed_is_reproducible(self):
        a, _ = timbre.render_note("guitar", 60, 0.3, sr=22050, seed=42)
        b, _ = timbre.render_note("guitar", 60, 0.3, sr=22050, seed=42)
        assert np.allclose(a, b)


class TestDrumBank:
    def test_every_pattern_instrument_has_samples(self):
        labels = drums.kit()
        assert labels, "drum bank is empty"
        for style in drums.styles():
            for label in drums.PATTERNS[style]:
                bank_label = {"open_hat": "cymbal"}.get(label, label)
                assert bank_label in labels, f"{style}/{label} has no recorded hits"

    def test_hit_is_real_audio(self):
        y = drums.hit("kick", sr=44100, seed=1)
        assert y is not None and y.size > 100
        assert np.max(np.abs(y)) > 0.3

    def test_no_hit_is_a_pure_tone(self):
        """A synthesised sine has no broadband content; a real kick does."""
        y = drums.hit("snare", sr=44100, seed=5)
        spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        freqs = np.fft.rfftfreq(len(y), 1 / 44100)
        band = spec[(freqs > 2000) & (freqs < 8000)].sum()
        assert band > 0.02 * spec.sum(), "no high-frequency content - not a real snare"

    def test_styles_cover_the_electronic_asking(self):
        s = drums.styles()
        for want in ("house", "techno", "trap", "dnb", "edm", "afrobeat", "garage"):
            assert want in s, want

    def test_style_aliases_resolve(self):
        assert drums.resolve_style("drum and bass") == "dnb"
        assert drums.resolve_style("r&b") == "afrobeat"
        assert drums.resolve_style("nothing here") == "house"


class TestDrumRender:
    def test_tempo_length_is_correct(self):
        y, meta = drums.render(bpm=120, bars=2, style="house", sr=44100, seed=1)
        expected = 2 * 4 * (60.0 / 120.0)
        assert abs(meta["seconds"] - expected) < 0.05

    def test_uses_many_real_hits(self):
        y, meta = drums.render(bpm=128, bars=2, style="house", sr=44100, seed=1)
        assert meta["hits_used"] >= 16, meta
        assert meta["source"].startswith("real recorded")

    def test_output_is_bounded_and_audible(self):
        y, _ = drums.render(bpm=140, bars=1, style="techno", sr=44100, seed=2)
        assert np.max(np.abs(y)) <= 1.0
        assert np.max(np.abs(y)) > 0.2

    def test_kick_energy_concentrates_on_the_grid(self):
        """Low-band (kick) onset energy must peak on 16th-note positions.

        Measured on the kick band rather than a full-band onset detector,
        because real cymbals ring for a second and legitimately produce
        detections away from the grid.
        """
        import librosa
        from scipy.signal import butter, sosfiltfilt

        bpm, sr, bars = 120, 44100, 2
        y, _ = drums.render(bpm=bpm, bars=bars, style="house", sr=sr, seed=1)
        step = 60.0 / bpm / 4
        total = bars * 4 * step

        low = sosfiltfilt(butter(4, 150, "lowpass", fs=sr, output="sos"), y)
        hop = 256
        env = librosa.onset.onset_strength(y=low, sr=sr, hop_length=hop)
        times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)

        on_grid, off_grid = [], []
        for t in times:
            if t >= total:
                continue
            nearest = round(t / step) * step
            # inside a fifth of a step of a grid position
            (on_grid if abs(t - nearest) <= step * 0.2 else off_grid).append(env[int(t / (hop / sr))])

        assert on_grid and off_grid, "not enough frames to compare"
        on_mean = float(np.mean(on_grid))
        off_mean = float(np.mean(off_grid))
        # Measured ~1.97x on a house pattern. Hats leak a little energy into the
        # sub-150 Hz band, so 1.5x is the bar; an actually-misaligned render
        # sits near 1.0x.
        assert on_mean > 1.5 * off_mean, (
            f"kick onset energy is not grid-aligned (on-grid {on_mean:.4f} vs "
            f"off-grid {off_mean:.4f}, ratio {on_mean / max(off_mean, 1e-9):.2f}x)"
        )

    def test_seed_changes_the_render(self):
        a, _ = drums.render(bpm=128, bars=1, style="house", sr=22050, seed=1)
        b, _ = drums.render(bpm=128, bars=1, style="house", sr=22050, seed=2)
        assert not np.allclose(a, b), "seed did not change the selected hits"