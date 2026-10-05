"""Two-file jamming: tempo, key and phase must actually match.

Every assertion here is against ground truth we synthesise, not against whatever
the detectors happen to say. Testing alignment by trusting the tempo detector
would prove nothing.

The bugs these lock down, all found by measuring output:
  - the stem was shifted by `first_beat_of_track - first_beat_of_stem`, but the
    first detected beat is usually NOT the downbeat, so parts sat out of phase;
  - nothing matched key at all, so two files a few semitones apart beat against
    each other;
  - level matching used peak, so a single transient set the gain.
"""

import numpy as np
import pytest
import soundfile as sf

from server.ml.audio_operations import _best_beat_offset, _estimate_key, mix_imported_stem

SR = 44100
A_PC, B_PC = 9, 2          # A minor and D minor
A_BPM, B_BPM = 120.0, 140.0
LAG_S = 0.37               # deliberately not a beat multiple


def _hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _tone(midi: float, dur: float, amp: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    s = np.zeros_like(t)
    for h, a in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.18), (5, 0.1), (6, 0.06)):
        s += a * np.sin(2 * np.pi * _hz(midi) * h * t)
    return (amp * s / (np.max(np.abs(s)) + 1e-9)).astype(np.float32)


def _kick(dur: float = 0.35) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    f = 120 * np.exp(-t * 28) + 46
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8)
    return (0.9 * s / (np.max(np.abs(s)) + 1e-9)).astype(np.float32)


def _make(bpm: float, root_pc: int, bars: int = 4, drums: bool = True) -> np.ndarray:
    beat = 60.0 / bpm
    n = int(bars * 4 * beat * SR)
    track = np.zeros(n, dtype=np.float32)
    # MIDI 33 is A1 (pc 9); map pitch-class the way the app does.
    root_midi = 33 + ((root_pc - 9) % 12)

    for b in range(bars * 4):
        at = int(b * beat * SR)
        seg = _tone(root_midi, beat * 0.85, 0.5)
        track[at:at + len(seg)] += seg
        if drums:
            k = _kick()
            track[at:at + len(k)] += k
        else:
            for iv in (0, 3, 7):
                seg = _tone(root_midi + 12 + iv, beat * 0.9, 0.34)
                track[at:at + len(seg)] += seg
    return (track / (float(np.max(np.abs(track))) + 1e-9) * 0.85).astype(np.float32)


@pytest.fixture(scope="module")
def jam_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("jam")
    paths = {}
    for name, bpm, pc, drums in (
        ("a_mix", A_BPM, A_PC, True),
        ("b_mix", B_BPM, B_PC, True),
        ("a_harm", A_BPM, A_PC, False),
        ("b_harm", B_BPM, B_PC, False),
    ):
        p = d / f"{name}.wav"
        sf.write(str(p), _make(bpm, pc, drums=drums), SR)
        paths[name] = str(p)

    lag = int(LAG_S * SR)
    lagged = np.concatenate([np.zeros(lag, np.float32), _make(A_BPM, A_PC)])[
        : len(_make(A_BPM, A_PC))]
    p = d / "lagged.wav"
    sf.write(str(p), lagged, SR)
    paths["lagged"] = str(p)
    return paths


class TestKeyEstimation:
    def test_finds_the_right_tonic(self, jam_files):
        """Regression risk: a wrong pitch-class mapping in the fixture made a
        working detector look broken. Assert against known pitch classes."""
        for key, truth in (("a_harm", A_PC), ("b_harm", B_PC),
                           ("a_mix", A_PC), ("b_mix", B_PC)):
            import librosa

            y, sr = librosa.load(jam_files[key], sr=None, mono=True)
            pc, conf = _estimate_key(y, sr)
            assert pc == truth, f"{key}: got pc {pc}, expected {truth}"
            assert 0.0 <= conf <= 1.0

    def test_silence_does_not_crash(self):
        pc, conf = _estimate_key(np.zeros(SR, dtype=np.float32), SR)
        assert conf <= 1.0


class TestTempoMatching:
    def test_stem_is_stretched_to_the_track_tempo(self, jam_files):
        _, meta, _ = mix_imported_stem(jam_files["a_mix"], jam_files["b_mix"], {})
        expected = B_BPM / A_BPM
        assert abs(meta["stretch_factor"] - expected) < 0.03, (
            f"stretched {meta['stretch_factor']}, expected ~{expected:.3f}"
        )

    def test_residual_tempo_is_zero_after_matching(self, jam_files):
        _, meta, _ = mix_imported_stem(jam_files["a_mix"], jam_files["b_mix"], {})
        assert abs(meta["tempo_residual_bpm"]) < 1.0

    def test_matching_tempos_leaves_audio_alone(self, jam_files):
        _, meta, _ = mix_imported_stem(jam_files["a_mix"], jam_files["lagged"], {})
        assert meta["stretch_factor"] == 1.0
        assert meta["tempo_residual_bpm"] == 0.0

    def test_absurd_tempo_gap_is_reported_not_hidden(self, tmp_path):
        """A 40 BPM vs 160 BPM gap must be declared, never silently faked."""
        slow = tmp_path / "slow.wav"
        fast = tmp_path / "fast.wav"
        sf.write(str(slow), _make(40.0, A_PC, bars=2), SR)
        sf.write(str(fast), _make(160.0, B_PC, bars=2), SR)
        _, meta, _ = mix_imported_stem(str(slow), str(fast), {})
        assert abs(meta["tempo_residual_bpm"]) > 1.0, (
            "a 4x tempo gap was reported as matched"
        )
        assert any("too large" in n for n in meta["jam_notes"]), meta["jam_notes"]


class TestKeyMatching:
    def test_stem_is_transposed_to_the_track_key(self, jam_files):
        _, meta, _ = mix_imported_stem(jam_files["a_harm"], jam_files["b_harm"], {})
        assert meta["song_key_pc"] == A_PC
        assert meta["stem_key_pc"] == B_PC
        # D -> A is +7 semitones, or equivalently -5. Never 0.
        assert meta["transposed_semitones"] in (-5, 7), meta["transposed_semitones"]
        assert any("transposed" in n for n in meta["jam_notes"])

    def test_same_key_is_not_transposed(self, jam_files):
        _, meta, _ = mix_imported_stem(jam_files["a_mix"], jam_files["lagged"], {})
        assert meta["transposed_semitones"] == 0
        assert any("already matched" in n for n in meta["jam_notes"])

    def test_key_matching_can_be_switched_off(self, jam_files):
        _, meta, _ = mix_imported_stem(
            jam_files["a_harm"], jam_files["b_harm"], {"match_key": False}
        )
        assert meta["transposed_semitones"] == 0

    def test_transposition_actually_changes_pitch(self, jam_files):
        """The reported semitones must match what came out of the speaker."""
        import librosa

        _, _, aligned = mix_imported_stem(jam_files["a_harm"], jam_files["b_harm"], {})
        pc, _ = _estimate_key(aligned[np.newaxis, :], SR)
        assert pc == A_PC, (
            f"aligned stem still reads as pc {pc}, expected {A_PC} "
            "(the track's key)"
        )
        _ = librosa


class TestPhaseAlignment:
    def test_onsets_line_up_after_jamming(self, jam_files):
        """The real test: do the two files' transients actually coincide?"""
        import librosa

        y, sr = librosa.load(jam_files["a_mix"], sr=None, mono=True)
        _, meta, aligned = mix_imported_stem(
            jam_files["a_mix"], jam_files["lagged"], {}
        )
        oa = librosa.onset.onset_detect(y=y, sr=sr, backtrack=False, units="time")
        ob = librosa.onset.onset_detect(y=aligned, sr=sr, backtrack=False, units="time")
        assert len(oa) > 4 and len(ob) > 4
        d = ob[:, None] - oa[None, :]
        nearest = np.min(np.abs(d), axis=1)
        median = float(np.median(nearest))
        assert median < 0.02, (
            f"median onset offset {median * 1000:.0f} ms -- the parts are "
            f"out of phase (beat period {60 / A_BPM * 1000:.0f} ms)"
        )

    def test_offset_recovers_a_known_lag(self, jam_files):
        _, meta, _ = mix_imported_stem(jam_files["a_mix"], jam_files["lagged"], {})
        # A stem lagged by +0.37s must be moved back by ~0.37s.
        assert abs(meta["beat_offset_sec"] + LAG_S) < 0.05, (
            f"offset {meta['beat_offset_sec']:+.3f}s, expected ~{-LAG_S:+.3f}s"
        )

    def test_offset_never_exceeds_a_beat(self, jam_files):
        _, meta, _ = mix_imported_stem(jam_files["a_mix"], jam_files["b_mix"], {})
        assert abs(meta["beat_offset_sec"]) <= 60.0 / A_BPM + 1e-6

    def test_best_beat_offset_is_bounded(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(SR).astype(np.float32)
        b = np.zeros(SR, dtype=np.float32)
        off = _best_beat_offset(a[np.newaxis, :], SR, b, 0.5)
        assert -0.5 - 1e-6 <= off <= 0.5 + 1e-6


class TestOutputSafety:
    @pytest.mark.parametrize("pair", [("a_mix", "b_mix"), ("a_harm", "lagged"),
                                      ("a_mix", "lagged")])
    def test_output_is_finite_and_bounded(self, jam_files, pair):
        mixed, meta, aligned = mix_imported_stem(jam_files[pair[0]], jam_files[pair[1]], {})
        assert np.all(np.isfinite(mixed)), "output contains NaN/Inf"
        assert np.max(np.abs(mixed)) <= 1.0
        assert mixed.shape[0] >= 1
        assert np.any(aligned), "aligned stem is silent"
        assert meta["jam_notes"], "no explanation of what was done"

    def test_level_matching_keeps_the_stem_audible(self, jam_files):
        """Peak-based matching buried stems; RMS keeps them in the mix."""
        import librosa

        _, meta, aligned = mix_imported_stem(
            jam_files["a_mix"], jam_files["b_mix"], {"stem_level": 1.0}
        )
        track, sr = librosa.load(jam_files["a_mix"], sr=None, mono=True)
        r_stem = float(np.sqrt(np.mean(aligned ** 2)))
        r_track = float(np.sqrt(np.mean(track ** 2)))
        assert 0.05 < r_stem / (r_track + 1e-9) < 5.0, (
            f"stem is {r_stem / (r_track + 1e-9):.2f}x the track's loudness"
        )

    def test_missing_stem_file_is_reported(self, jam_files):
        with pytest.raises(Exception):
            mix_imported_stem(jam_files["a_mix"], jam_files["a_mix"] + ".nope", {})