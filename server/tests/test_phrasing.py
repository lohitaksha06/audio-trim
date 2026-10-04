"""Phrase model: does a generated part actually move?

The complaint these tests lock down is "the bass was a static note". So the
assertions are about *variety and phrasing*, not just "did it produce audio".

Skips if the phrase model has not been trained yet.
"""

import numpy as np
import pytest

from server.ml.synthesis import phrasing, timbre

pytestmark = pytest.mark.skipif(
    not phrasing.available(),
    reason="phrase model not built - run scripts/train_phrasing_model.py",
)

FAMS = [f.split("/")[0] for f in timbre.families()]


def _part(**kw):
    base = dict(family="bass", bpm=120.0, bars=4, root_midi=33, sr=22050, seed=1)
    base.update(kw)
    return phrasing.generate_part(**base)


class TestItMoves:
    def test_is_not_one_static_note(self):
        _, meta = _part()
        assert meta["distinct_pitches"] >= 3, (
            f"only {meta['distinct_pitches']} distinct pitch(es) - that is the "
            "static-note problem we are trying to fix"
        )

    def test_has_more_than_one_idea(self):
        _, meta = _part()
        assert meta["phrases"] >= 2, (
            f"{meta['phrases']} phrase(s) across 4 bars - a real part changes "
            "every 2 bars or so"
        )

    def test_has_some_rests(self):
        """Space is part of phrasing; a note on every 16th reads as a machine."""
        _, meta = _part()
        assert meta["rests"] > 0, "no rests at all - too dense to sound played"

    def test_variety_scales_with_length(self):
        _, short = _part(bars=2)
        _, long = _part(bars=8)
        assert long["notes"] > short["notes"]
        assert long["distinct_pitches"] >= short["distinct_pitches"]


class TestMusicalConstraints:
    @pytest.mark.parametrize("root", [33, 40, 45])
    def test_notes_stay_in_the_scale(self, root):
        _, meta = _part(root_midi=root)
        assert meta["pitches"], "no pitches reported"
        off = [p for p in meta["pitches"] if (p - root) % 12 not in phrasing.NAT_MINOR]
        assert not off, f"notes outside natural minor: {off} (root {root})"

    def test_notes_stay_in_range(self):
        lo, hi = 28, 55
        _, meta = _part(family="bass", pitch_lo=lo, pitch_hi=hi)
        out = [p for p in meta["pitches"] if not (lo <= p <= hi)]
        assert not out, f"pitches outside {lo}-{hi}: {out}"

    def test_is_monophonic_not_a_chord(self):
        """Notes must not overlap, or the part turns into a pad."""
        _, meta = _part()
        ends = []
        for s, ln in zip(meta["note_steps"], meta["note_lengths"]):
            ends.append((s, s + ln))
        ends.sort()
        for (_, prev_end), (start, _) in zip(ends, ends[1:]):
            assert start >= prev_end, (
                f"notes overlap at step {start} (previous ended {prev_end}) - "
                "the part is not monophonic"
            )


class TestRegressions:
    """Bugs found by measuring generated output, not by reading the code."""

    def test_a_quiet_track_still_gets_a_part(self):
        """Regression: a fully quiet energy curve produced ZERO notes.

        The state fed the model the *note's own* loudness, which is
        structurally 0 on every rest row, so it learned "energy 0 == rest" and
        'add bass' on a quiet track rendered pure silence.
        """
        for e in (0.0, 0.05):
            y, meta = _part(bars=8, energy=[e] * 128)
            assert meta["notes"] > 0, (
                f"energy={e} produced no notes - the part is silent"
            )
            assert np.any(y), f"energy={e} produced silent audio"

    def test_fill_stays_in_a_usable_band(self):
        """Left to the model the part came out ~10% filled - too sparse."""
        for density in (0.5, 0.8, 1.0):
            _, meta = _part(bars=8, density=density, energy=[0.6] * 128)
            total = meta["notes"] + meta["rests"]
            fill = meta["notes"] / max(total, 1)
            assert 0.15 <= fill <= 0.95, (
                f"density {density}: fill {fill:.0%} is not a usable part"
            )

    def test_no_stuttering_same_pitch_runs(self):
        """Contiguous same-pitch notes must merge, not re-strike."""
        for seed in (1, 5, 9):
            _, meta = _part(bars=8, seed=seed)
            pitches = meta["pitches"]
            starts = meta["note_steps"]
            lens = meta["note_lengths"]
            for i in range(1, len(pitches)):
                if pitches[i] == pitches[i - 1]:
                    touching = starts[i] == starts[i - 1] + lens[i - 1]
                    assert not touching, (
                        f"seed {seed}: pitch {pitches[i]} re-struck at step "
                        f"{starts[i]} instead of being held"
                    )

    def test_notes_have_varied_lengths(self):
        """All-same-length notes are a machine-gun, not a part."""
        _, meta = _part(bars=8, energy=[0.6] * 128)
        assert len(set(meta["note_lengths"])) >= 3, (
            f"note lengths: {sorted(set(meta['note_lengths']))}"
        )

    def test_energy_increases_density_monotonically(self):
        fills = []
        for e in (0.0, 0.5, 1.0):
            _, meta = _part(bars=8, density=1.0, energy=[e] * 128)
            fills.append(meta["notes"] / max(meta["notes"] + meta["rests"], 1))
        assert fills[0] < fills[-1], f"density did not follow energy: {fills}"

    def test_never_returns_silence_for_any_seed(self):
        for seed in range(1, 25):
            y, meta = _part(bars=4, seed=seed)
            assert meta["notes"] > 0, f"seed {seed} produced no notes"
            assert np.any(y), f"seed {seed} produced silent audio"


class TestRespondsToContext:
    def test_energy_curve_changes_density(self):
        sr, steps = 22050, 64
        step = 60.0 / 120.0 / 4.0
        quiet = [0.02] * steps
        loud = [1.0] * steps
        _, q = _part(bars=4, energy=quiet)
        _, l = _part(bars=4, energy=loud)
        assert l["notes"] > q["notes"], (
            f"energy ignored: quiet={q['notes']} notes, loud={l['notes']} notes"
        )

    def test_density_parameter_is_respected(self):
        _, sparse = _part(density=0.2)
        _, dense = _part(density=1.0)
        assert dense["notes"] > sparse["notes"]

    def test_tempo_changes_length(self):
        _, slow = _part(bpm=90.0)
        _, fast = _part(bpm=150.0)
        assert slow["seconds"] > fast["seconds"]


class TestDeterminismAndSafety:
    def test_same_seed_gives_same_audio(self):
        a, _ = _part(seed=7)
        b, _ = _part(seed=7)
        assert np.allclose(a, b)

    def test_different_seed_gives_different_audio(self):
        a, _ = _part(seed=1)
        b, _ = _part(seed=2)
        assert not np.allclose(a, b)

    def test_output_is_bounded_and_audible(self):
        y, _ = _part()
        assert np.max(np.abs(y)) <= 1.0
        assert np.max(np.abs(y)) > 0.05
        assert np.all(np.isfinite(y))

    def test_every_trained_family_can_be_generated(self):
        for fam in sorted(set(FAMS)):
            y, meta = _part(family=fam, bars=2)
            assert y.size > 0, fam
            assert meta["notes"] > 0, f"{fam} produced no notes"
