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
