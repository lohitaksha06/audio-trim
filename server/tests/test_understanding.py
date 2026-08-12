"""Unit tests for Phase 1 understanding/diarization/inpainting modules."""

import pytest

from server.ml.audio_understanding import (
    classify_instruments,
    compute_mood_curve,
    describe_mood,
    detect_structure,
)
from server.ml.diarization import diarize
from server.ml.inpainting import inpaint


def test_instruments_schema(tone_path):
    res = classify_instruments(tone_path)
    assert "instruments" in res
    assert "texture" in res
    assert res["duration_seconds"] == 10.0


def test_structure_schema(tone_path):
    res = detect_structure(tone_path)
    assert res["sections"][0]["start"] == 0.0
    assert all("label" in s for s in res["sections"])


def test_mood_curve(tone_path):
    res = compute_mood_curve(tone_path)
    assert len(res["curve"]) > 0
    assert {"t", "energy", "tension", "brightness", "loudness"} <= set(res["curve"][0])


def test_describe_mood(tone_path):
    res = describe_mood(tone_path)
    assert res["mood"] in {"dark", "calm", "energetic", "neutral"}


def test_diarize_schema(speech_path):
    res = diarize(speech_path)
    assert {"segments", "speakers", "speech_duration_seconds"} <= set(res)
    assert res["duration_seconds"] == pytest.approx(7.8, abs=0.2)


def test_inpaint_duration(tone_path):
    res = inpaint(tone_path, 2.0, 4.0)
    assert abs(res["new_duration_seconds"] - 8.0) < 0.1
    assert res["removed_start"] == 2.0
    assert res["removed_end"] == 4.0