"""Unit tests for Version A — deterministic regex prompt engine."""

import pytest

from server.ml.prompt_engine import (
    Intent,
    classify_intent,
    extract_instrument,
    extract_time_range,
    parse_timestamp,
    plan_from_prompt,
    regex_plan_from_prompt,
)


class TestIntentClassification:
    CASES = [
        ("trim the intro", Intent.TRIM),
        ("cut from 0:05", Intent.TRIM),
        ("remove the section from 0:02 to 0:04", Intent.REMOVE),
        ("remove the drums", Intent.REMOVE),
        ("remove the kick drum", Intent.REMOVE),
        ("remove ums and ahs", Intent.REMOVE_FILLERS),
        ("remove the silence", Intent.REMOVE_SILENCE),
        ("separate into stems", Intent.SEPARATE),
        ("split the audio", Intent.SEPARATE),
        ("convert to mp3", Intent.CONVERT),
        ("export as wav", Intent.CONVERT),
        ("keep only the vocals", Intent.ISOLATE),
        ("extract just the bass", Intent.ISOLATE),
        ("fade in and out", Intent.FADE),
        ("normalize the volume", Intent.NORMALIZE),
        ("make it darker", Intent.MOOD),
        ("make it energetic", Intent.MOOD),
        ("speed it up", Intent.SPEED),
        ("twice as fast", Intent.SPEED),
        ("slow it down", Intent.SPEED),
        ("add reverb", Intent.REVERB),
        ("make it echo", Intent.REVERB),
        ("remove that cymbal crash and fill smoothly", Intent.PAINT),
        ("clean up the cough at 0:03", Intent.PAINT),
        ("hello world", Intent.UNKNOWN),
    ]

    @pytest.mark.parametrize("prompt,expected", CASES)
    def test_intent(self, prompt, expected):
        assert classify_intent(prompt) == expected

    def test_filler_substring_bug(self):
        # "drums" contains "um" — must NOT be treated as a filler word
        assert classify_intent("remove the drums") == Intent.REMOVE
        assert classify_intent("remove the kick drum") == Intent.REMOVE


class TestParsers:
    def test_parse_timestamp_mmss(self):
        assert parse_timestamp("2:30") == 150.0

    def test_parse_timestamp_seconds(self):
        assert parse_timestamp("90s") == 90.0

    def test_extract_time_range(self):
        start, end = extract_time_range("trim from 0:02 to 0:08")
        assert (start, end) == (2.0, 8.0)

    def test_extract_time_range_at(self):
        start, end = extract_time_range("at 0:45")
        assert start == 45.0 and end is None

    def test_extract_instrument(self):
        assert extract_instrument("keep only the bass") == "bass"
        assert extract_instrument("remove the cymbal") == "drums"


@pytest.mark.parametrize(
    "prompt,field,value",
    [
        ("speed it up", "speed_factor", 1.5),
        ("make it twice as fast", "speed_factor", 2.0),
        ("slow it down", "speed_factor", 0.75),
        ("make it darker", "mood", "dark"),
        ("convert to mp3", "format", "mp3"),
        ("keep only the vocals", "instrument", "vocals"),
    ],
)
def test_prompt_params(prompt, field, value):
    plan = regex_plan_from_prompt(prompt)
    assert plan.params.get(field) == value


def test_offline_fallback():
    """plan_from_prompt stays deterministic (regex) without an LLM enabled."""
    plan = plan_from_prompt("trim from 0:01 to 0:05")
    assert plan.intent == Intent.TRIM
    assert plan.params == {"start": 1.0, "end": 5.0}