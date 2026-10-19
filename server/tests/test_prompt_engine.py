"""Unit tests for Version A — deterministic regex prompt engine."""

import pytest

from server.ml.prompt_engine import (
    Intent,
    classify_intent,
    extract_instrument,
    extract_time_range,
    normalize_prompt,
    parse_timestamp,
    plan_from_prompt,
    regex_plan_from_prompt,
    suggest_alternatives,
)


class TestCasualPhrasing:
    """Everyday wording must reach the same intent as the formal phrasing.

    The prompt bench is written alongside the keyword tables, so it confirms
    the rules agree with themselves rather than with how people speak. These
    cases are the ones that were previously falling through to UNKNOWN.
    """

    CASES = [
        ("pls cut the boring intro its too long", Intent.TRIM),
        ("get rid of that awful silence in the middle", Intent.REMOVE_SILENCE),
        ("kill the hiss", Intent.REMOVE),
        ("take out the ums and ahs", Intent.REMOVE_FILLERS),
        ("i want just the singing no music", Intent.ISOLATE),
        ("just the vocals no music", Intent.ISOLATE),
        ("boost the vocals a bit pls", Intent.BOOST),
        ("too quiet can you turn it up", Intent.BOOST),
        ("shorten to 30 secs for reels", Intent.TRIM),
        ("make it sound darker and moodier", Intent.MOOD),
        ("make it darker", Intent.MOOD),
        ("can we make it more cinematic", Intent.STYLE),
        ("add some drums that fit", Intent.ADD_INSTRUMENT),
        ("same speakers same volume pls", Intent.NORMALIZE),
        ("fade the end out pls", Intent.FADE),
        ("match it with my other track", Intent.MIX_STEM),
        ("combine these two files", Intent.MIX_STEM),
        ("2x speed", Intent.SPEED),
        # normalize_prompt rewrites these to a bare "boost", which used to
        # match neither "boost " nor "boost the" and fall through to UNKNOWN.
        ("make it louder", Intent.BOOST),
        ("boost", Intent.BOOST),
        ("amplify this", Intent.BOOST),
    ]

    @pytest.mark.parametrize("prompt,expected", CASES)
    def test_casual_prompt_is_accepted(self, prompt, expected):
        assert classify_intent(prompt) == expected, (
            f"{prompt!r} -> {classify_intent(prompt)}, expected {expected}"
        )

    def test_just_is_not_treated_as_filler(self):
        """"just the vocals" means *only*; stripping it broke ISOLATE."""
        assert "just" in normalize_prompt("just the vocals")

    def test_darker_stays_a_mood_request(self):
        """STYLE is matched before MOOD, so "dark" must not swallow "darker"."""
        assert classify_intent("make it darker") == Intent.MOOD
        assert classify_intent("make it darker") != Intent.STYLE

    def test_vague_prompt_returns_usable_suggestions(self):
        for prompt in ("can u make it sound way better", "asdf qwerty", "hmm do something cool"):
            ideas = suggest_alternatives(prompt)
            assert ideas, f"no suggestions for {prompt!r}"
            # Suggestions must themselves be prompts the engine understands,
            # otherwise we are sending the user in circles.
            for idea in ideas:
                assert classify_intent(idea) != Intent.UNKNOWN, idea

    def test_suggestions_follow_the_topic(self):
        assert any(
            "voice" in s.lower() for s in suggest_alternatives("my voice is bad")
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