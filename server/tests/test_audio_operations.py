"""Unit tests for audio operations (Version A)."""

import soundfile as sf

from server.ml.audio_operations import execute_plan
from server.ml.prompt_engine import Intent, regex_plan_from_prompt


def _run(tone_path, prompt, expected_intent: Intent):
    plan = regex_plan_from_prompt(prompt)
    result = execute_plan(tone_path, plan)
    assert result["intent"] == expected_intent.value
    assert result.get("output_path") or result.get("stems")
    return result


def test_trim_duration(tone_path):
    result = _run(tone_path, "trim from 0:02 to 0:08", Intent.TRIM)
    assert round(sf.info(result["output_path"]).duration, 1) == 6.0


def test_remove_duration(tone_path):
    result = _run(tone_path, "remove the section from 0:02 to 0:04", Intent.REMOVE)
    assert round(sf.info(result["output_path"]).duration, 1) == 8.0


def test_mood_operation(tone_path):
    plan = regex_plan_from_prompt("make it darker")
    result = execute_plan(tone_path, plan)
    assert result["intent"] == "mood"
    assert result.get("output_path")


def test_speed_operation(tone_path):
    result = _run(tone_path, "speed it up", Intent.SPEED)
    assert round(sf.info(result["output_path"]).duration, 1) == 6.7


def test_convert(tone_path):
    plan = regex_plan_from_prompt("convert to mp3")
    result = execute_plan(tone_path, plan)
    assert result["metadata"]["format"] == "mp3"
    assert result["output_path"].endswith(".mp3")


def test_paint(tone_path):
    plan = regex_plan_from_prompt("remove that cymbal crash at 0:05 and fill smoothly")
    result = execute_plan(tone_path, plan)
    assert result["intent"] == "paint"
    assert round(sf.info(result["output_path"]).duration, 1) == 9.5


def test_separate_returns_stems(tone_path):
    """Demucs runs and returns the four stems (slow, heavy)."""
    plan = regex_plan_from_prompt("separate into stems")
    result = execute_plan(tone_path, plan)
    stems = result.get("stems") or {}
    assert {"vocals", "drums", "bass", "other"}.issubset(stems.keys())
    for p in stems.values():
        assert sf.info(p).samplerate > 0