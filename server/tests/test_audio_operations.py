"""Unit tests for audio operations (Version A)."""

import numpy as np
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


def test_remove_silence_honours_pause_threshold(tmp_path):
    """A 2 s pause is removed at 0.5 s, but kept at 3 s.

    The old `_remove_silence` concatenated every non-silent island, so the
    threshold in "cut pauses over X" was ignored. This is the regression test
    for that: same audio, two thresholds, two different durations.
    """
    sr = 22050

    def tone(freq, dur, amp=0.5):
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        return amp * np.sin(2 * np.pi * freq * t)

    signal = np.concatenate([tone(440, 1.0), np.zeros(sr * 2), tone(660, 1.0)])
    path = tmp_path / "pauses.wav"
    sf.write(str(path), signal, sr)

    plan = regex_plan_from_prompt("remove pauses longer than 0.5 seconds")
    assert plan.intent == Intent.REMOVE_SILENCE
    assert plan.params.get("max_pause") == 0.5
    result = execute_plan(str(path), plan)
    duration = sf.info(result["output_path"]).duration
    assert 1.9 < duration < 2.4, duration
    assert result["metadata"]["min_pause_seconds"] == 0.5
    assert result["metadata"]["removed_seconds"] > 1.5

    plan = regex_plan_from_prompt("remove pauses longer than 3 seconds")
    result = execute_plan(str(path), plan)
    assert round(sf.info(result["output_path"]).duration, 1) == 4.0


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
    """"...and fill smoothly" must reconstruct, not delete.

    The old path returned a 0.5 s shorter file. Inpainting now keeps the
    timeline and fills the hole, so the duration is unchanged.
    """
    plan = regex_plan_from_prompt("remove that cymbal crash at 0:05 and fill smoothly")
    result = execute_plan(tone_path, plan)
    assert result["intent"] == "paint"
    assert round(sf.info(result["output_path"]).duration, 1) == 10.0

    info = (result.get("metadata") or {}).get("inpainted") or {}
    assert info.get("reconstructed") is True
    assert info.get("method")


def test_separate_returns_stems(tone_path):
    """Demucs runs and returns the four stems (slow, heavy)."""
    plan = regex_plan_from_prompt("separate into stems")
    result = execute_plan(tone_path, plan)
    stems = result.get("stems") or {}
    assert {"vocals", "drums", "bass", "other"}.issubset(stems.keys())
    for p in stems.values():
        assert sf.info(p).samplerate > 0