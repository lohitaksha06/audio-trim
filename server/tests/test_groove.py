"""TDD RED: beat-synced add-instrument, vocal enhance, style presets."""

import soundfile as sf

from server.ml.prompt_engine import Intent, regex_plan_from_prompt
from server.ml.audio_operations import execute_plan


def _tone_with_beat(tmp_path_factory=None, sr=22050, bpm=100, beats=8):
    """Generate click-track + piano-ish tone at known BPM for beat tests."""
    import numpy as np
    import soundfile as sf_mod
    import tempfile, os
    beat_sec = 60.0 / bpm
    dur = beat_sec * beats
    n = int(sr * dur)
    y = np.zeros(n, dtype=np.float32)
    # piano-ish chord bed
    t = np.arange(n) / sr
    y += 0.2 * np.sin(2 * np.pi * 261.63 * t)
    # kick clicks on beats so beat tracker locks
    for b in range(beats):
        s = int(b * beat_sec * sr)
        e = min(s + int(0.03 * sr), n)
        ct = np.arange(e - s) / sr
        y[s:e] += 0.8 * np.exp(-ct * 60) * np.sin(2 * np.pi * 60 * ct)
    p = os.path.join(tempfile.gettempdir(), "groove_test.wav")
    sf_mod.write(p, y, sr)
    return p


def test_groove_params_parsed():
    plan = regex_plan_from_prompt("add funky syncopated drums at 100 bpm")
    assert plan.intent == Intent.ADD_INSTRUMENT
    assert plan.params.get("instrument") == "drums"
    assert plan.params.get("groove") in {"funky", "syncopated", "four_on_floor", "half_time", "double_time", "default"}


def test_house_style_intent():
    plan = regex_plan_from_prompt("convert this song into a house music style")
    assert plan.intent == Intent.STYLE
    assert plan.params.get("style") == "house"


def test_tropical_style_intent():
    plan = regex_plan_from_prompt("make it tropical edm style")
    assert plan.intent == Intent.STYLE
    assert "tropical" in plan.params.get("style", "")


def test_vocal_enhance_intent():
    plan = regex_plan_from_prompt("make voices clearer and remove background noise")
    assert plan.intent == Intent.ENHANCE_VOCALS


def test_add_drums_beat_synced():
    path = _tone_with_beat(bpm=100, beats=8)
    plan = regex_plan_from_prompt("add drums")
    res = execute_plan(path, plan)
    assert res["intent"] == "add_instrument"
    assert res.get("output_path")
    meta = res.get("metadata", {})
    # must report detected tempo + beat count — proves it listened to the song
    assert "tempo_bpm" in meta
    assert 80 <= meta["tempo_bpm"] <= 130
    assert meta.get("beat_count", 0) >= 4


def test_add_bass_uses_key():
    path = _tone_with_beat(bpm=100, beats=8)
    plan = regex_plan_from_prompt("add bass guitar following the groove")
    res = execute_plan(path, plan)
    assert res["intent"] == "add_instrument"
    assert res.get("metadata", {}).get("added_instrument") in {"bass", "guitar"}


def test_style_house_runs():
    path = _tone_with_beat(bpm=120, beats=8)
    plan = regex_plan_from_prompt("convert this into house music style")
    res = execute_plan(path, plan)
    assert res["intent"] == "style"
    assert res.get("output_path")


def test_vocal_enhance_runs(tone_path):
    plan = regex_plan_from_prompt("enhance vocals and denoise")
    res = execute_plan(tone_path, plan)
    assert res["intent"] == "enhance_vocals"
    assert res.get("output_path")
