"""New-school grooves (garage/amapiano/afro/jungle/grime) + editing options."""

import numpy as np
import soundfile as sf

from server.ml import audio_operations as AO
from server.ml.prompt_engine import Intent, extract_groove, regex_plan_from_prompt

SR = 22050


def _tone_wav(tmp_path, name="t.wav", seconds=10.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)
    p = str(tmp_path / name)
    sf.write(p, y, SR)
    return p


def test_new_grooves_parse():
    assert extract_groove("add uk garage drums") == "garage"
    assert extract_groove("add a 2-step beat") == "garage"
    assert extract_groove("add amapiano drums") == "amapiano"
    assert extract_groove("add an afro house groove") == "afro_house"
    assert extract_groove("add a jungle break") == "jungle"
    assert extract_groove("add a grime beat") == "grime"


def test_new_instruments_parse():
    assert regex_plan_from_prompt("add amapiano").params["instrument"] == "amapiano"
    assert regex_plan_from_prompt("add piano").params["instrument"] == "piano"
    assert regex_plan_from_prompt("add jungle").params["instrument"] == "jungle"
    assert regex_plan_from_prompt("add dnb break").params["instrument"] == "dnb"
    assert regex_plan_from_prompt("convert to garage style").params["style"] == "garage"
    assert regex_plan_from_prompt("convert to amapiano style").params["style"] == "amapiano"


def test_new_intents_parse():
    assert regex_plan_from_prompt("reverse the intro").intent == Intent.REVERSE
    assert regex_plan_from_prompt("play it backwards").intent == Intent.REVERSE
    # "reverse bass" is a sound, not the reverse effect
    assert regex_plan_from_prompt("add reverse bass").intent == Intent.ADD_INSTRUMENT
    r = regex_plan_from_prompt("loop the chorus 3 times")
    assert r.intent == Intent.REPEAT and r.params["times"] == 3
    assert regex_plan_from_prompt("repeat the intro").params["times"] == 2
    t = regex_plan_from_prompt("pitch it up 2 semitones")
    assert t.intent == Intent.TRANSPOSE and t.params["semitones"] == 2.0
    assert regex_plan_from_prompt("transpose down").params["semitones"] == -2.0
    assert regex_plan_from_prompt("pitch it down 3 semitones").params["semitones"] == -3.0


def test_new_drum_patterns_execute(tmp_path):
    path = _tone_wav(tmp_path)
    for groove in ("garage", "amapiano", "afro_house", "jungle", "grime"):
        plan = regex_plan_from_prompt(f"add {groove} drums")
        assert plan.intent == Intent.ADD_INSTRUMENT, groove
        res = AO.execute_plan(path, plan)
        assert res.get("output_path"), groove
        assert res["metadata"]["hits"] > 0, groove


def test_new_styles_execute(tmp_path):
    path = _tone_wav(tmp_path)
    for style in ("garage", "amapiano", "afro_house", "jungle", "grime"):
        res = AO.execute_plan(path, regex_plan_from_prompt(f"convert to {style} style"))
        assert res["intent"] == "style", style
        assert res.get("output_path"), style


def test_reverse_roundtrip(tmp_path):
    path = _tone_wav(tmp_path)
    once = AO.execute_plan(path, regex_plan_from_prompt("reverse it"))["output_path"]
    y1, _ = sf.read(once, always_2d=True)
    twice = AO.execute_plan(once, regex_plan_from_prompt("reverse it"))["output_path"]
    y2, _ = sf.read(twice, always_2d=True)
    y0, _ = sf.read(path, always_2d=True)
    n = min(len(y0), len(y2))
    assert np.max(np.abs(y0[:n, 0] - y2[:n, 0])) < 1e-4


def test_repeat_duration(tmp_path):
    path = _tone_wav(tmp_path)
    res = AO.execute_plan(path, regex_plan_from_prompt("loop from 0:02 to 0:04 3 times"))
    assert res["metadata"]["times"] == 3
    info = sf.info(res["output_path"])
    assert abs(info.duration - (10.0 + 2 * 2.0)) < 0.15


def test_transpose_keeps_duration(tmp_path):
    path = _tone_wav(tmp_path)
    res = AO.execute_plan(path, regex_plan_from_prompt("pitch it up 2 semitones"))
    assert res["metadata"]["semitones"] == 2.0
    assert abs(sf.info(res["output_path"]).duration - 10.0) < 0.2
