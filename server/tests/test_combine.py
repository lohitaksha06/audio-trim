"""RED: combine layers, boost quiet stems, base->bass, swing groove, strings."""

import numpy as np
import soundfile as sf

from server.ml.prompt_engine import Intent, extract_instrument, regex_plan_from_prompt
from server.ml.audio_operations import execute_plan


def test_base_guitar_is_bass():
    assert extract_instrument("now add a base guitar in the back too") == "bass"
    plan = regex_plan_from_prompt("now add a base guitar in the back too")
    assert plan.intent == Intent.ADD_INSTRUMENT
    assert plan.params["instrument"] == "bass"


def test_based_does_not_hijack():
    # "based on the groove" must not invent a bass instrument
    assert extract_instrument("add drums based on the groove") == "drums"


def test_combine_intents():
    for p in [
        "can u comebin both drum and base",
        "combine both drums and bass",
        "add both drums and bass",
        "layer drums and bass together",
        "merge drums, bass and guitar",
    ]:
        plan = regex_plan_from_prompt(p)
        assert plan.intent == Intent.COMBINE, f"{p} -> {plan.intent}"
    plan = regex_plan_from_prompt("combine both drums and bass")
    assert set(plan.params["instruments"]) >= {"drums", "bass"}


def test_single_add_not_combine():
    plan = regex_plan_from_prompt("can you add drums in the background")
    assert plan.intent == Intent.ADD_INSTRUMENT


def test_boost_intent():
    for p in ["i cant hear the drums", "the bass is too quiet", "make the drums louder", "boost the vocals"]:
        plan = regex_plan_from_prompt(p)
        assert plan.intent == Intent.BOOST, f"{p} -> {plan.intent}"
    assert regex_plan_from_prompt("i cant hear the drums").params["target"] == "drums"


def test_swing_groove_parses():
    plan = regex_plan_from_prompt("add swing drums")
    assert plan.intent == Intent.ADD_INSTRUMENT
    assert plan.params["groove"] == "swing"


def test_strings_executes(tone_path):
    plan = regex_plan_from_prompt("add strings")
    res = execute_plan(tone_path, plan)
    assert res["intent"] == "add_instrument"
    assert res.get("output_path")


def _band_energy(path, fmin, fmax):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(np.abs(S[band]) ** 2))


def test_combine_executes_both(tone_path):
    plan = regex_plan_from_prompt("combine drums and bass")
    res = execute_plan(tone_path, plan)
    assert res["intent"] == "combine"
    assert res.get("output_path")
    assert set(res["metadata"]["combined"]) >= {"drums", "bass"}
    out = res["output_path"]
    # drums: kick-band energy up vs input
    assert _band_energy(out, 0, 150) > _band_energy(tone_path, 0, 150) * 1.5


def test_boost_drums_raises_kick_band(tone_path):
    plan = regex_plan_from_prompt("i cant hear the drums")
    res = execute_plan(tone_path, plan)
    assert res["intent"] == "boost"
    assert res.get("output_path")
    assert _band_energy(res["output_path"], 0, 200) > _band_energy(tone_path, 0, 200) * 1.2


def test_chained_tempo_stays_musical(tone_path):
    """Two stacked layers must not double the grid each time."""
    from server.ml.audio_operations import _add_instrument
    import librosa

    plan = regex_plan_from_prompt("add drums")
    r1 = execute_plan(tone_path, plan)
    y, sr = librosa.load(r1["output_path"], sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    y2, meta2, _ = _add_instrument(y, sr, "bass", {"groove": "default"})
    assert meta2["tempo_bpm"] <= 180, meta2


def test_process_accepts_storage_key():
    """Chaining: a previous download_key must work as the next audio_path."""
    from fastapi.testclient import TestClient
    from server.main import app

    c = TestClient(app)
    import tempfile, os
    sr = 22050
    t = np.linspace(0, 4, sr * 4, endpoint=False)
    p = os.path.join(tempfile.gettempdir(), "chain_src.wav")
    sf.write(p, 0.4 * np.sin(2 * np.pi * 220 * t), sr)
    up = c.post("/api/upload", files={"file": ("c.wav", open(p, "rb"), "audio/wav")})
    assert up.status_code == 200
    ap = up.json()["audio_path"]
    r1 = c.post("/api/process", json={"audio_path": ap, "prompt": "add drums"})
    assert r1.status_code == 200
    key = r1.json()["download_key"]
    assert key
    # feed the storage key back in (what the frontend chains with)
    r2 = c.post("/api/process", json={"audio_path": key, "prompt": "add bass"})
    assert r2.status_code == 200, r2.text[:300]
    assert r2.json()["intent"] == "add_instrument"
