"""EDM catalog, synth waves, stem mixer (mix intent) and Mix Doctor."""

import numpy as np
import soundfile as sf

from server.ml.prompt_engine import (
    Intent,
    extract_groove,
    extract_instrument,
    extract_instruments,
    extract_mix_gains,
    extract_wave,
    regex_plan_from_prompt,
)
from server.ml.audio_operations import execute_plan


class TestEDMIntents:
    def test_new_kits_parse(self):
        for prompt, inst in [
            ("add techno drums", "drums"),
            ("add a phonk cowbell", "phonk"),
            ("add synthwave pad", "synthwave"),
            ("add an 808 bass", "808"),
            ("add a supersaw lead", "supersaw"),
            ("add a reese bass", "reese"),
            ("add an acid bassline", "acid"),
            ("add trumpet", "trumpet"),
            ("add choir pad", "choir"),
            ("add hardstyle kick", "drums"),
        ]:
            plan = regex_plan_from_prompt(prompt)
            assert plan.intent == Intent.ADD_INSTRUMENT, f"{prompt} -> {plan.intent}"
            assert plan.params["instrument"] == inst, f"{prompt} -> {plan.params}"

    def test_new_grooves(self):
        assert extract_groove("add techno drums") == "techno"
        assert extract_groove("add trance arp") == "trance"
        assert extract_groove("add trap hats") == "trap"
        assert extract_groove("add dnb break") == "dnb"
        assert extract_groove("add hardstyle kick") == "hardstyle"
        assert extract_groove("add phonk cowbell") == "phonk"
        assert extract_groove("add synthwave pad") == "synthwave"
        assert extract_groove("add deep house chords") == "deep_house"

    def test_new_styles(self):
        for style in ["techno", "trance", "hardstyle", "synthwave", "deep_house", "phonk", "dnb"]:
            plan = regex_plan_from_prompt(f"convert to {style} style")
            assert plan.intent == Intent.STYLE, f"{style} -> {plan.intent}"
            assert plan.params["style"] == style

    def test_waves(self):
        assert extract_wave("add a supersaw lead") == "supersaw"
        assert extract_wave("add a square wave lead") == "square"
        assert extract_wave("add an acid bassline") == "acid"
        assert extract_wave("add an 808 bass") == "808"
        assert extract_wave("add a reese bass") == "reese"

    def test_compounds_stay_single(self):
        assert regex_plan_from_prompt("add choir pad").params["instrument"] == "choir"
        assert regex_plan_from_prompt("add tropical pluck").params["instrument"] == "tropical"
        assert regex_plan_from_prompt("add techno drums").params["instrument"] == "drums"
        plan = regex_plan_from_prompt("combine drums and bass")
        assert plan.intent == Intent.COMBINE
        assert set(plan.params["instruments"]) >= {"drums", "bass"}


class TestMixIntent:
    def test_multi_target_is_mix(self):
        for prompt in [
            "make the drums louder and the vocals quieter",
            "turn up the drums and turn down the vocals",
            "prioritize drums over vocals",
            "balance the mix: vocals -3dB, drums +6dB",
        ]:
            plan = regex_plan_from_prompt(prompt)
            assert plan.intent == Intent.MIX, f"{prompt} -> {plan.intent}"

    def test_single_target_stays_boost(self):
        assert regex_plan_from_prompt("i cant hear the drums").intent == Intent.BOOST
        assert regex_plan_from_prompt("make the drums louder").intent == Intent.BOOST

    def test_mix_gains(self):
        gains = extract_mix_gains("make drums louder and vocals lower")
        assert gains["drums"] > 0 and gains["vocals"] < 0
        gains = extract_mix_gains("balance the mix: vocals -3dB drums +6dB")
        assert gains == {"vocals": -3.0, "drums": 6.0}
        gains = extract_mix_gains("prioritize drums over vocals")
        assert gains["drums"] > 0 and gains["vocals"] < 0

    def test_mix_executes_eq_fallback(self, tone_path, monkeypatch):
        """Mixer runs without Demucs (fast EQ-balance path) and shifts bands."""
        from server.ml import audio_operations as ops

        class _FailSep:
            def separate(self, *a, **k):
                raise RuntimeError("no model in unit test")

        monkeypatch.setattr(ops, "SourceSeparator", _FailSep)
        plan = regex_plan_from_prompt("make the drums louder and the vocals quieter")
        res = execute_plan(tone_path, plan)
        assert res["intent"] == "mix"
        assert res.get("output_path")
        assert res["metadata"]["method"] == "eq_balance"
        assert res["metadata"]["gains_db"]["drums"] > 0
        assert res["metadata"]["gains_db"]["vocals"] < 0


def _band_energy(path, fmin, fmax):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(np.abs(S[band]) ** 2))


class TestNewLayersAudible:
    def test_808_raises_sub(self, tone_path):
        res = execute_plan(tone_path, regex_plan_from_prompt("add an 808 bass"))
        assert res["intent"] == "add_instrument"
        assert _band_energy(res["output_path"], 0, 150) > _band_energy(tone_path, 0, 150)

    def test_supersaw_adds_highs(self, tone_path):
        res = execute_plan(tone_path, regex_plan_from_prompt("add a supersaw lead"))
        assert res["intent"] == "add_instrument"
        assert _band_energy(res["output_path"], 2000, 8000) > _band_energy(tone_path, 2000, 8000)

    def test_techno_style_executes(self, tone_path):
        res = execute_plan(tone_path, regex_plan_from_prompt("convert to techno style"))
        assert res["intent"] == "style"
        assert res["metadata"]["groove"] == "techno"

    def test_hardstyle_style_executes(self, tone_path):
        res = execute_plan(tone_path, regex_plan_from_prompt("convert to hardstyle style"))
        assert res["intent"] == "style"
        assert res.get("output_path")

    def test_trumpet_executes(self, tone_path):
        res = execute_plan(tone_path, regex_plan_from_prompt("add trumpet"))
        assert res["intent"] == "add_instrument"
        assert res.get("output_path")


class TestMixDoctor:
    def test_analyze_tone(self, tone_path):
        from server.ml.audio_understanding.mix_doctor import analyze_mix

        doc = analyze_mix(tone_path)
        assert 0 <= doc["score"] <= 100
        assert doc["summary"]["peak"] > 0
        assert len(doc["tips"]) >= 1
        for tip in doc["tips"]:
            assert set(tip) >= {"severity", "title", "detail", "fix_prompt"}

    def test_optimize_route(self, tone_path):
        from fastapi.testclient import TestClient
        from server.main import app

        c = TestClient(app)
        r = c.post("/api/ml/optimize", json={"audio_path": tone_path})
        assert r.status_code == 200, r.text[:300]
        assert "tips" in r.json()

    def test_catalog_route(self):
        from fastapi.testclient import TestClient
        from server.main import app

        c = TestClient(app)
        r = c.get("/api/ml/catalog")
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        ids = {i["id"] for i in body["instruments"]}
        assert {"808", "supersaw", "techno", "trumpet", "choir", "phonk"} <= ids
        assert "hardstyle" in body["grooves"]

    def test_mix_route_end_to_end(self, tone_path, monkeypatch):
        """POST /api/process with a mix prompt returns gains metadata."""
        from fastapi.testclient import TestClient
        from server.main import app
        from server.ml import audio_operations as ops

        class _FailSep:
            def separate(self, *a, **k):
                raise RuntimeError("no model in unit test")

        monkeypatch.setattr(ops, "SourceSeparator", _FailSep)
        c = TestClient(app)
        r = c.post("/api/process", json={
            "audio_path": tone_path,
            "prompt": "make the drums louder and the vocals quieter",
        })
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert body["intent"] == "mix"
        assert body["metadata"]["gains_db"]["drums"] > 0
