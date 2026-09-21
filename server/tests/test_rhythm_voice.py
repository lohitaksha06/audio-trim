"""Rhythm understanding + voice-clarity tests (all lightweight, no model downloads)."""

import numpy as np
import soundfile as sf

from server.ml import audio_operations as AO
from server.ml.prompt_engine import Intent, regex_plan_from_prompt
from server.ml.rhythm import analyze_rhythm, auto_groove

SR = 22050


def _kick():
    n = int(0.14 * SR)
    t = np.arange(n) / SR
    return np.exp(-t * 28) * np.sin(2 * np.pi * 55 * t)


def _snare():
    n = int(0.16 * SR)
    r = np.random.default_rng(1)
    return (r.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / SR * 30) * 0.5).astype(np.float32)


def _hat():
    n = int(0.04 * SR)
    r = np.random.default_rng(2)
    return (r.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / SR * 120) * 0.3).astype(np.float32)


def _loop(bpm, pat, dur=8.0):
    y = np.zeros(int(dur * SR), dtype=np.float32)
    step = 60.0 / bpm
    K, S, H = _kick(), _snare(), _hat()
    for i in range(int(dur / step)):
        b = i * step
        for kind, at in pat(i % 4):
            h = {"k": K, "s": S, "h": H}[kind]
            s = int((b + at * step) * SR)
            if 0 <= s < len(y):
                y[s:s + len(h)] += h[: max(0, min(len(h), len(y) - s))]
    return y


def _house(p):
    ev = [("k", 0), ("h", 0.5)]
    if p in (1, 3):
        ev.append(("s", 0))
    return ev


def _trap(p):
    ev = [("h", 0.33), ("h", 0.66)]
    if p == 0:
        ev.append(("k", 0))
    if p == 2:
        ev.append(("s", 0))
    return ev


def _wav(tmp_path, y, name="t.wav"):
    p = str(tmp_path / name)
    sf.write(p, y, SR)
    return p


def test_tempo_house_and_trap():
    for bpm, pat in ((128, _house), (150, _trap)):
        r = analyze_rhythm(_loop(bpm, pat), SR)
        assert abs(r["tempo_bpm"] - bpm) / bpm < 0.04, (bpm, r["tempo_bpm"])
        assert r["beat_count"] > 4


def test_auto_groove_matches_family():
    assert auto_groove(_loop(128, _house), SR)["groove"] in ("house", "big_room", "techno")
    assert auto_groove(_loop(150, _trap), SR)["groove"] in ("trap", "dubstep", "phonk")


def test_add_drums_auto_metadata(tmp_path):
    path = _wav(tmp_path, _loop(128, _house))
    plan = regex_plan_from_prompt("add drums that match the song")
    assert plan.intent == Intent.ADD_INSTRUMENT
    res = AO.execute_plan(path, plan)
    assert res["intent"] == "add_instrument"
    assert res["metadata"]["groove"] in ("house", "big_room", "techno")
    assert abs(res["metadata"]["tempo_bpm"] - 128) < 6


def _speechy():
    def tone(f, dur, amp):
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        return amp * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t))

    sig = np.concatenate([tone(220, 1.5, 0.6), np.zeros(int(SR * 0.6)),
                          tone(300, 1.5, 0.6), np.zeros(int(SR * 0.6))])
    noise = 0.06 * np.random.default_rng(3).standard_normal(len(sig))
    return np.stack([sig + noise, sig + noise]).astype(np.float32)


def _band(a, f0, f1):
    import librosa
    S = np.abs(librosa.stft(a))
    f = librosa.fft_frequencies(sr=SR)
    return float(np.sum(np.abs(S[(f >= f0) & (f <= f1)]) ** 2))


def test_spectral_subtraction_raises_snr():
    y = _speechy()
    before = _band(y[0], 200, 500) / _band(y[0], 6000, 10000)
    d = AO._spectral_subtract(y, SR, 1.2)
    after = _band(d[0], 200, 500) / _band(d[0], 6000, 10000)
    assert after > before * 1.3


def test_voice_prompt_flags():
    p = regex_plan_from_prompt("remove background noise")
    assert p.intent == Intent.ENHANCE_VOCALS and p.params["denoise_mix"] is True
    p = regex_plan_from_prompt("remove background noise from her voice")
    assert p.params["denoise_mix"] is False
    p = regex_plan_from_prompt("make her voice audible and clear")
    assert p.params["level_boost"] is True
    p = regex_plan_from_prompt("remove ALL background noise, it is very noisy")
    assert p.params["aggressive"] is True


def test_enhance_fallback_without_separator(tmp_path, monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no model here")

    monkeypatch.setattr(AO, "SourceSeparator", Boom)
    path = _wav(tmp_path, _loop(128, _house))
    plan = regex_plan_from_prompt("make the voice clearer")
    res = AO.execute_plan(path, plan)
    assert res["intent"] == "enhance_vocals"
    assert res["metadata"]["method"] == "eq_fallback"
    assert res.get("output_path")


def test_enhance_stem_remix_with_fake_separator(tmp_path, monkeypatch):
    src = _wav(tmp_path, _loop(128, _house), "src.wav")

    class FakeSep:
        def __init__(self, *a, **k):
            pass

        def separate(self, audio_path, out_dir=None):
            import soundfile as _sf
            d = tmp_path / "stems"
            d.mkdir(exist_ok=True)
            y, _ = _sf.read(audio_path, always_2d=True)
            stems = {}
            for name, gain in (("vocals", 1.0), ("drums", 0.0), ("bass", 0.0), ("other", 0.0)):
                p = str(d / f"{name}.wav")
                _sf.write(p, (y[:, 0] * gain).astype(np.float32), SR)
                stems[name] = p
            return stems

    monkeypatch.setattr(AO, "SourceSeparator", FakeSep)
    plan = regex_plan_from_prompt("make the voice clearer and louder")
    res = AO.execute_plan(src, plan)
    assert res["metadata"]["method"] == "stem_remix"
    assert res["metadata"]["vocal_boost_db"] >= 4.0
    assert res.get("output_path")


def test_mix_denoise_path(tmp_path):
    y = _speechy()[0]
    path = _wav(tmp_path, y, "n.wav")
    before = _band(y, 200, 500) / _band(y, 6000, 10000)
    plan = regex_plan_from_prompt("remove background noise")
    res = AO.execute_plan(path, plan)
    assert res["metadata"]["method"] == "mix_denoise"
    out, _ = sf.read(res["output_path"], always_2d=True)
    after = _band(out[:, 0], 200, 500) / _band(out[:, 0], 6000, 10000)
    assert after > before
