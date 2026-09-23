"""Sudden-disturbance suppression (horns, drills) + vocal compression.

Synthetic fixtures only: speech-like tone bursts with pauses, a tonal horn
burst (400+500 Hz, 0.6 s) and a broadband drill burst (AM noise, 0.5 s).
"""

import numpy as np
import soundfile as sf

from server.ml import audio_operations as AO
from server.ml.prompt_engine import Intent, regex_plan_from_prompt

SR = 22050


def _speechy_with_disturbances():
    def tone(f, dur, amp):
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        return amp * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t))

    sig = np.concatenate([tone(220, 1.5, 0.5), np.zeros(int(SR * 0.4)),
                          tone(300, 1.5, 0.5), np.zeros(int(SR * 0.4)),
                          tone(250, 1.5, 0.5)])
    # horn burst at ~1.9-2.5 s (inside the first pause + speech edge)
    horn_start = int(SR * 1.9)
    horn_n = int(SR * 0.6)
    th = np.arange(horn_n) / SR
    horn = 0.7 * (np.sin(2 * np.pi * 400 * th) + 0.8 * np.sin(2 * np.pi * 500 * th))
    horn *= np.minimum(1.0, th * 20) * np.minimum(1.0, (0.6 - th) * 20 + 0.2)
    # drill burst at ~3.8-4.3 s: broadband noise with 30 Hz AM + HF content
    drill_start = int(SR * 3.8)
    drill_n = int(SR * 0.5)
    td = np.arange(drill_n) / SR
    rng = np.random.default_rng(11)
    drill = rng.standard_normal(drill_n).astype(np.float32) * (0.5 + 0.5 * np.sin(2 * np.pi * 30 * td))
    drill = (0.55 * drill).astype(np.float32)
    sig[horn_start:horn_start + horn_n] += horn[:max(0, min(horn_n, len(sig) - horn_start))]
    sig[drill_start:drill_start + drill_n] += drill[:max(0, min(drill_n, len(sig) - drill_start))]
    y = np.stack([sig, sig]).astype(np.float32)
    return y, horn_start, horn_n, drill_start, drill_n


def _band(a, f0, f1):
    import librosa
    S = np.abs(librosa.stft(a))
    f = librosa.fft_frequencies(sr=SR)
    return float(np.sum(np.abs(S[(f >= f0) & (f <= f1)]) ** 2))


def test_suppress_horn_tames_tonal_burst():
    y, hs, hn, _, _ = _speechy_with_disturbances()
    horn_win = y[0, hs:hs + hn]
    before = _band(horn_win, 350, 550) / (_band(horn_win, 80, 12000) + 1e-9)
    d, n = AO._suppress_disturbances(y, SR, aggressive=False)
    after = _band(d[0, hs:hs + hn], 350, 550) / (_band(d[0, hs:hs + hn], 80, 12000) + 1e-9)
    assert after < before * 0.6
    assert n >= 1


def test_suppress_drill_tames_broadband_burst():
    y, _, _, ds, dn = _speechy_with_disturbances()
    # broadband dips lower the whole frame: measure drill energy relative to
    # clean speech (first 1 s), a ratio immune to output normalization
    ref_b = _band(y[0, :SR], 80, 12000)
    before = _band(y[0, ds:ds + dn], 80, 12000) / (ref_b + 1e-9)
    d, _ = AO._suppress_disturbances(y, SR, aggressive=False)
    ref_a = _band(d[0, :SR], 80, 12000)
    after = _band(d[0, ds:ds + dn], 80, 12000) / (ref_a + 1e-9)
    assert after < before * 0.6


def test_voice_outside_bursts_preserved():
    y, hs, hn, ds, dn = _speechy_with_disturbances()
    mask = np.ones(y.shape[1], dtype=bool)
    mask[hs - 2205:hs + hn + 2205] = False
    mask[ds - 2205:ds + dn + 2205] = False
    # scale-invariant: voice-band share of total energy outside the windows
    seg_b = y[0][mask][: SR * 2]
    before = _band(seg_b, 200, 300) / (_band(seg_b, 80, 12000) + 1e-9)
    d, _ = AO._suppress_disturbances(y, SR, aggressive=False)
    seg_a = d[0][mask][: SR * 2]
    after = _band(seg_a, 200, 300) / (_band(seg_a, 80, 12000) + 1e-9)
    assert after > before * 0.85


def test_compress_lifts_quiet_word():
    t = np.linspace(0, 1, SR, endpoint=False)
    loud = 0.8 * np.sin(2 * np.pi * 220 * t)
    quiet = 0.1 * np.sin(2 * np.pi * 260 * t)
    y = np.stack([np.concatenate([loud, quiet])]).astype(np.float32)

    def rms(a):
        return float(np.sqrt(np.mean(a ** 2)))

    before = rms(y[0, SR:]) / rms(y[0, :SR])
    d = AO._compress_vocal(y, SR, aggressive=False)
    after = rms(d[0, SR:]) / rms(d[0, :SR])
    assert after > before * 1.3


def test_disturbance_prompt_flags():
    p = regex_plan_from_prompt("remove the car horn from my recording")
    assert p.intent == Intent.ENHANCE_VOCALS and p.params["aggressive"] is True
    p = regex_plan_from_prompt("remove drill noise in the background")
    assert p.intent == Intent.ENHANCE_VOCALS
    p = regex_plan_from_prompt("make me louder and clearer")
    assert p.intent == Intent.ENHANCE_VOCALS and p.params["level_boost"] is True
    p = regex_plan_from_prompt("make my voice more audible")
    assert p.params["level_boost"] is True
    # no hijack: brass horns still belong to music intents
    assert regex_plan_from_prompt("add horns").intent == Intent.ADD_INSTRUMENT
    assert regex_plan_from_prompt("bring horns forward").intent != Intent.ENHANCE_VOCALS


def test_denoise_reports_disturbances(tmp_path):
    y, _, _, _, _ = _speechy_with_disturbances()
    p = str(tmp_path / "d.wav")
    sf.write(p, y[0], SR)
    plan = regex_plan_from_prompt("remove background noise")
    res = AO.execute_plan(p, plan)
    assert res["metadata"]["method"] == "mix_denoise"
    assert "disturbances_removed" in res["metadata"]
