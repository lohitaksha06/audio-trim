"""RED: added drums must be audibly present, and a drums-only stem must ship."""

import numpy as np
import soundfile as sf

from server.ml.prompt_engine import regex_plan_from_prompt
from server.ml.audio_operations import execute_plan


def _sparse_piano(sr=22050, bpm=96, bars=4):
    """Sparse piano chords (one per bar) — drums must clearly show up in gaps."""
    beat = 60.0 / bpm
    bar = beat * 4
    dur = bar * bars
    n = int(sr * dur)
    y = np.zeros(n, dtype=np.float32)
    for b in range(bars):
        s = int(b * bar * sr)
        e = min(s + int(1.2 * sr), n)
        f = [261.63, 293.66, 329.63, 392.0][b % 4]
        tt = np.arange(e - s) / sr
        y[s:e] += 0.35 * np.sin(2 * np.pi * f * tt) * np.exp(-tt * 2.5)
    import tempfile, os
    p = os.path.join(tempfile.gettempdir(), "sparse_piano.wav")
    sf.write(p, y, sr)
    return p, sr


def _lowband_energy(path, fmax=150.0):
    import librosa
    y, sr = librosa.load(path, sr=22050, mono=True)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    band = freqs <= fmax
    return float(np.sum(np.abs(S[band]) ** 2))


def test_added_drums_are_audible():
    src, _ = _sparse_piano()
    before = _lowband_energy(src)
    plan = regex_plan_from_prompt("can you add drums in the background")
    assert plan.intent.value == "add_instrument"
    res = execute_plan(src, plan)
    out = res["output_path"]
    after = _lowband_energy(out)
    # kicks live <150 Hz — output must carry clearly more kick-band energy
    assert after > before * 2.0, f"drums buried: {before=} {after=}"


def test_added_drums_overall_level_up():
    src, _ = _sparse_piano()
    y0, _ = sf.read(src)
    plan = regex_plan_from_prompt("add drums")
    res = execute_plan(src, plan)
    y1, _ = sf.read(res["output_path"])
    rms0 = float(np.sqrt(np.mean(y0 ** 2)))
    rms1 = float(np.sqrt(np.mean(y1 ** 2)))
    assert rms1 > rms0 * 1.15, f"mix too quiet: {rms0=} {rms1=}"


def test_drums_layer_stem_shipped():
    src, _ = _sparse_piano()
    plan = regex_plan_from_prompt("add drums")
    res = execute_plan(src, plan)
    layer = res.get("layer_path")
    assert layer, "no drums-only stem returned"
    info = sf.info(layer)
    assert info.duration > 1.0
    # stem must be drums only (near-silent where no hits? at least non-silent overall)
    y, _ = sf.read(layer)
    assert float(np.max(np.abs(y))) > 0.3, "drum stem too quiet to be useful"
