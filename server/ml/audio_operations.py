import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import librosa

from server.ml.prompt_engine import PromptPlan, Intent
from server.ml.source_separation.separator import SourceSeparator
from server.ml.inpainting.inpainter import inpaint
from server.services.converter import convert_file


def execute_plan(audio_path: str, plan: PromptPlan) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]

    output_path = None
    layer_path_out = None
    stems = None
    metadata = {}

    if plan.intent == Intent.TRIM:
        y = _trim(y, sr, plan.params)
        output_path = _save_wav(y, sr)
    elif plan.intent == Intent.REMOVE:
        instrument = plan.params.get("instrument")
        if instrument:
            y = _remove_stem(audio_path, instrument, y, sr)
            output_path = _save_wav(y, sr)
            metadata["removed_stem"] = instrument
        else:
            y = _remove_section(y, sr, plan.params)
            output_path = _save_wav(y, sr)
    elif plan.intent == Intent.PAINT:
        start = plan.params.get("start", 0.0)
        end = plan.params.get("end")
        if end is None:
            end = min(start + 0.5, y.shape[1] / sr)
        res = inpaint(audio_path, start, end)
        output_path = res["output_path"]
        metadata["inpainted"] = {"removed_start": res["removed_start"], "removed_end": res["removed_end"]}
    elif plan.intent == Intent.FADE:
        y = _fade(y, sr, plan.params)
        output_path = _save_wav(y, sr)
    elif plan.intent == Intent.NORMALIZE:
        y = _normalize(y)
        output_path = _save_wav(y, sr)
    elif plan.intent == Intent.REMOVE_SILENCE:
        y = _remove_silence(y, sr)
        output_path = _save_wav(y, sr)
    elif plan.intent == Intent.SPEED:
        y = _speed_change(y, sr, plan.params)
        output_path = _save_wav(y, sr)
    elif plan.intent == Intent.REVERB:
        y = _add_reverb(y, sr)
        output_path = _save_wav(y, sr)
    elif plan.intent == Intent.MOOD:
        y = _mood_adjust(y, sr, plan.params)
        output_path = _save_wav(y, sr)
    elif plan.intent == Intent.REMOVE_FILLERS:
        y = _remove_silence(y, sr, threshold_db=15)
        output_path = _save_wav(y, sr)
    elif plan.intent == Intent.SEPARATE:
        separator = SourceSeparator()
        stems = separator.separate(audio_path)
        metadata["stems"] = stems
    elif plan.intent == Intent.ISOLATE:
        separator = SourceSeparator()
        target = plan.params.get("instrument", "vocals")
        stem_path = separator.isolate(audio_path, target)
        if stem_path:
            output_path = stem_path
            metadata["isolated_stem"] = target
        else:
            stems = separator.separate(audio_path)
            metadata["stems"] = stems
            metadata["note"] = f"Stem '{target}' not found; returning all stems"
    elif plan.intent == Intent.CONVERT:
        target_format = plan.params.get("format", "mp3")
        output_path = convert_file(audio_path, target_format)
        metadata["format"] = target_format
    elif plan.intent == Intent.ADD_INSTRUMENT:
        target = plan.params.get("instrument", "other")
        y, groove_meta, layer = _add_instrument(y, sr, target, plan.params)
        output_path = _save_wav(y, sr)
        metadata["added_instrument"] = target
        metadata.update(groove_meta)
        if layer is not None:
            layer_path_out = _save_wav(layer[np.newaxis, :], sr)
            metadata["layer_kind"] = f"{target}_only"
    elif plan.intent == Intent.COMBINE:
        instruments = plan.params.get("instruments") or ["drums", "bass"]
        stems = {}
        total_hits = 0
        tempo_seen = 0.0
        beats_seen = 0
        for inst in instruments:
            y, gm, layer = _add_instrument(y, sr, inst, {**plan.params, "instrument": inst})
            total_hits += int(gm.get("hits", 0))
            tempo_seen = float(gm.get("tempo_bpm", tempo_seen))
            beats_seen = max(beats_seen, int(gm.get("beat_count", 0)))
            if layer is not None:
                stems[f"{inst}_added"] = _save_wav(layer[np.newaxis, :], sr)
        output_path = _save_wav(y, sr)
        metadata["combined"] = instruments
        metadata["tempo_bpm"] = round(tempo_seen, 1)
        metadata["beat_count"] = beats_seen
        metadata["hits"] = total_hits
        metadata["groove"] = plan.params.get("groove", "default")
    elif plan.intent == Intent.BOOST:
        target = plan.params.get("target", "other")
        y = _boost_target(y, sr, target)
        output_path = _save_wav(y, sr)
        metadata["boosted"] = target
    elif plan.intent == Intent.ENHANCE_VOCALS:
        y = _enhance_vocals(y, sr, plan.params)
        output_path = _save_wav(y, sr)
        metadata["enhanced"] = "vocals"
    elif plan.intent == Intent.STYLE:
        style = plan.params.get("style", "house")
        y, style_meta = _apply_style(y, sr, style, plan.params)
        output_path = _save_wav(y, sr)
        metadata.update(style_meta)

    result: dict[str, Any] = {"intent": plan.intent.value, "params": plan.params}

    if output_path:
        result["output_path"] = output_path
    if layer_path_out:
        result["layer_path"] = layer_path_out
    if stems:
        result["stems"] = stems
    if metadata:
        result["metadata"] = metadata

    return result


def _save_wav(y: np.ndarray, sr: int) -> str:
    output_path = tempfile.mktemp(suffix=".wav")
    sf.write(output_path, y.T if y.shape[0] > 1 else y[0], sr)
    return output_path


def _trim(y: np.ndarray, sr: int, params: dict) -> np.ndarray:
    start = params.get("start", 0)
    end = params.get("end", y.shape[1] / sr)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return y[:, start_sample:end_sample]


def _remove_stem(audio_path: str, instrument: str, y: np.ndarray, sr: int) -> np.ndarray:
    separator = SourceSeparator()
    stem_path = separator.isolate(audio_path, instrument)
    if not stem_path:
        return y
    y_stem, _ = librosa.load(stem_path, sr=sr, mono=False)
    if y_stem.ndim == 1:
        y_stem = y_stem[np.newaxis, :]
    n = min(y.shape[1], y_stem.shape[1])
    return np.clip(y[:, :n] - y_stem[:, :n], -1, 1)


def _remove_section(y: np.ndarray, sr: int, params: dict) -> np.ndarray:
    start = params.get("start", 0)
    end = params.get("end", y.shape[1] / sr)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return np.concatenate([y[:, :start_sample], y[:, end_sample:]], axis=1)


def _fade(y: np.ndarray, sr: int, params: dict) -> np.ndarray:
    fade_in = params.get("fade_in", False)
    fade_out = params.get("fade_out", False)
    fade_len = int(2.0 * sr)

    if fade_in and fade_len > 0:
        fade_len = min(fade_len, y.shape[1])
        ramp = np.linspace(0, 1, fade_len)
        y[:, :fade_len] *= ramp
    if fade_out and fade_len > 0:
        fade_len = min(fade_len, y.shape[1])
        ramp = np.linspace(1, 0, fade_len)
        y[:, -fade_len:] *= ramp
    return y


def _normalize(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9
    return y


def _remove_silence(y: np.ndarray, sr: int, threshold_db: float = 20) -> np.ndarray:
    intervals = librosa.effects.split(y[0], top_db=threshold_db)
    if len(intervals) == 0:
        return y
    result = np.concatenate([y[:, s:e] for s, e in intervals], axis=1)
    return result


def _speed_change(y: np.ndarray, sr: int, params: dict) -> np.ndarray:
    factor = params.get("speed_factor", 1.0)
    result = librosa.effects.time_stretch(y[0], rate=factor)
    return result[np.newaxis, :]


def _add_reverb(y: np.ndarray, sr: int) -> np.ndarray:
    delay_samples = int(0.05 * sr)
    decay = 0.4
    result = y.copy()
    if result.shape[1] > delay_samples:
        result[:, delay_samples:] += y[:, :-delay_samples] * decay
    peak = np.max(np.abs(result))
    if peak > 0:
        result = result / peak * 0.9
    return result


def _mood_adjust(y: np.ndarray, sr: int, params: dict) -> np.ndarray:
    mood = params.get("mood", "neutral")
    if mood == "dark":
        freqs = librosa.fft_frequencies(sr=sr)
        low_boost = np.ones(len(freqs))
        low_boost[freqs < 500] = 1.3
        low_boost[freqs > 4000] = 0.7
        S = librosa.stft(y[0])
        S = S * low_boost[:, np.newaxis]
        y[0] = librosa.istft(S, length=y.shape[1])
    elif mood == "bright":
        freqs = librosa.fft_frequencies(sr=sr)
        high_boost = np.ones(len(freqs))
        high_boost[freqs > 3000] = 1.4
        high_boost[freqs < 200] = 0.8
        S = librosa.stft(y[0])
        S = S * high_boost[:, np.newaxis]
        y[0] = librosa.istft(S, length=y.shape[1])
    elif mood == "energetic":
        y = _normalize(y) * 1.1
        y = np.clip(y, -1, 1)
    return y


def _detect_beats(y: np.ndarray, sr: int) -> tuple[float, list[float]]:
    """Detect tempo + beat times. Falls back to 120 BPM grid (deterministic)."""
    mono = y[0] if y.ndim > 1 else y
    try:
        tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, units="time")
        tempo_f = float(np.atleast_1d(tempo)[0])
        if np.isnan(tempo_f) or tempo_f < 40 or tempo_f > 220:
            raise ValueError("bad tempo")
        beat_list = [float(b) for b in np.atleast_1d(beats)]
        if len(beat_list) >= 2:
            # layered mixes lure the tracker onto hats (2x tempo) — fold back
            # to a musical grid so chained layers don't get frantic
            while tempo_f > 180 and len(beat_list) >= 4:
                tempo_f /= 2
                beat_list = beat_list[::2]
            return tempo_f, beat_list
    except Exception:
        pass
    # fallback grid: 120 BPM from 0
    dur = y.shape[-1] / sr if y.ndim > 1 else len(mono) / sr
    step = 0.5
    return 120.0, [round(i * step, 4) for i in range(int(dur / step))]


def _detect_key_root(y: np.ndarray, sr: int) -> float:
    """Estimate song key root -> bass fundamental near A1 (55 Hz)."""
    try:
        mono = y[0] if y.ndim > 1 else y
        chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
        pc = int(np.argmax(np.sum(chroma, axis=1)))
        midi = 33 + ((pc - 9) % 12)  # A=9 -> A1=33
        return float(440.0 * 2 ** ((midi - 69) / 12))
    except Exception:
        return 55.0


def _kick_hit(sr: int, amp: float = 0.9) -> np.ndarray:
    n = int(0.14 * sr)
    t = np.arange(n) / sr
    return (amp * np.exp(-t * 28) * np.sin(2 * np.pi * 55 * t)).astype(np.float32)


def _snare_hit(sr: int, amp: float = 0.55) -> np.ndarray:
    n = int(0.16 * sr)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / sr * 30)
    t = np.arange(n) / sr
    tone = 0.4 * np.exp(-t * 25) * np.sin(2 * np.pi * 190 * t)
    return (amp * (noise * 0.6 + tone)).astype(np.float32)


def _hat_hit(sr: int, amp: float = 0.3, dur: float = 0.04) -> np.ndarray:
    n = max(1, int(dur * sr))
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / sr * 120)
    return (amp * noise).astype(np.float32)


def _place(gen: np.ndarray, sr: int, at_sec: float, hit: np.ndarray) -> bool:
    s = int(at_sec * sr)
    if s >= len(gen):
        return False
    e = min(s + len(hit), len(gen))
    if e <= s:
        return False
    gen[s:e] += hit[: e - s]
    return True


def _drum_pattern(
    tempo: float, beats: list[float], dur: float, groove: str, sr: int
) -> tuple[np.ndarray, int]:
    gen = np.zeros(int(dur * sr), dtype=np.float32)
    beat_sec = 60.0 / max(tempo, 40.0)
    # extend grid beyond detected beats so tails get drums
    grid = list(beats)
    t = (grid[-1] + beat_sec) if grid else 0.0
    while t < dur:
        grid.append(round(t, 4))
        t += beat_sec
    kick, snare, hat = _kick_hit(sr), _snare_hit(sr), _hat_hit(sr)
    open_hat = _hat_hit(sr, amp=0.28, dur=0.12)
    hits = 0
    for i, b in enumerate(grid):
        if b >= dur:
            break
        bar_pos = i % 4
        if groove == "four_on_floor":
            hits += _place(gen, sr, b, kick)
            hits += _place(gen, sr, b + beat_sec / 2, open_hat)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.5))
        elif groove == "funky":
            if bar_pos in (0, 2) or (i % 8 == 5):
                hits += _place(gen, sr, b, kick)
            if i % 2 == 1:
                hits += _place(gen, sr, b, _kick_hit(sr, 0.5))
            hits += _place(gen, sr, b + beat_sec / 4, hat)
            hits += _place(gen, sr, b + beat_sec / 2, hat)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, snare)
        elif groove == "swing":
            # shuffled ride: long-short 8ths (2:1 triplet feel), kick light, snare 2 & 4
            hits += _place(gen, sr, b, _kick_hit(sr, 0.55))
            hits += _place(gen, sr, b + beat_sec * 2 / 3, hat)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, snare)
            else:
                hits += _place(gen, sr, b + beat_sec / 3, _hat_hit(sr, 0.18))
        elif groove == "half_time":
            if bar_pos in (0, 2):
                hits += _place(gen, sr, b, kick)
            if bar_pos == 2:
                hits += _place(gen, sr, b, snare)
            hits += _place(gen, sr, b, _hat_hit(sr, 0.2))
        elif groove == "double_time":
            hits += _place(gen, sr, b, _kick_hit(sr, 0.7))
            hits += _place(gen, sr, b + beat_sec / 2, _kick_hit(sr, 0.5))
            hits += _place(gen, sr, b + beat_sec / 4, hat)
            hits += _place(gen, sr, b + beat_sec * 0.75, hat)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, snare)
        else:  # default pop: kick on beats, hats 8ths, snare 2 & 4
            hits += _place(gen, sr, b, kick)
            hits += _place(gen, sr, b + beat_sec / 2, hat)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, snare)
    return gen, hits


def _bass_line(tempo: float, beats: list[float], dur: float, sr: int, groove: str, root: float) -> np.ndarray:
    n = int(dur * sr)
    gen = np.zeros(n, dtype=np.float32)
    beat_sec = 60.0 / max(tempo, 40.0)
    grid = list(beats)
    t = (grid[-1] + beat_sec) if grid else 0.0
    while t < dur:
        grid.append(round(t, 4))
        t += beat_sec
    fifth = root * 1.5
    octave = root * 2
    for i, b in enumerate(grid):
        if b >= dur:
            break
        # root on the beat (quarter-note bass), fifth on offbeat for movement
        for f, at, ln, amp in (
            (root, b, beat_sec * 0.9, 0.32),
            (fifth if groove in ("funky", "default") else root, b + beat_sec / 2, beat_sec * 0.4, 0.2),
        ):
            if at >= dur:
                continue
            s = int(at * sr)
            e = min(s + int(ln * sr), n)
            tt = np.arange(e - s) / sr
            env = np.minimum(1.0, tt * 30) * np.exp(-tt * 4)
            tone = np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * f * 2 * tt)
            gen[s:e] += (amp * tone * env).astype(np.float32)
        if groove == "funky" and i % 4 == 3:  # octave pop
            s = int(b * sr)
            e = min(s + int(beat_sec * 0.3 * sr), n)
            tt = np.arange(e - s) / sr
            gen[s:e] += (0.22 * np.sin(2 * np.pi * octave * tt) * np.exp(-tt * 8)).astype(np.float32)
    return gen


def _add_instrument(
    y: np.ndarray, sr: int, instrument: str, params: dict | None = None
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """Beat-synced instrument layer. Returns (mixed_audio, metadata, layer_mono).

    The layer is normalized hot (0.9 peak) and mixed loud (0.6) with a final
    master normalize, so added drums/bass are clearly audible — not buried.
    """
    params = params or {}
    groove = params.get("groove", "default")
    n = y.shape[1]
    dur = n / sr
    tempo, beats = _detect_beats(y, sr)
    if params.get("target_bpm"):
        try:
            tempo = float(params["target_bpm"])
            step = 60.0 / tempo
            beats = [round(i * step, 4) for i in range(int(dur / step))]
        except Exception:
            pass
    gen = np.zeros(n, dtype=np.float32)
    hits = 0

    if instrument == "drums":
        gen, hits = _drum_pattern(tempo, beats, dur, groove, sr)
    elif instrument == "bass":
        root = _detect_key_root(y, sr)
        gen = _bass_line(tempo, beats, dur, sr, groove, root)
        hits = len(beats) * 2
    elif instrument == "guitar":
        # strummed triad from detected key, 8th-note groove
        root = _detect_key_root(y, sr) * 2
        chord = [root, root * 2 ** (4 / 12), root * 2 ** (7 / 12)]
        beat_sec = 60.0 / tempo
        step = beat_sec / 2
        tt = 0.0
        while tt < dur:
            s = int(tt * sr)
            e = min(s + int(step * 0.9 * sr), n)
            tarr = np.arange(e - s) / sr
            tone = sum(0.12 * np.sin(2 * np.pi * f * tarr) for f in chord)
            env = np.minimum(1.0, tarr * 40) * np.exp(-tarr * 6)
            gen[s:e] += (tone * env).astype(np.float32)
            tt += step
            hits += 1
    elif instrument == "strings":
        # sustained string section: root+fifth+octave, slow bowed attack, vibrato
        t = np.arange(n) / sr
        root = _detect_key_root(y, sr) * 4
        for i, f in enumerate((root, root * 1.5, root * 2)):
            vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t + i)
            gen += 0.11 * np.sin(2 * np.pi * f * vib * t)
        gen *= np.minimum(1.0, t / 1.5)  # 1.5s bowed swell
        gen *= 1 + 0.08 * np.sin(2 * np.pi * (tempo / 60 / 8) * t)
        hits = len(beats)
    elif instrument in ("keys", "other"):
        t = np.arange(n) / sr
        root = _detect_key_root(y, sr) * 4
        freqs = [root, root * 2 ** (4 / 12), root * 2 ** (7 / 12)]
        for f in freqs:
            gen += 0.1 * np.sin(2 * np.pi * f * t)
        gen *= np.minimum(1.0, t * 2)
        gen *= 1 + 0.1 * np.sin(2 * np.pi * (tempo / 60 / 4) * t)  # pulse at bar rate
    else:
        t = np.arange(n) / sr
        gen = 0.15 * np.sin(2 * np.pi * 220.0 * t) + 0.08 * np.sin(2 * np.pi * 330.0 * t)

    peak = float(np.max(np.abs(gen)))
    if peak > 1e-9:
        gen = (gen / peak * 0.9).astype(np.float32)
    else:
        return y, {"tempo_bpm": round(float(tempo), 1), "beat_count": len(beats), "groove": groove, "hits": 0}, None
    layer = gen.copy()
    for ch in range(y.shape[0]):
        y[ch] = y[ch] * 0.8 + gen * 0.6
    master = float(np.max(np.abs(y)))
    if master > 1e-9:
        y = (y / master * 0.9).astype(np.float32)
    meta = {
        "tempo_bpm": round(float(tempo), 1),
        "beat_count": len(beats),
        "groove": groove,
        "hits": hits,
    }
    return y, meta, layer


def _boost_target(y: np.ndarray, sr: int, target: str) -> np.ndarray:
    """Turn up one element in the mix: shelf its home band + master normalize.

    drums -> sub/low punch (<200 Hz) + snap (2-4 kHz)
    bass -> low shelf (<250 Hz)
    anything else -> presence (1-6 kHz)
    """
    out = np.zeros_like(y)
    for ch in range(y.shape[0]):
        S = librosa.stft(y[ch])
        freqs = librosa.fft_frequencies(sr=sr)
        gain = np.ones(len(freqs))
        if target == "drums":
            gain[freqs < 200] = 2.2
            band = (freqs >= 2000) & (freqs <= 4000)
            gain[band] = 1.5
        elif target == "bass":
            gain[freqs < 250] = 2.2
            gain[(freqs >= 250) & (freqs < 500)] = 1.2
        else:
            band = (freqs >= 1000) & (freqs <= 6000)
            gain[band] = 1.7
        out[ch] = librosa.istft(S * gain[:, np.newaxis], length=y.shape[1])
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = (out / peak * 0.9).astype(np.float32)
    return out


def _enhance_vocals(y: np.ndarray, sr: int, params: dict | None = None) -> np.ndarray:
    """Vocal clarity: HP <80 Hz, presence 2-5 kHz, spectral gate denoise."""
    params = params or {}
    out = np.zeros_like(y)
    for ch in range(y.shape[0]):
        S = librosa.stft(y[ch])
        freqs = librosa.fft_frequencies(sr=sr)
        gain = np.ones(len(freqs))
        gain[freqs < 80] = 0.05  # rumble cut
        band = (freqs >= 2000) & (freqs <= 5000)
        gain[band] = 1.5  # presence
        gain[freqs > 12000] = 0.6  # hiss tame
        S_enh = S * gain[:, np.newaxis]
        frame_rms = np.sqrt(np.mean(np.abs(S_enh) ** 2, axis=0) + 1e-12)
        if params.get("denoise", True):
            thresh = np.median(frame_rms) * 0.25
            mask = np.clip((frame_rms - thresh) / (thresh + 1e-9), 0.25, 1.0)
            S_enh = S_enh * mask[np.newaxis, :]
        rec = librosa.istft(S_enh, length=y.shape[1])
        out[ch] = rec
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.9
    return out


def _apply_style(y: np.ndarray, sr: int, style: str, params: dict | None = None) -> tuple[np.ndarray, dict]:
    """Style presets (honest DSP, not generative): drums + EQ movement."""
    params = params or {}
    groove = params.get("groove", "default")
    tempo, beats = _detect_beats(y, sr)
    dur = y.shape[1] / sr
    style_l = style.lower()
    if "house" in style_l or "edm" in style_l:
        groove = "four_on_floor" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, groove, sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.45
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.8 + drums * 0.5, -1, 1)
        # pump: sidechain-ish LFO at beat rate
        t = np.arange(y.shape[1]) / sr
        pump = 1 - 0.25 * (0.5 * (1 + np.sin(2 * np.pi * (tempo / 60) * t - np.pi / 2)))
        y = y * pump[np.newaxis, :]
    elif "tropical" in style_l:
        drums, _ = _drum_pattern(tempo, beats, dur, "funky", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.3
        # offbeat plucks (major triad from key)
        root = _detect_key_root(y, sr) * 4
        pluck = np.zeros(y.shape[1], dtype=np.float32)
        beat_sec = 60.0 / tempo
        for b in beats:
            for off in (0.5,):
                at = b + beat_sec * off
                if at >= dur:
                    continue
                s = int(at * sr)
                e = min(s + int(0.22 * sr), len(pluck))
                tt = np.arange(e - s) / sr
                tone = sum(0.3 * np.sin(2 * np.pi * f * tt) * np.exp(-tt * 10) for f in (root, root * 1.25, root * 1.5))
                pluck[s:e] += tone.astype(np.float32)
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.82 + drums * 0.35 + pluck * 0.3, -1, 1)
        y = _add_reverb(y, sr)
    elif "lofi" in style_l:
        # darken + soft half-time drums
        for ch in range(y.shape[0]):
            S = librosa.stft(y[ch])
            freqs = librosa.fft_frequencies(sr=sr)
            gain = np.ones(len(freqs))
            gain[freqs > 6000] = 0.25
            gain[freqs < 300] = 1.15
            y[ch] = librosa.istft(S * gain[:, np.newaxis], length=y.shape[1])
        drums, _ = _drum_pattern(tempo, beats, dur, "half_time", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.25
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.9 + drums * 0.3, -1, 1)
    else:
        drums, _ = _drum_pattern(tempo, beats, dur, groove, sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.35
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.85 + drums * 0.35, -1, 1)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9
    return y, {"style": style, "tempo_bpm": round(float(tempo), 1), "beat_count": len(beats), "groove": groove}
