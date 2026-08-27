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
        y = _add_instrument(y, sr, target)
        output_path = _save_wav(y, sr)
        metadata["added_instrument"] = target

    result: dict[str, Any] = {"intent": plan.intent.value, "params": plan.params}

    if output_path:
        result["output_path"] = output_path
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


def _add_instrument(y: np.ndarray, sr: int, instrument: str) -> np.ndarray:
    """Synthesize a simple instrument layer and mix it in.

    This is a heuristic placeholder — not a generative model — but it gives
    audible feedback for 'add bass / synth / drums' prompts and keeps the
    pipeline end-to-end testable. Replace with MusicGen/Riffusion later.
    """
    n = y.shape[1]
    t = np.arange(n) / sr
    # detect a rough tempo for rhythmic instruments: use 120 BPM if unknown
    # so drums land on the beat and are audible in tests
    gen = np.zeros(n, dtype=np.float32)

    if instrument == "bass":
        # low sine + octave, half-note pulse
        f0 = 55.0  # A1
        pulse = 0.5 * (1 + np.sin(2 * np.pi * 1.0 * t))  # 1 Hz sway
        gen = 0.35 * np.sin(2 * np.pi * f0 * t) * pulse + 0.15 * np.sin(2 * np.pi * f0 * 2 * t) * pulse
    elif instrument == "drums":
        # click every 0.5s (120 BPM) as a short decaying burst
        interval = int(0.5 * sr)
        for start in range(0, n, interval):
            end = min(start + int(0.04 * sr), n)
            click_t = np.arange(end - start) / sr
            burst = np.exp(-click_t * 80) * np.sin(2 * np.pi * 80 * click_t)
            # add a hi frequency tick
            burst += 0.5 * np.exp(-click_t * 120) * np.sin(2 * np.pi * 3000 * click_t) * (click_t < 0.01)
            gen[start:end] += burst * 0.6
    elif instrument == "guitar":
        # arpeggiated 440Hz + harmonics
        f0 = 110.0
        gen = 0.2 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * f0 * 2 * t) + 0.06 * np.sin(2 * np.pi * f0 * 3 * t)
        gen *= 0.5 * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    elif instrument in ("keys", "other"):
        # warm pad: C3 major chord (C4, E4, G4) with slow attack
        freqs = [261.63, 329.63, 391.99] if instrument == "keys" else [220.0, 277.18, 329.63]
        for f in freqs:
            gen += 0.12 * np.sin(2 * np.pi * f * t)
        gen *= np.minimum(1.0, t * 2)  # fade in
        # gentle tremolo
        gen *= 1 + 0.1 * np.sin(2 * np.pi * 0.8 * t)
    else:
        # fallback pad
        gen = 0.15 * np.sin(2 * np.pi * 220.0 * t) + 0.08 * np.sin(2 * np.pi * 330.0 * t)

    # mix into each channel, keep headroom
    for ch in range(y.shape[0]):
        y[ch] = np.clip(y[ch] * 0.85 + gen * 0.35, -1, 1)
    return y
