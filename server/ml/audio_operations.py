import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import librosa

from server.ml.prompt_engine import PromptPlan, Intent
from server.ml.source_separation.separator import SourceSeparator
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
        y = _remove_section(y, sr, plan.params)
        output_path = _save_wav(y, sr)
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
        y[0] = librosa.istft(S)
    elif mood == "bright":
        freqs = librosa.fft_frequencies(sr=sr)
        high_boost = np.ones(len(freqs))
        high_boost[freqs > 3000] = 1.4
        high_boost[freqs < 200] = 0.8
        S = librosa.stft(y[0])
        S = S * high_boost[:, np.newaxis]
        y[0] = librosa.istft(S)
    elif mood == "energetic":
        y = _normalize(y) * 1.1
        y = np.clip(y, -1, 1)
    return y
