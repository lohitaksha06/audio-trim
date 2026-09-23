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
    elif plan.intent == Intent.GAIN:
        db = float(plan.params.get("gain_db", 0.0))
        y = _apply_gain(y, db)
        output_path = _save_wav(y, sr)
        metadata["gain_db"] = round(db, 1)
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
        # voice-first: clarify the vocal before layering (combined request
        # arriving via the LLM path, or "add X and make the voice clear").
        if plan.params.get("also_enhance"):
            y, enh_meta = _enhance_voice(audio_path, y, sr, plan.params)
            metadata.update(enh_meta)
        target = plan.params.get("instrument", "other")
        y, groove_meta, layer = _add_instrument(y, sr, target, plan.params)
        output_path = _save_wav(y, sr)
        metadata["added_instrument"] = target
        if plan.params.get("drum_parts"):
            metadata["drum_parts"] = plan.params["drum_parts"]
        if plan.params.get("wave"):
            metadata["wave"] = plan.params["wave"]
        metadata.update(groove_meta)
        if layer is not None:
            layer_path_out = _save_wav(layer[np.newaxis, :], sr)
            metadata["layer_kind"] = f"{target}_only"
    elif plan.intent == Intent.COMBINE:
        # voice-first: clarify before layering when the combo includes it
        # ("add bass and clean up the voice").
        voice_first = bool(plan.params.get("also_enhance"))
        if voice_first:
            y, enh_meta = _enhance_voice(audio_path, y, sr, plan.params)
            metadata.update(enh_meta)
        instruments = plan.params.get("instruments") or ["drums", "bass"]
        stems = {}
        total_hits = 0
        tempo_seen = 0.0
        beats_seen = 0
        for inst in instruments:
            layer_params = {**plan.params, "instrument": inst}
            if voice_first:
                layer_params["mix_gain"] = 0.35
            y, gm, layer = _add_instrument(y, sr, inst, layer_params)
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
    elif plan.intent == Intent.MIX:
        gains = dict(plan.params.get("gains_db") or {})
        y, mix_meta = _rebalance_mix(audio_path, y, sr, gains)
        output_path = _save_wav(y, sr)
        metadata.update(mix_meta)
    elif plan.intent == Intent.ENHANCE_VOCALS:
        # voice first: isolate/polish/lift the vocal (or denoise the mix),
        # before anything else is layered.
        y, enh_meta = _enhance_voice(audio_path, y, sr, plan.params)
        metadata.update(enh_meta)
        # combined request ("...add drums..."): layer the instrument quietly
        # underneath so the clarified voice stays on top and audible.
        added = plan.params.get("add_instrument")
        if added:
            add_params = dict(plan.params)
            add_params["instrument"] = added
            add_params["groove"] = plan.params.get("add_groove", "default")
            add_params["mix_gain"] = 0.35
            if plan.params.get("add_target_bpm"):
                add_params["target_bpm"] = plan.params["add_target_bpm"]
            if plan.params.get("add_wave"):
                add_params["wave"] = plan.params["add_wave"]
            y, groove_meta, layer = _add_instrument(y, sr, added, add_params)
            output_path = _save_wav(y, sr)
            metadata["added_instrument"] = added
            metadata.update(groove_meta)
            if plan.params.get("drum_parts"):
                metadata["drum_parts"] = plan.params["drum_parts"]
            if layer is not None:
                layer_path_out = _save_wav(layer[np.newaxis, :], sr)
                metadata["layer_kind"] = f"{added}_only"
        else:
            output_path = _save_wav(y, sr)
    elif plan.intent == Intent.STYLE:
        style = plan.params.get("style", "house")
        y, style_meta = _apply_style(y, sr, style, plan.params)
        output_path = _save_wav(y, sr)
        metadata.update(style_meta)
    elif plan.intent == Intent.REVERSE:
        y, rev_meta = _reverse_audio(y, sr, plan.params)
        output_path = _save_wav(y, sr)
        metadata.update(rev_meta)
    elif plan.intent == Intent.REPEAT:
        y, rep_meta = _repeat_section(y, sr, plan.params)
        output_path = _save_wav(y, sr)
        metadata.update(rep_meta)
    elif plan.intent == Intent.TRANSPOSE:
        y, tr_meta = _transpose_audio(y, sr, plan.params)
        output_path = _save_wav(y, sr)
        metadata.update(tr_meta)
    elif plan.intent == Intent.MIX_STEM:
        stem_src = plan.params.get("stem_path")
        if not stem_src:
            raise ValueError("mix_stem needs a stem_path — upload your stem file first")
        import server.services.storage as storage_mod

        stem_file = stem_src
        if not Path(stem_file).is_file():
            try:
                stem_file = storage_mod.resolve(stem_src)
            except Exception:
                pass
        mixed, mix_meta, aligned = mix_imported_stem(audio_path, stem_file, plan.params)
        y = mixed
        output_path = _save_wav(y, sr)
        metadata.update(mix_meta)
        if aligned is not None:
            layer_path_out = _save_wav(aligned[np.newaxis, :], sr)
            metadata["layer_kind"] = "imported stem (BPM-matched)"

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


def _reverse_audio(y: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    """Flip a range (or the whole track) backwards."""
    dur = y.shape[1] / sr
    start = float(params.get("start", 0.0))
    end = params.get("end")
    end = float(end) if end is not None else dur
    start = max(0.0, min(start, dur))
    end = max(start, min(end, dur))
    s, e = int(start * sr), int(end * sr)
    y[:, s:e] = y[:, s:e][:, ::-1]
    return y, {"reversed_start": round(start, 3), "reversed_end": round(end, 3)}


def _repeat_section(y: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    """Loop a section: total ``times`` plays back-to-back (2-8)."""
    dur = y.shape[1] / sr
    start = float(params.get("start", 0.0))
    end = params.get("end")
    end = float(end) if end is not None else dur
    start = max(0.0, min(start, dur))
    end = max(start, min(end, dur))
    times = int(params.get("times", 2))
    times = max(2, min(8, times))
    s, e = int(start * sr), int(end * sr)
    if e <= s:
        return y, {"repeated": False}
    seg = y[:, s:e]
    # tiny crossfade at joints to avoid clicks
    xf = min(int(0.01 * sr), seg.shape[1] // 4)
    out = y[:, :e]
    for _ in range(times - 1):
        if xf > 0 and out.shape[1] >= xf:
            tail = out[:, -xf:].astype(np.float32)
            head = seg[:, :xf].astype(np.float32)
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            blend = tail * (1 - ramp) + head * ramp
            out = np.concatenate([out[:, :-xf], blend, seg[:, xf:]], axis=1)
        else:
            out = np.concatenate([out, seg], axis=1)
    out = np.concatenate([out, y[:, e:]], axis=1)
    return np.clip(out, -1, 1).astype(np.float32), {
        "repeated_start": round(start, 3), "repeated_end": round(end, 3),
        "times": times}


def _transpose_audio(y: np.ndarray, sr: int, params: dict) -> tuple[np.ndarray, dict]:
    """Pitch-shift by semitones without changing tempo (phase vocoder)."""
    semi = max(-12.0, min(12.0, float(params.get("semitones", 2.0))))
    steps = semi
    out = np.zeros_like(y)
    for ch in range(y.shape[0]):
        out[ch] = librosa.effects.pitch_shift(y[ch].astype(np.float32), sr=sr, n_steps=steps)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = (out / peak * 0.9).astype(np.float32)
    return out, {"semitones": round(float(semi), 2)}


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
    try:
        in_dur = max(0.0, float(params.get("fade_in_dur", 2.0)))
    except (TypeError, ValueError):
        in_dur = 2.0
    try:
        out_dur = max(0.0, float(params.get("fade_out_dur", 2.0)))
    except (TypeError, ValueError):
        out_dur = 2.0

    if fade_in and in_dur > 0:
        fade_len = min(int(in_dur * sr), y.shape[1])
        ramp = np.linspace(0, 1, fade_len)
        y[:, :fade_len] *= ramp
    if fade_out and out_dur > 0:
        fade_len = min(int(out_dur * sr), y.shape[1])
        ramp = np.linspace(1, 0, fade_len)
        y[:, -fade_len:] *= ramp
    return y


def _normalize(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9
    return y


def _apply_gain(y: np.ndarray, db: float) -> np.ndarray:
    """Manual volume knob: fixed dB adjustment, then soft-clip to [-1, 1]."""
    db = max(-24.0, min(24.0, db))
    y = y * (10.0 ** (db / 20.0))
    return np.clip(y, -1, 1).astype(np.float32)


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
    """Detect tempo + beat times via the rhythm analyzer (cached per audio).

    The rhythm import stays inside the function so server startup and
    non-rhythm requests pay nothing; the estimator only loads when a user
    actually asks for beat-synced work.
    """
    try:
        res = _rhythm_of(y, sr)
        return float(res["tempo_bpm"]), [float(b) for b in res["beats"]]
    except Exception:
        pass
    # fallback grid: 120 BPM from 0
    mono = y[0] if y.ndim > 1 else y
    dur = len(mono) / sr
    step = 0.5
    return 120.0, [round(i * step, 4) for i in range(int(dur / step))]


_BEAT_CACHE: dict = {}


def _rhythm_of(y: np.ndarray, sr: int) -> dict:
    """Full rhythm read (tempo, beats, auto-groove...), cached by content."""
    from server.ml.rhythm import analyze_rhythm, cache_key

    key = cache_key(y, sr)
    res = _BEAT_CACHE.get(key)
    if res is None:
        a = np.asarray(y)
        dur = float(a.shape[-1] / sr)
        res = analyze_rhythm(a, sr, full_duration=dur)
        if len(_BEAT_CACHE) >= 4:
            _BEAT_CACHE.pop(next(iter(_BEAT_CACHE)))
        _BEAT_CACHE[key] = res
    return res


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


# Aliased base hit generators so _drum_pattern can wrap them (selective parts)
# without Python's conditional-def scoping trap (a conditional def would leave
# the name unbound when the condition is False).
_BASE_KICK_HIT = _kick_hit
_BASE_SNARE_HIT = _snare_hit
_BASE_HAT_HIT = _hat_hit


def _place(gen: np.ndarray, sr: int, at_sec: float, hit: np.ndarray) -> bool:
    if not np.any(hit):
        return False  # silenced drum family (selective parts) — nothing to add
    s = int(at_sec * sr)
    if s >= len(gen):
        return False
    e = min(s + len(hit), len(gen))
    if e <= s:
        return False
    gen[s:e] += hit[: e - s]
    return True


def _crash_hit(sr: int, amp: float = 0.22) -> np.ndarray:
    n = int(1.0 * sr)
    rng = np.random.default_rng(13)
    noise = rng.standard_normal(n).astype(np.float32) * np.exp(-np.arange(n) / sr * 4)
    return (amp * noise).astype(np.float32)


def _drum_pattern(
    tempo: float, beats: list[float], dur: float, groove: str, sr: int,
    parts: list[str] | None = None, swing: float = 0.0,
) -> tuple[np.ndarray, int]:
    gen = np.zeros(int(dur * sr), dtype=np.float32)
    # Selective drum pieces ("add only snare"): wrap the hit generators so the
    # excluded families synthesize silence — every pattern branch below (full
    # kit, trap, house, ...) then automatically plays only the requested
    # pieces. parts=None = full kit. (Unconditional defs: always bound.)
    _want = set(parts) if parts else {"kick", "snare", "hats"}

    def _kick_hit(sr: int, amp: float = 0.9) -> np.ndarray:  # noqa: F811
        if "kick" not in _want:
            return np.zeros(int(0.14 * sr), dtype=np.float32)
        return _BASE_KICK_HIT(sr, amp)

    def _snare_hit(sr: int, amp: float = 0.55) -> np.ndarray:  # noqa: F811
        if "snare" not in _want:
            return np.zeros(int(0.16 * sr), dtype=np.float32)
        return _BASE_SNARE_HIT(sr, amp)

    def _hat_hit(sr: int, amp: float = 0.3, dur: float = 0.04) -> np.ndarray:  # noqa: F811
        if "hats" not in _want:
            return np.zeros(max(1, int(dur * sr)), dtype=np.float32)
        return _BASE_HAT_HIT(sr, amp, dur)
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
        elif groove == "tropical":
            # sunny dembow-lite: kick on 1 & the "and" of 2, rim/snare on 2 & 4, shaker 16ths
            if bar_pos in (0, 2) or (i % 8 == 5):
                hits += _place(gen, sr, b, _kick_hit(sr, 0.7))
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, snare)
            for off in (0.25, 0.5, 0.75):
                hits += _place(gen, sr, b + beat_sec * off, _hat_hit(sr, 0.14, dur=0.03))
        elif groove in ("future", "futuristic", "future_bass"):
            # future-bass: four-on-floor kick + trap-ish hats + clap on 2 & 4
            hits += _place(gen, sr, b, _kick_hit(sr, 0.85))
            hits += _place(gen, sr, b + beat_sec / 3, _hat_hit(sr, 0.16))
            hits += _place(gen, sr, b + beat_sec * 2 / 3, _hat_hit(sr, 0.16))
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, snare)
                hits += _place(gen, sr, b + 0.01, _hat_hit(sr, 0.2, dur=0.08))
        elif groove in ("dubstep", "wobble"):
            # dubstep half-time: kick on 1, snare on 3, sparse hats — heavy, not frantic
            if bar_pos == 0:
                hits += _place(gen, sr, b, _kick_hit(sr, 0.95))
            if bar_pos == 2:
                hits += _place(gen, sr, b, _snare_hit(sr, 0.7))
                hits += _place(gen, sr, b, _kick_hit(sr, 0.4))
            if bar_pos == 3:  # pickup kick rolling into the next bar
                hits += _place(gen, sr, b + beat_sec * 0.75, _kick_hit(sr, 0.5))
            hits += _place(gen, sr, b + beat_sec / 2, _hat_hit(sr, 0.18))
        elif groove in ("big_room", "bigroom", "festival"):
            # big-room EDM: hard four-on-floor + offbeat open hats + clap stack
            hits += _place(gen, sr, b, _kick_hit(sr, 1.0))
            hits += _place(gen, sr, b + beat_sec / 2, open_hat)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.6))
                hits += _place(gen, sr, b + 0.015, _snare_hit(sr, 0.4))
        elif groove in ("house", "deep_house", "tech_house"):
            # house family: warm four-floor, offbeat open hats, soft clap 2 & 4
            vel = 0.7 if groove == "deep_house" else 0.9
            hits += _place(gen, sr, b, _kick_hit(sr, vel))
            hits += _place(gen, sr, b + beat_sec / 2, open_hat)
            if groove == "tech_house":
                hits += _place(gen, sr, b + beat_sec * 0.75, _hat_hit(sr, 0.16))
            else:
                for off in (0.25, 0.75):
                    hits += _place(gen, sr, b + beat_sec * off, _hat_hit(sr, 0.12, dur=0.03))
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.45))
        elif groove == "techno":
            # driving techno: hard kick 4/4, closed 16ths, clap 2 & 4
            hits += _place(gen, sr, b, _kick_hit(sr, 0.95))
            for off in (0.25, 0.5, 0.75):
                hits += _place(gen, sr, b + beat_sec * off, _hat_hit(sr, 0.15, dur=0.03))
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.5))
        elif groove == "trance":
            # uplifting trance: 4-floor kick, offbeat hats, clap stack 2 & 4
            hits += _place(gen, sr, b, _kick_hit(sr, 0.9))
            hits += _place(gen, sr, b + beat_sec / 2, open_hat)
            hits += _place(gen, sr, b + beat_sec * 0.25, _hat_hit(sr, 0.12, dur=0.03))
            hits += _place(gen, sr, b + beat_sec * 0.75, _hat_hit(sr, 0.12, dur=0.03))
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.55))
                hits += _place(gen, sr, b + 0.02, _snare_hit(sr, 0.35))
        elif groove == "trap":
            # trap half-time: syncopated kick, snare on 3, triplet hats
            if bar_pos == 0 or (i % 8 == 6):
                hits += _place(gen, sr, b, _kick_hit(sr, 0.9))
            if bar_pos == 2:
                hits += _place(gen, sr, b, _snare_hit(sr, 0.7))
            for off in (1 / 3, 2 / 3):
                hits += _place(gen, sr, b + beat_sec * off, _hat_hit(sr, 0.15, dur=0.03))
            hits += _place(gen, sr, b + beat_sec / 2, _hat_hit(sr, 0.13, dur=0.03))
        elif groove == "dnb":
            # dnb break: kick + snare interplay on a fast grid, hats 8ths
            if bar_pos in (0, 2) or (i % 8 == 5):
                hits += _place(gen, sr, b, _kick_hit(sr, 0.8))
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.6))
            hits += _place(gen, sr, b + beat_sec / 2, _hat_hit(sr, 0.16))
            hits += _place(gen, sr, b + beat_sec * 0.75, _hat_hit(sr, 0.12, dur=0.03))
        elif groove == "hardstyle":
            # hardstyle: distorted punch kick every beat, clap 2 & 4
            hard = _kick_hit(sr, 1.0) + 0.35 * _snare_hit(sr, 0.5)[: len(_kick_hit(sr))]
            hits += _place(gen, sr, b, hard)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.55))
            hits += _place(gen, sr, b + beat_sec / 2, _hat_hit(sr, 0.13, dur=0.03))
        elif groove == "phonk":
            # drift phonk: swung kick, snare 2 & 4, cowbell motif that moves
            sw = 0.5 + 0.17 * max(0.0, min(1.0, swing)) if swing > 0 else 2 / 3
            if bar_pos in (0, 2) or (i % 8 == 6):
                hits += _place(gen, sr, b, _kick_hit(sr, 0.85))
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, snare)
            hits += _place(gen, sr, b + beat_sec * sw, _hat_hit(sr, 0.16))
            if i % 2 == 0 and "hats" in _want:  # cowbell motif: short high sine + click
                motif = (840.0, 660.0, 990.0, 740.0)
                bell_f = motif[(i // 2) % len(motif)]
                n = int(0.09 * sr)
                tt = np.arange(n) / sr
                bell = (0.3 * np.sin(2 * np.pi * bell_f * tt) * np.exp(-tt * 30)).astype(np.float32)
                hits += _place(gen, sr, b + beat_sec / 2, bell)
        elif groove == "synthwave":
            # retro gated: soft 4-floor, big snare 2 & 4, hats 8ths
            hits += _place(gen, sr, b, _kick_hit(sr, 0.75))
            hits += _place(gen, sr, b + beat_sec / 2, _hat_hit(sr, 0.15))
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.65))
        elif groove == "garage":
            # UKG 2-step: kick on 1, swung skipped snare, shuffling hats.
            # The swing comes from the song when matched, else a 2-step lilt.
            sw = 0.5 + 0.17 * max(0.0, min(1.0, swing)) if swing > 0 else 0.62
            if bar_pos == 0:
                hits += _place(gen, sr, b, _kick_hit(sr, 0.9))
            if bar_pos == 2:
                # syncopated extra kick, the 2-step skip
                hits += _place(gen, sr, b + beat_sec * 0.7, _kick_hit(sr, 0.6))
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.55))
            hits += _place(gen, sr, b + beat_sec * sw, _hat_hit(sr, 0.16))
            hits += _place(gen, sr, b + beat_sec * 0.25, _hat_hit(sr, 0.11, dur=0.03))
            if bar_pos == 3:
                hits += _place(gen, sr, b + beat_sec * sw, _hat_hit(sr, 0.2, dur=0.1))
        elif groove == "amapiano":
            # soft four-floor + driving shaker 16ths + light clap 2 & 4
            hits += _place(gen, sr, b, _kick_hit(sr, 0.7))
            for off in (0.25, 0.5, 0.75):
                hits += _place(gen, sr, b + beat_sec * off, _hat_hit(sr, 0.17, dur=0.03))
            hits += _place(gen, sr, b + beat_sec / 2, open_hat)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.4))
        elif groove == "afro_house":
            # deep four-floor + conga-ish syncopation + shaker bed
            vel = 0.85
            hits += _place(gen, sr, b, _kick_hit(sr, vel))
            hits += _place(gen, sr, b + beat_sec / 2, open_hat)
            # conga slaps: low-mid toms on the "and"s (percussion family)
            if "snare" in _want:
                for off, amp in ((0.25, 0.35), (0.75, 0.4)):
                    n = int(0.12 * sr)
                    tt = np.arange(n) / sr
                    tom = (amp * np.sin(2 * np.pi * 190 * tt) * np.exp(-tt * 22)).astype(np.float32)
                    hits += _place(gen, sr, b + beat_sec * off, tom)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.4))
            hits += _place(gen, sr, b + beat_sec * 0.5, _hat_hit(sr, 0.13, dur=0.03))
        elif groove == "jungle":
            # chopped Amen: busy kick/snare interplay + rolling 16th hats
            if bar_pos in (0, 2) or (i % 8 == 3) or (i % 8 == 6):
                hits += _place(gen, sr, b, _kick_hit(sr, 0.8))
            if bar_pos in (1, 3) or (i % 8 == 7):
                hits += _place(gen, sr, b, _snare_hit(sr, 0.6))
            for off in (0.25, 0.5, 0.75):
                hits += _place(gen, sr, b + beat_sec * off, _hat_hit(sr, 0.15, dur=0.03))
        elif groove == "grime":
            # eski half-time: booming kick on 1, stark snare on 3, icy sparse hats
            if bar_pos == 0:
                hits += _place(gen, sr, b, _kick_hit(sr, 0.95))
            if i % 8 == 5:
                hits += _place(gen, sr, b, _kick_hit(sr, 0.55))
            if bar_pos == 2:
                hits += _place(gen, sr, b, _snare_hit(sr, 0.65))
            hits += _place(gen, sr, b + beat_sec / 2, _hat_hit(sr, 0.14, dur=0.03))
        else:  # default pop: kick on beats, hats 8ths, snare 2 & 4
            hits += _place(gen, sr, b, kick)
            hits += _place(gen, sr, b + beat_sec / 2, hat)
            if bar_pos in (1, 3):
                hits += _place(gen, sr, b, snare)
    # crash on phrase starts (every 4 bars) for the big electronic feels —
    # hats-family, so selective kick/snare-only adds stay clean.
    if groove in ("big_room", "bigroom", "festival", "house", "deep_house",
                  "tech_house", "techno", "trance", "dubstep", "wobble",
                  "phonk", "synthwave", "future", "futuristic", "future_bass",
                  "garage", "amapiano", "afro_house", "jungle", "grime",
                  "default") and "hats" in _want:
        crash = _crash_hit(sr)
        for j in range(0, len(grid), 16):
            if grid[j] < dur:
                hits += _place(gen, sr, grid[j], crash)
    return gen, hits


def _bass_line(tempo: float, beats: list[float], dur: float, sr: int, groove: str, root: float) -> np.ndarray:
    """Plucky bass-guitar line (not a drone sine).

    Why the old one sounded like "eerie straight noise" on voice+piano:
    notes were ~0.9 beats long with a slow exp(-tt*4) decay, so they bled
    into each other as a constant hum — and the 55 Hz fundamental is
    inaudible on small speakers, leaving only beating harmonics.

    Fix: short plucked notes (0.32-beat), fast decay, sharp attack + pick
    transient, string-like harmonics, root up an octave for audibility
    with a quiet sub layer underneath. Sparse pattern: root on beats,
    fifth only as a passing 8th, octave pop only for funky.
    """
    n = int(dur * sr)
    gen = np.zeros(n, dtype=np.float32)
    sub = np.zeros(n, dtype=np.float32)
    beat_sec = 60.0 / max(tempo, 40.0)
    grid = list(beats)
    t = (grid[-1] + beat_sec) if grid else 0.0
    while t < dur:
        grid.append(round(t, 4))
        t += beat_sec
    # Audible pluck one octave up from detected sub root; keep sub quiet.
    pluck_root = root * 2.0
    fifth = pluck_root * 1.5
    octave = pluck_root * 2.0
    for i, b in enumerate(grid):
        if b >= dur:
            break
        bar_pos = i % 4
        # Sparse: beats 0 & 2 always, 1 & 3 softer (leaves room for voice)
        amp = 0.5 if bar_pos in (0, 2) else 0.34
        note_len = beat_sec * 0.38
        for f, at, ln, a in (
            (pluck_root, b, note_len, amp),
            # passing fifth only on funky/default, not every beat
            (fifth, b + beat_sec / 2, beat_sec * 0.22, 0.22)
            if groove in ("funky", "default") and bar_pos in (1, 3)
            else (None, -1, 0, 0),
        ):
            if f is None or at >= dur or at < 0:
                continue
            s = int(at * sr)
            e = min(s + int(ln * sr), n)
            if e <= s:
                continue
            tt = np.arange(e - s) / sr
            # pluck envelope: instant attack, fast exponential decay
            env = np.minimum(1.0, tt * 160) * np.exp(-tt * 9)
            # string harmonics: fundamental + 2nd/3rd/4th decaying
            tone = (
                np.sin(2 * np.pi * f * tt)
                + 0.35 * np.sin(2 * np.pi * f * 2 * tt)
                + 0.18 * np.sin(2 * np.pi * f * 3 * tt)
                + 0.08 * np.sin(2 * np.pi * f * 4 * tt)
            )
            # pick transient: first 8 ms click
            pick_n = min(len(tt), int(0.008 * sr))
            if pick_n > 0:
                tone[:pick_n] += 0.5 * np.sign(np.sin(2 * np.pi * f * tt[:pick_n]))
            # gentle saturation like a bass amp
            note = np.tanh(a * tone * env * 2.0) * 0.6
            gen[s:e] += note.astype(np.float32)
            # sub layer: pure sine at original root, very quiet, same envelope
            ss = int(at * sr)
            ee = min(ss + int(ln * sr), n)
            tts = np.arange(ee - ss) / sr
            sub[ss:ee] += (
                0.12 * np.sin(2 * np.pi * root * tts) * np.minimum(1.0, tts * 120) * np.exp(-tts * 7)
            ).astype(np.float32)
        if groove == "funky" and i % 4 == 3:  # octave pop
            s = int(b * sr)
            e = min(s + int(beat_sec * 0.22 * sr), n)
            tt = np.arange(e - s) / sr
            pop = 0.3 * np.sin(2 * np.pi * octave * tt) * np.minimum(1.0, tt * 200) * np.exp(-tt * 12)
            gen[s:e] += pop.astype(np.float32)
    # mute tail between notes: already short notes + fast decay, just sum
    return (gen + sub).astype(np.float32)


def _saw(freqs: list[float], t: np.ndarray) -> np.ndarray:
    """Cheap supersaw approx: sum of first 6 harmonics, normalized."""
    out = np.zeros_like(t)
    for k in range(1, 7):
        out += (1.0 / k) * sum(np.sin(2 * np.pi * f * k * t) for f in freqs) / max(len(freqs), 1)
    return (out / 2.2).astype(np.float32)


def _osc_wave(freq: float, t: np.ndarray, wave: str) -> np.ndarray:
    """Single-oscillator tone for named synth waves (saw/square/triangle/sine/fm)."""
    ph = 2 * np.pi * freq * t
    if wave == "square":
        return np.sign(np.sin(ph)).astype(np.float32) * 0.6
    if wave == "triangle":
        return (2 / np.pi * np.arcsin(np.sin(ph))).astype(np.float32) * 0.8
    if wave == "fm":
        # 2-op FM: bright bell-ish keys tone
        mod = np.sin(2 * np.pi * freq * 2.0 * t) * 2.0
        return np.sin(ph + mod).astype(np.float32) * 0.7
    if wave == "saw":
        out = np.zeros_like(t)
        for k in range(1, 7):
            out += (1.0 / k) * np.sin(ph * k)
        return (out / 2.2).astype(np.float32)
    # sine / default
    return np.sin(ph).astype(np.float32)


def _square(freqs: list[float], t: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    for f in freqs:
        out += np.sign(np.sin(2 * np.pi * f * t))
    return (out / max(len(freqs), 1) * 0.5).astype(np.float32)


def _synth_layer(kind: str, tempo: float, beats: list[float], dur: float, sr: int, root: float) -> tuple[np.ndarray, int]:
    """Segregated synth/EDM layers. kind in synth/tropical/future/dubstep/edm."""
    n = int(dur * sr)
    gen = np.zeros(n, dtype=np.float32)
    beat_sec = 60.0 / max(tempo, 40.0)
    hits = 0
    # chord tones from detected key (major triad + 9th for air)
    base = root * 4.0
    triad = [base, base * 2 ** (4 / 12), base * 2 ** (7 / 12), base * 2 ** (14 / 12)]

    if kind == "tropical":
        # marimba-like pluck on offbeats — sunny, sparse, leaves room for voice
        for b in beats:
            at = b + beat_sec * 0.5
            if at >= dur:
                continue
            s = int(at * sr)
            e = min(s + int(0.28 * sr), n)
            tt = np.arange(e - s) / sr
            f = triad[(int(b / beat_sec) % len(triad))]
            tone = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2 * tt)
            gen[s:e] += (0.42 * tone * np.exp(-tt * 11)).astype(np.float32)
            hits += 1
    elif kind in ("future", "futuristic", "future_bass"):
        # supersaw chords, one per bar, sidechain-pumped
        t = np.arange(n) / sr
        chords = np.zeros(n, dtype=np.float32)
        for idx, b in enumerate(beats[::4]):
            s = int(b * sr)
            e = min(s + int(beat_sec * 4 * sr), n)
            seg = t[s:e] - (b if e > s else 0)
            # rotate chord inversion per bar
            inv = triad[idx % len(triad):] + triad[: idx % len(triad)]
            chords[s:e] += _saw(inv[:3], seg) * 0.5
        pump = 1 - 0.45 * (0.5 * (1 + np.sin(2 * np.pi * (tempo / 60) * t - np.pi / 2)))
        gen = (chords * pump * 0.55).astype(np.float32)
        hits = len(beats[::4]) * 3
    elif kind in ("dubstep", "wobble"):
        # wobble bass: sub sine + LFO-gated harmonics, half-time (plays on beat 1 & 3)
        for i, b in enumerate(beats):
            if i % 2 == 1:
                continue
            s = int(b * sr)
            e = min(s + int(beat_sec * 1.6 * sr), n)
            if e <= s:
                continue
            tt = np.arange(e - s) / sr
            f = root * 2.0  # audible wobble fundamental
            lfo_rate = 6.0 if tempo < 120 else 8.0  # classic wobble rate
            lfo = 0.5 * (1 + np.sin(2 * np.pi * lfo_rate * tt))
            tone = np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.5 * tt) * lfo
            env = np.minimum(1.0, tt * 60) * np.exp(-tt * 2.2)
            gen[s:e] += (0.55 * tone * (0.35 + 0.65 * lfo) * env).astype(np.float32)
            hits += 1
    elif kind in ("edm", "big_room", "bigroom", "festival"):
        # big-room stab: supersaw hit on every beat + driving sub pulse
        for b in beats:
            s = int(b * sr)
            e = min(s + int(0.3 * sr), n)
            tt = np.arange(e - s) / sr
            stab = _saw(triad[:3], tt) * np.exp(-tt * 8)
            gen[s:e] += (0.5 * stab).astype(np.float32)
            hits += 1
        # sub pulse underneath
        t = np.arange(n) / sr
        gen += (0.12 * np.sin(2 * np.pi * root * 2 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * (tempo / 60) * t))).astype(np.float32)
    elif kind in ("house", "tech_house"):
        # house chords: warm stab offbeat + four-floor sub pulse
        for b in beats:
            at = b + beat_sec * 0.5
            if at >= dur:
                continue
            s = int(at * sr)
            e = min(s + int(0.28 * sr), n)
            tt = np.arange(e - s) / sr
            stab = _saw(triad[:3], tt) * np.exp(-tt * 9)
            gen[s:e] += (0.42 * stab).astype(np.float32)
            hits += 1
        t = np.arange(n) / sr
        gen += (0.1 * np.sin(2 * np.pi * root * 2 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * (tempo / 60) * t))).astype(np.float32)
    elif kind == "deep_house":
        # deep house: Rhodes-ish soft chords, low-slung, one per bar
        t = np.arange(n) / sr
        for idx, b in enumerate(beats[::4]):
            s = int(b * sr)
            e = min(s + int(beat_sec * 3.5 * sr), n)
            seg = t[s:e] - (b if e > s else 0)
            inv = triad[idx % len(triad):] + triad[: idx % len(triad)]
            tone = sum(0.2 * np.sin(2 * np.pi * f * seg) + 0.06 * np.sin(2 * np.pi * f * 2 * seg) for f in inv[:3])
            gen[s:e] += (tone * np.exp(-seg * 1.2)).astype(np.float32)
            hits += 3
    elif kind == "techno":
        # techno: dark rolling sub + minor stab every 2 beats
        for i, b in enumerate(beats):
            if i % 2 == 1:
                continue
            s = int(b * sr)
            e = min(s + int(0.35 * sr), n)
            tt = np.arange(e - s) / sr
            f = root * 2.0
            stab = _saw([f, f * 2 ** (3 / 12)], tt) * np.exp(-tt * 7)
            gen[s:e] += (0.45 * stab).astype(np.float32)
            hits += 1
        t = np.arange(n) / sr
        gen += (0.14 * np.sin(2 * np.pi * root * 2 * t)).astype(np.float32)
    elif kind == "trance":
        # trance: gated supersaw arp 16ths + bar chords
        step = beat_sec / 4
        tt = 0.0
        k = 0
        while tt < dur:
            s = int(tt * sr)
            e = min(s + int(0.16 * sr), n)
            seg = np.arange(e - s) / sr
            f = triad[k % len(triad)] * 2
            gate = 0.5 * (1 + np.sign(np.sin(2 * np.pi * (tempo / 60) * 2 * (tt))))
            gen[s:e] += (0.35 * _saw([f], seg) * np.exp(-seg * 10) * (0.4 + 0.6 * gate)).astype(np.float32)
            tt += step
            k += 1
            hits += 1
    elif kind == "trap":
        # trap: dark bell lead + 808-ish sub glides on downbeats
        for i, b in enumerate(beats):
            if i % 2 == 1:
                continue
            s = int(b * sr)
            e = min(s + int(0.5 * sr), n)
            tt = np.arange(e - s) / sr
            f = triad[i % len(triad)] * 2
            bell = (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 5) + 0.3 * np.sin(2 * np.pi * f * 2.4 * tt) * np.exp(-tt * 8))
            gen[s:e] += (0.4 * bell).astype(np.float32)
            hits += 1
        t = np.arange(n) / sr
        gen += (0.13 * np.sin(2 * np.pi * root * t)).astype(np.float32)
    elif kind == "dnb":
        # dnb: reese-ish detuned sub + sparse atmospheric pad
        for i, b in enumerate(beats):
            if i % 2 == 1:
                continue
            s = int(b * sr)
            e = min(s + int(beat_sec * 1.4 * sr), n)
            tt = np.arange(e - s) / sr
            f = root * 2.0
            tone = np.sin(2 * np.pi * f * tt) + np.sin(2 * np.pi * f * 1.007 * tt)
            gen[s:e] += (0.3 * tone * np.minimum(1.0, tt * 40) * np.exp(-tt * 2.5)).astype(np.float32)
            hits += 1
        t = np.arange(n) / sr
        gen += (_saw(triad[:3], t) * 0.12 * np.minimum(1.0, t / 3.0)).astype(np.float32)
    elif kind == "hardstyle":
        # hardstyle: screech lead stabs on beats + reverse-bass-ish pulse
        for b in beats:
            s = int(b * sr)
            e = min(s + int(0.28 * sr), n)
            tt = np.arange(e - s) / sr
            f = triad[int(b / beat_sec) % len(triad)] * 4
            lead = _saw([f, f * 1.005], tt) * np.exp(-tt * 9)
            gen[s:e] += (0.42 * lead).astype(np.float32)
            hits += 1
        t = np.arange(n) / sr
        gen += (0.12 * np.sin(2 * np.pi * root * 2 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * (tempo / 60) * t))).astype(np.float32)
    elif kind == "phonk":
        # phonk: cowbell-ish lead pattern + Memphis sub
        for i, b in enumerate(beats):
            if i % 2 == 0:
                s = int(b * sr)
                e = min(s + int(0.22 * sr), n)
                tt = np.arange(e - s) / sr
                f = triad[i % len(triad)] * 4
                bell = (0.35 * np.sin(2 * np.pi * f * tt) + 0.2 * np.sin(2 * np.pi * f * 1.48 * tt)) * np.exp(-tt * 14)
                gen[s:e] += bell.astype(np.float32)
                hits += 1
        t = np.arange(n) / sr
        gen += (0.13 * np.sin(2 * np.pi * root * t)).astype(np.float32)
    elif kind == "synthwave":
        # synthwave: retro square/saw pad + gated arp
        t = np.arange(n) / sr
        pad = _square(triad[:3], t) * 0.2 * np.minimum(1.0, t / 2.0)
        gen += pad
        step = beat_sec / 2
        tt = 0.0
        k = 0
        while tt < dur:
            s = int(tt * sr)
            e = min(s + int(0.2 * sr), n)
            seg = np.arange(e - s) / sr
            f = triad[k % len(triad)] * 2
            gen[s:e] += (0.28 * np.sin(2 * np.pi * f * seg) * np.exp(-seg * 9)).astype(np.float32)
            tt += step
            k += 1
            hits += 1
    elif kind in ("supersaw", "square_lead", "saw", "square", "triangle", "sine", "fm", "wavetable", "hoover"):
        # named oscillator waves: lead line following the beat grid
        wave = {"supersaw": "saw", "square_lead": "square", "hoover": "saw"}.get(kind, kind if kind in ("saw", "square", "triangle", "sine", "fm") else "saw")
        for i, b in enumerate(beats):
            s = int(b * sr)
            e = min(s + int(beat_sec * 0.85 * sr), n)
            if e <= s:
                continue
            tt = np.arange(e - s) / sr
            f = triad[i % len(triad)] * (2 if kind in ("supersaw", "square_lead", "hoover") else 1)
            if kind in ("supersaw", "hoover"):
                tone = _saw([f, f * 1.006, f * 0.994], tt)
            elif kind == "wavetable":
                morph = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * tt))
                tone = (1 - morph) * _osc_wave(f, tt, "saw") + morph * _osc_wave(f, tt, "square")
            else:
                tone = _osc_wave(f, tt, wave)
            env = np.minimum(1.0, tt * 80) * np.exp(-tt * 3.0)
            gen[s:e] += (0.45 * tone * env).astype(np.float32)
            hits += 1
    elif kind in ("pluck", "marimba"):
        for b in beats:
            at = b + beat_sec * 0.5
            if at >= dur:
                continue
            s = int(at * sr)
            e = min(s + int(0.26 * sr), n)
            tt = np.arange(e - s) / sr
            f = triad[int(b / beat_sec) % len(triad)] * 2
            tone = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2 * tt)
            gen[s:e] += (0.42 * tone * np.exp(-tt * 11)).astype(np.float32)
            hits += 1
    elif kind in ("pad", "choir"):
        t = np.arange(n) / sr
        if kind == "choir":
            # soft vocal-ish pad: stacked sines with slow vibrato + swell
            for i, f in enumerate(triad[:3]):
                vib = 1 + 0.005 * np.sin(2 * np.pi * 5.0 * t + i)
                gen += 0.14 * np.sin(2 * np.pi * f * vib * t)
        else:
            gen += _saw(triad[:3], t) * 0.3
        gen *= np.minimum(1.0, t / 2.0)
        hits = len(beats)
    elif kind == "arp":
        step = beat_sec / 4
        tt = 0.0
        k = 0
        while tt < dur:
            s = int(tt * sr)
            e = min(s + int(0.14 * sr), n)
            seg = np.arange(e - s) / sr
            f = triad[k % len(triad)] * 4
            gen[s:e] += (0.3 * np.sin(2 * np.pi * f * seg) * np.exp(-seg * 12)).astype(np.float32)
            tt += step
            k += 1
            hits += 1
    elif kind in ("808", "acid", "reese", "fm_bass", "trap_bass"):
        # sub/bass synth family: pattern on beats with family timbre
        for i, b in enumerate(beats):
            if kind in ("acid",) and i % 2 == 1:
                continue
            s = int(b * sr)
            ln = beat_sec * (0.9 if kind == "808" else 0.5)
            e = min(s + int(ln * sr), n)
            if e <= s:
                continue
            tt = np.arange(e - s) / sr
            f0 = root * (1.0 if kind in ("808", "reese") else 2.0)
            if kind == "808":
                # sliding 808: glide up into the note, long decay
                glide = f0 * (0.85 + 0.15 * np.minimum(1.0, tt * 20))
                phase = 2 * np.pi * np.cumsum(glide) / sr
                tone = np.sin(phase) * np.exp(-tt * 2.5)
                gen[s:e] += (0.55 * tone).astype(np.float32)
            elif kind == "acid":
                # squelchy: saw + resonant-ish 3rd harmonic wobble
                lfo = 0.5 * (1 + np.sin(2 * np.pi * 8.0 * tt))
                tone = _osc_wave(f0, tt, "saw") + 0.5 * np.sin(2 * np.pi * f0 * 3 * tt) * lfo
                gen[s:e] += (0.4 * tone * np.minimum(1.0, tt * 60) * np.exp(-tt * 4)).astype(np.float32)
            elif kind == "reese":
                tone = np.sin(2 * np.pi * f0 * tt) + np.sin(2 * np.pi * f0 * 1.008 * tt)
                gen[s:e] += (0.32 * tone * np.minimum(1.0, tt * 30) * np.exp(-tt * 2.8)).astype(np.float32)
            else:  # fm_bass / trap_bass: FM sub thump
                tone = _osc_wave(f0, tt, "fm")
                gen[s:e] += (0.45 * tone * np.minimum(1.0, tt * 60) * np.exp(-tt * 4)).astype(np.float32)
            hits += 1
    elif kind in ("piano", "organ", "lofi_keys"):
        t = np.arange(n) / sr
        for idx, b in enumerate(beats[::2]):
            s = int(b * sr)
            e = min(s + int(beat_sec * 1.8 * sr), n)
            seg = t[s:e] - (b if e > s else 0)
            inv = triad[idx % len(triad):] + triad[: idx % len(triad)]
            if kind == "organ":
                tone = sum(0.16 * np.sin(2 * np.pi * f * seg) + 0.05 * np.sin(2 * np.pi * f * 2 * seg) for f in inv[:3])
            elif kind == "lofi_keys":
                tone = sum(0.2 * np.sin(2 * np.pi * f * seg) * np.exp(-seg * 2.5) for f in inv[:3])
            else:
                tone = sum(0.22 * np.sin(2 * np.pi * f * seg) * np.exp(-seg * 3.5) for f in inv[:3])
            gen[s:e] += tone.astype(np.float32)
            hits += 3
    elif kind in ("brass", "trumpet", "sax", "flute", "violin", "cello", "harp"):
        # orchestral / acoustic leads: legato line with vibrato, family voicing
        step_beats = 2 if kind in ("brass", "cello") else 1
        for i, b in enumerate(beats[::step_beats]):
            s = int(b * sr)
            e = min(s + int(beat_sec * step_beats * 0.95 * sr), n)
            if e <= s:
                continue
            seg = np.arange(e - s) / sr
            mult = {"brass": 2.0, "trumpet": 4.0, "sax": 2.0, "flute": 4.0, "violin": 4.0, "cello": 1.0, "harp": 4.0}[kind]
            f = triad[i % len(triad)] * mult
            if kind == "harp":
                # glissando arp instead of sustained line
                for k2, ff in enumerate(triad):
                    ss = s + int(k2 * 0.06 * sr)
                    ee = min(ss + int(0.3 * sr), n)
                    if ss >= n or ee <= ss:
                        continue
                    t2 = np.arange(ee - ss) / sr
                    gen[ss:ee] += (0.3 * np.sin(2 * np.pi * ff * 2 * t2) * np.exp(-t2 * 8)).astype(np.float32)
                    hits += 1
                continue
            vib = 1 + 0.006 * np.sin(2 * np.pi * 5.5 * seg)
            if kind in ("brass", "trumpet"):
                tone = _saw([f], seg) * 0.5 + 0.2 * np.sin(2 * np.pi * f * vib * seg)
                env = np.minimum(1.0, seg * 25)
            elif kind == "sax":
                tone = np.sin(2 * np.pi * f * vib * seg) + 0.3 * np.sin(2 * np.pi * f * 2 * seg)
                env = np.minimum(1.0, seg * 12) * np.exp(-seg * 0.8)
            else:
                tone = np.sin(2 * np.pi * f * vib * seg) + 0.2 * np.sin(2 * np.pi * f * 2 * seg)
                env = np.minimum(1.0, seg * 8) * np.exp(-seg * 0.6)
            gen[s:e] += (0.4 * tone * env).astype(np.float32)
            hits += 1
    elif kind == "garage":
        # warpy 2-step: deep sub pulse on 1 + sparse detuned pluck stabs
        for i, b in enumerate(beats):
            if i % 2 == 0:
                s = int(b * sr)
                e = min(s + int(beat_sec * 0.8 * sr), n)
                tt = np.arange(e - s) / sr
                f = root * (1.0 if i % 4 == 0 else 1.5)
                tone = np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * f * 1.005 * tt)
                gen[s:e] += (0.4 * tone * np.minimum(1.0, tt * 50) * np.exp(-tt * 3)).astype(np.float32)
                hits += 1
            if i % 4 == 3:
                s = int((b + beat_sec * 0.7) * sr)
                e = min(s + int(0.2 * sr), n)
                if e > s:
                    seg = np.arange(e - s) / sr
                    f = triad[i % len(triad)] * 2
                    gen[s:e] += (0.3 * _saw([f, f * 1.008], seg) * np.exp(-seg * 10)).astype(np.float32)
                    hits += 1
    elif kind == "amapiano":
        # log drum: hollow percussive bass melody + soft shaker bed
        for i, b in enumerate(beats):
            if i % 2 == 0:
                s = int(b * sr)
                e = min(s + int(beat_sec * 0.9 * sr), n)
                if e > s:
                    tt = np.arange(e - s) / sr
                    f = root * (1.0, 1.189, 1.335, 1.498)[(i // 2) % 4]
                    glide = f * (0.8 + 0.2 * np.minimum(1.0, tt * 25))
                    phase = 2 * np.pi * np.cumsum(glide) / sr
                    tone = np.sin(phase) + 0.35 * np.sin(phase * 2.0)
                    gen[s:e] += (0.5 * tone * np.minimum(1.0, tt * 90) * np.exp(-tt * 4)).astype(np.float32)
                    hits += 1
        t = np.arange(n) / sr
        step16 = beat_sec / 4
        tt = 0.0
        while tt < dur:
            s = int(tt * sr)
            e = min(s + int(0.06 * sr), n)
            seg = np.arange(e - s) / sr
            rng = np.random.default_rng(int(tt * 100) % (2 ** 31))
            gen[s:e] += (0.1 * rng.standard_normal(e - s) * np.exp(-seg * 90)).astype(np.float32)
            tt += step16
    elif kind == "afro_house":
        # deep percussion bed: conga pattern + warm chord chops
        for i, b in enumerate(beats):
            for off, fmult in ((0.25, 1.0), (0.5, 1.335), (0.75, 1.0)):
                s = int((b + beat_sec * off) * sr)
                e = min(s + int(0.18 * sr), n)
                if e > s:
                    seg = np.arange(e - s) / sr
                    f = root * 2.0 * fmult
                    gen[s:e] += (0.32 * np.sin(2 * np.pi * f * seg) * np.exp(-seg * 14)).astype(np.float32)
                    hits += 1
            if i % 2 == 1:
                s = int((b + beat_sec * 0.5) * sr)
                e = min(s + int(0.24 * sr), n)
                if e > s:
                    seg = np.arange(e - s) / sr
                    gen[s:e] += (0.3 * _saw(triad[:3], seg) * np.exp(-seg * 9)).astype(np.float32)
                    hits += 1
    elif kind == "jungle":
        # chopped break bass: rolling sub + ragga-style skank chops
        for i, b in enumerate(beats):
            if i % 2 == 0:
                s = int(b * sr)
                e = min(s + int(beat_sec * 0.7 * sr), n)
                tt = np.arange(e - s) / sr
                f = root
                tone = np.sin(2 * np.pi * f * tt) + np.sin(2 * np.pi * f * 1.007 * tt)
                gen[s:e] += (0.32 * tone * np.minimum(1.0, tt * 40) * np.exp(-tt * 3)).astype(np.float32)
                hits += 1
            if i % 4 in (1, 3):
                s = int((b + beat_sec * 0.5) * sr)
                e = min(s + int(0.16 * sr), n)
                if e > s:
                    seg = np.arange(e - s) / sr
                    f = triad[i % len(triad)] * 2
                    gen[s:e] += (0.3 * _square([f], seg) * np.exp(-seg * 11)).astype(np.float32)
                    hits += 1
    elif kind == "grime":
        # eski: harsh square blips + deep sub drops on the 1
        for i, b in enumerate(beats):
            if i % 4 == 0:
                s = int(b * sr)
                e = min(s + int(beat_sec * 1.2 * sr), n)
                tt = np.arange(e - s) / sr
                gen[s:e] += (0.4 * np.sin(2 * np.pi * root * tt) * np.minimum(1.0, tt * 40) * np.exp(-tt * 2.5)).astype(np.float32)
                hits += 1
            if i % 2 == 0:
                s = int((b + beat_sec * 0.5) * sr)
                e = min(s + int(0.14 * sr), n)
                if e > s:
                    seg = np.arange(e - s) / sr
                    f = triad[(i * 3) % len(triad)] * 4
                    gen[s:e] += (0.28 * np.sign(np.sin(2 * np.pi * f * seg)) * np.exp(-seg * 12)).astype(np.float32)
                    hits += 1
    else:  # generic warm synth: pad + gentle arp
        t = np.arange(n) / sr
        pad = _saw(triad[:3], t) * 0.28 * np.minimum(1.0, t / 2.0)
        arp = np.zeros(n, dtype=np.float32)
        step = beat_sec / 2
        tt = 0.0
        k = 0
        while tt < dur:
            s = int(tt * sr)
            e = min(s + int(0.22 * sr), n)
            seg = np.arange(e - s) / sr
            f = triad[k % len(triad)] * 2
            arp[s:e] += (0.3 * np.sin(2 * np.pi * f * seg) * np.exp(-seg * 9)).astype(np.float32)
            tt += step
            k += 1
            hits += 1
        gen = (pad + arp * 0.8).astype(np.float32)
    return gen, hits


def _add_instrument(
    y: np.ndarray, sr: int, instrument: str, params: dict | None = None
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """Beat-synced instrument layer. Returns (mixed_audio, metadata, layer_mono).

    The layer is normalized hot (0.9 peak) and mixed loud (mix_gain, default
    0.6) with a final master normalize, so added drums/bass are clearly
    audible — not buried. params["drum_parts"] (subset of kick/snare/hats)
    synthesizes only those pieces; None = full kit.
    """
    params = params or {}
    groove = params.get("groove", "default")
    n = y.shape[1]
    dur = n / sr
    tempo, beats = _detect_beats(y, sr)
    # "add drums" with no style named: listen to the song and match its feel
    # (house/phonk/dubstep/...) instead of adding blindly. Prompt-named
    # grooves always win; this only fills the default.
    auto_meta: dict = {}
    swing = 0.0
    if instrument == "drums" and groove == "default":
        try:
            res = _rhythm_of(y, sr)
            auto = res.get("auto_groove", "default")
            if auto and auto != "default":
                groove = auto
            swing = float(res.get("swing", 0.0))
            auto_meta = {"auto_groove": groove,
                         "groove_confidence": res.get("groove_confidence"),
                         "groove_source": res.get("groove_source")}
        except Exception:
            pass
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
        gen, hits = _drum_pattern(tempo, beats, dur, groove, sr,
                                  parts=params.get("drum_parts"), swing=swing)
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
    elif instrument in ("keys", "other", "piano", "organ", "choir", "pad", "arp",
                          "pluck", "marimba", "harp", "brass", "trumpet", "sax",
                          "flute", "violin", "cello", "supersaw", "square_lead",
                          "saw", "square", "triangle", "sine", "fm", "wavetable",
                          "hoover", "808", "acid", "reese", "fm_bass", "trap_bass",
                          "lofi_keys"):
        # named waves / acoustic / sub families — each gets its own timbre
        root = _detect_key_root(y, sr)
        gen, hits = _synth_layer(instrument, tempo, beats, dur, sr, root)
    elif instrument in ("synth", "tropical", "future", "futuristic", "future_bass",
                         "dubstep", "wobble", "edm", "big_room", "bigroom", "festival",
                         "house", "deep_house", "tech_house", "techno", "trance",
                         "trap", "dnb", "hardstyle", "phonk", "synthwave",
                         "garage", "amapiano", "afro_house", "jungle", "grime"):
        # segregated synth / EDM kits — each gets its own timbre (multi-add ready)
        kind = instrument
        if kind in ("futuristic", "future_bass"):
            kind = "future"
        if kind in ("wobble",):
            kind = "dubstep"
        if kind in ("big_room", "bigroom", "festival"):
            kind = "edm"
        if kind == "tech_house":
            kind = "house"
        root = _detect_key_root(y, sr)
        wave = (params.get("wave") or "").lower()
        # a named wave ("add an acid bassline") wins over the kit default
        if wave in ("acid", "reese", "808", "pluck", "supersaw", "square", "arp", "pad"):
            kind = {"supersaw": "supersaw", "square": "square_lead"}.get(wave, wave)
        gen, hits = _synth_layer(kind, tempo, beats, dur, sr, root)
        # auto groove so drums layered next to it match: tropical->tropical etc.
        if groove == "default":
            groove = {"tropical": "tropical", "future": "future",
                      "dubstep": "dubstep", "edm": "big_room",
                      "house": "house", "deep_house": "deep_house",
                      "tech_house": "house", "techno": "techno",
                      "trance": "trance", "trap": "trap", "dnb": "dnb",
                      "hardstyle": "hardstyle", "phonk": "phonk",
                      "synthwave": "synthwave", "garage": "garage",
                      "amapiano": "amapiano", "afro_house": "afro_house",
                      "jungle": "jungle", "grime": "grime"}.get(kind, groove)
    else:
        t = np.arange(n) / sr
        gen = 0.15 * np.sin(2 * np.pi * 220.0 * t) + 0.08 * np.sin(2 * np.pi * 330.0 * t)

    peak = float(np.max(np.abs(gen)))
    if peak > 1e-9:
        gen = (gen / peak * 0.9).astype(np.float32)
    else:
        return y, {"tempo_bpm": round(float(tempo), 1), "beat_count": len(beats), "groove": groove, "hits": 0}, None
    layer = gen.copy()
    # mix_gain lets voice-first chains layer drums quietly under the vocal
    # instead of burying it (default hot mix keeps standalone adds punchy).
    try:
        layer_gain = float(params.get("mix_gain", 0.6))
    except (TypeError, ValueError):
        layer_gain = 0.6
    layer_gain = min(1.0, max(0.05, layer_gain))
    for ch in range(y.shape[0]):
        y[ch] = y[ch] * 0.8 + gen * layer_gain
    master = float(np.max(np.abs(y)))
    if master > 1e-9:
        y = (y / master * 0.9).astype(np.float32)
    meta = {
        "tempo_bpm": round(float(tempo), 1),
        "beat_count": len(beats),
        "groove": groove,
        "hits": hits,
    }
    meta.update(auto_meta)
    return y, meta, layer


def mix_imported_stem(
    audio_path: str, stem_path: str, params: dict | None = None
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """Mix a user-imported stem with the original song.

    The AI listens to BOTH files: detects each BPM + beat grid, time-stretches
    the stem to the song's tempo, beat-aligns downbeat to downbeat, then mixes.
    Returns (mixed_audio[ch, n], metadata, aligned_stem_mono).
    """
    params = params or {}
    stem_level = float(params.get("stem_level", 0.6))
    stem_level = min(1.0, max(0.05, stem_level))

    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    ys, sr_s = librosa.load(stem_path, sr=sr, mono=True)

    orig_tempo, orig_beats = _detect_beats(y, sr)
    stem_mono2d = ys[np.newaxis, :]
    stem_tempo, stem_beats = _detect_beats(stem_mono2d, sr)

    # 1. tempo match: stretch stem -> original tempo (skip if within 3%)
    stretch = 1.0
    if stem_tempo > 40 and orig_tempo > 40:
        ratio = stem_tempo / orig_tempo
        if abs(ratio - 1.0) > 0.03 and 0.5 <= ratio <= 2.0:
            ys = librosa.effects.time_stretch(ys, rate=float(ratio))
            stretch = float(ratio)
            # scale detected stem beats into stretched time
            stem_beats = [b / ratio for b in stem_beats]
            stem_tempo = orig_tempo

    # 2. beat-align: shift stem so downbeats coincide
    offset_sec = 0.0
    if orig_beats and stem_beats:
        offset_sec = float(orig_beats[0]) - float(stem_beats[0])
    elif orig_beats:
        offset_sec = float(orig_beats[0])

    n = y.shape[1]
    aligned = np.zeros(n, dtype=np.float32)
    off_samp = int(round(offset_sec * sr))
    if off_samp >= 0:
        src_end = min(len(ys), n - off_samp)
        if src_end > 0:
            aligned[off_samp: off_samp + src_end] = ys[:src_end]
    else:
        src_start = min(-off_samp, len(ys))
        dst_len = min(len(ys) - src_start, n)
        if dst_len > 0:
            aligned[:dst_len] = ys[src_start: src_start + dst_len]

    # 3. level-match stem to song (avoid one burying the other), then mix
    peak_song = float(np.max(np.abs(y))) + 1e-9
    peak_stem = float(np.max(np.abs(aligned))) + 1e-9
    aligned = (aligned / peak_stem * 0.9 * stem_level).astype(np.float32)
    mixed = y.copy()
    for ch in range(mixed.shape[0]):
        mixed[ch] = mixed[ch] * 0.85 + aligned * 0.65
    master = float(np.max(np.abs(mixed))) + 1e-9
    mixed = (mixed / master * 0.9).astype(np.float32)

    meta = {
        "song_bpm": round(float(orig_tempo), 1),
        "stem_bpm": round(float(stem_tempo), 1),
        "tempo_bpm": round(float(orig_tempo), 1),
        "stretch_factor": round(float(stretch), 3),
        "beat_offset_sec": round(float(offset_sec), 3),
        "stem_level": round(float(stem_level), 2),
        "beat_count": len(orig_beats),
    }
    return mixed, meta, aligned


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
        elif target in ("bass", "808"):
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


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (max(-12.0, min(12.0, db)) / 20.0))


def _mix_eq_curve(freqs: np.ndarray, gains_db: dict) -> np.ndarray:
    """Multi-band balance curve: each stem owns frequency home(s); series mix.

    vocals -> presence 1-6 kHz (+ air 6-12 kHz at half strength)
    drums  -> punch <200 Hz + snap 2-4 kHz
    bass   -> lows <250 Hz (+ low-mids 250-500 at half strength)
    other  -> broadband tilt (everything else)
    """
    curve = np.ones(len(freqs))
    gv = _db_to_lin(float(gains_db.get("vocals", 0.0)))
    gd = _db_to_lin(float(gains_db.get("drums", 0.0)))
    gb = _db_to_lin(float(gains_db.get("bass", 0.0)))
    go = _db_to_lin(float(gains_db.get("other", 0.0)))
    if gv != 1.0:
        curve[(freqs >= 1000) & (freqs <= 6000)] *= gv
        curve[(freqs > 6000) & (freqs <= 12000)] *= 1 + (gv - 1) * 0.5
    if gd != 1.0:
        curve[freqs < 200] *= gd * gd  # punch gets extra weight
        curve[(freqs >= 2000) & (freqs <= 4000)] *= 1 + (gd - 1) * 0.7
    if gb != 1.0:
        curve[freqs < 250] *= gb
        curve[(freqs >= 250) & (freqs < 500)] *= 1 + (gb - 1) * 0.5
    if go != 1.0:
        curve *= go
    return curve


def _rebalance_mix(
    audio_path: str, y: np.ndarray, sr: int, gains_db: dict
) -> tuple[np.ndarray, dict]:
    """Prioritize sounds in the mix: {stem: dB} over vocals/drums/bass/other.

    True stem remix when Demucs separation works; otherwise an honest
    multi-band EQ-balance approximation (documented in metadata.method).
    """
    gains_db = {k: max(-12.0, min(12.0, float(v))) for k, v in (gains_db or {}).items()}
    if not gains_db:
        gains_db = {"drums": 6.0, "vocals": -3.0}

    # 1. true stem remix — surgical when the model loads
    try:
        separator = SourceSeparator()
        stems = separator.separate(audio_path)
        if stems and set(stems) >= {"vocals", "drums", "bass", "other"}:
            loaded: dict[str, np.ndarray] = {}
            n = y.shape[1]
            for name, path in stems.items():
                s, _ = librosa.load(path, sr=sr, mono=True)
                if len(s) < n:
                    s = np.pad(s, (0, n - len(s)))
                loaded[name] = s[:n].astype(np.float32) * _db_to_lin(gains_db.get(name, 0.0))
            remix = sum(loaded.values())
            mix = np.stack([remix] * y.shape[0]).astype(np.float32)
            peak = float(np.max(np.abs(mix)))
            if peak > 1e-9:
                mix = (mix / peak * 0.9).astype(np.float32)
            return mix, {"method": "stem_remix", "gains_db": gains_db,
                         "note": "Separated with Demucs, per-stem faders applied, remixed."}
    except Exception as e:
        stem_error = str(e)[:160]
    else:
        stem_error = "stems incomplete"

    # 2. EQ-balance fallback — fast, deterministic, no model needed
    out = np.zeros_like(y)
    for ch in range(y.shape[0]):
        S = librosa.stft(y[ch])
        freqs = librosa.fft_frequencies(sr=sr)
        out[ch] = librosa.istft(S * _mix_eq_curve(freqs, gains_db)[:, np.newaxis], length=y.shape[1])
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = (out / peak * 0.9).astype(np.float32)
    return out, {"method": "eq_balance", "gains_db": gains_db,
                 "note": f"EQ-balance approximation ({stem_error}); separate into stems for surgical moves."}


def _suppress_disturbances(
    y: np.ndarray, sr: int, aggressive: bool = False
) -> tuple[np.ndarray, int]:
    """Tame loud non-stationary intrusions spectral subtraction misses.

    Quiet-profile subtraction only learns *stationary* background (hiss/hum):
    a car horn, drill burst or door slam is louder than the voice, so it is
    never in the quiet frames and survives. This stage hunts two signatures:

    1. broadband bursts (drill/clatter/slam): frame energy spikes vs a ~1 s
       rolling median *with* elevated high-frequency content;
    2. tonal intrusions (horn/siren: a narrow peak 200-2500 Hz that persists
       0.25-2 s while surrounding frames differ).

    Burst frames get a smooth broadband dip; tonal bins get a selective notch.
    Returns (audio, n_events). Gentle by design — voice harmonics are
    broadband-stable and rarely trigger either detector.
    """
    out = np.zeros_like(y)
    n_events = 0
    spike_k = 2.5 if aggressive else 3.5
    dip = 0.12 if aggressive else 0.25
    for ch in range(y.shape[0]):
        S = librosa.stft(y[ch])
        mag = np.abs(S) + 1e-12
        freqs = librosa.fft_frequencies(sr=sr)
        n_frames = mag.shape[1]
        if n_frames < 8:
            out[ch] = y[ch]
            continue
        frame_e = np.mean(mag, axis=0)
        hf = np.sum(mag[freqs >= 4000], axis=0) / (np.sum(mag, axis=0) + 1e-12)
        # rolling median baseline (~1 s window) so slow fades don't trigger
        win = max(9, min(n_frames, int(sr / 512)))
        pad = win // 2
        padded = np.pad(frame_e, (pad, pad), mode="edge")
        roll_med = np.median(
            np.lib.stride_tricks.sliding_window_view(padded, win), axis=1
        ) + 1e-12
        hf_med = max(float(np.median(hf)), 0.01)
        hf_abs = hf  # broadband loud events carry real HF energy; voice doesn't
        # NOTE: no pure-energy bypass — speech onsets after a pause also
        # spike energy but carry no HF, and must never be gated. Tonal
        # events are handled by the notch path below, not here.
        burst = (frame_e > spike_k * roll_med) & (hf > 1.8 * hf_med)
        # drill-type events: only modestly louder than music, but HF-rich
        burst |= (frame_e > 1.4 * roll_med) & (hf_abs > 0.12) & (hf > 5.0 * hf_med)
        # smooth attack/release: spread by 2 frames so gating doesn't click
        burst_sm = burst.copy()
        for _ in range(2):
            burst_sm[1:] |= burst_sm[:-1]
            burst_sm[:-1] |= burst_sm[1:]
        gain = np.ones(n_frames, dtype=np.float32)
        # tonal intrusion notch (horn/siren): narrow peak 200-2500 Hz that
        # stands far above its ±15-frame neighborhood and persists ±3 frames.
        # Independent of the burst gate — a horn over a pause has no HF
        # content and may not spike broadband energy, but its peak-to-neighbor
        # ratio is huge. Voice vowels rarely trigger: their harmonics lift
        # the whole band median, keeping the ratio below threshold.
        band = (freqs >= 200) & (freqs <= 2500)
        band_idx = np.where(band)[0]
        tonal_hit = np.zeros(n_frames, dtype=bool)
        if len(band_idx) and n_frames >= 31:
            notch = np.ones_like(mag)
            pk_ratio = 8.0
            band_freqs = freqs[band_idx]
            for f in range(15, n_frames - 15):
                col = mag[band_idx, f].copy()
                neigh = float(np.median(mag[band_idx, f - 15:f + 16]))
                persist = float(np.min([
                    np.max(mag[band_idx, max(0, f + d)]) for d in (-3, 3)
                ]))
                # a horn/siren can stack several tones (400+500 Hz): notch
                # every persistent peak, not just the tallest — unless it
                # carries voice-like harmonic support (energy at 2x/3x),
                # which vetoes the notch so sung vowels survive.
                for _ in range(3):
                    pk = int(np.argmax(col))
                    peak_v = float(col[pk])
                    if not (peak_v > pk_ratio * neigh and persist > 5.0 * neigh):
                        break
                    f0 = float(band_freqs[pk])
                    harmonic = False
                    for mult in (2, 3):
                        if f0 * mult > 2500.0:
                            break
                        j = int(np.argmin(np.abs(band_freqs - f0 * mult)))
                        if float(col[j]) > 0.15 * peak_v:
                            harmonic = True
                            break
                    if harmonic:
                        col[pk - 2:pk + 3] = 0.0
                        continue
                    tonal_hit[f] = True
                    lo = max(band_idx[0], band_idx[pk] - 2)
                    hi = min(band_idx[-1], band_idx[pk] + 2)
                    notch[lo:hi + 1, max(0, f - 1):f + 2] = 0.08 if aggressive else 0.15
                    col[pk - 2:pk + 3] = 0.0
            mag = mag * notch
        hit = burst_sm | tonal_hit
        gain = np.where(burst_sm, dip, 1.0).astype(np.float32)
        # tonal-only frames get a milder dip on top of the notch
        gain = np.where(tonal_hit & ~burst_sm, 0.6 if aggressive else 0.7, gain)
        # count contiguous hit regions as events
        idx = np.where(hit)[0]
        if len(idx):
            n_events = max(n_events, int(1 + np.sum(np.diff(idx) > 4)))
        S_clean = mag * (S / (np.abs(S) + 1e-12)) * gain[np.newaxis, :]
        out[ch] = librosa.istft(S_clean.astype(np.complex64), length=y.shape[1])
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = (out / peak * 0.9).astype(np.float32)
    return out, n_events


def _compress_vocal(v: np.ndarray, sr: int, aggressive: bool = False) -> np.ndarray:
    """Gentle upward compression so quiet words stay audible.

    Per-frame RMS vs global median: lift quiet frames (up to +6 dB),
    lightly tame loud peaks. Smoothed so there is no pumping. This is what
    makes "make me louder" work on uneven speech, where a flat +dB lift
    would just clip the loud parts.
    """
    v = np.asarray(v, dtype=np.float32)
    out = np.zeros_like(v)
    hop = 512
    ceil = 2.0 if aggressive else 1.6
    for ch in range(v.shape[0]):
        rms = librosa.feature.rms(y=v[ch], hop_length=hop)[0] + 1e-12
        med = float(np.median(rms))
        raw = np.clip(med / rms, 0.7, ceil)
        ker = np.ones(5, dtype=np.float32) / 5
        sm = np.convolve(raw, ker, mode="same")
        # expand frame gains to samples
        g = np.repeat(sm, hop)[: v.shape[1]]
        if len(g) < v.shape[1]:
            g = np.pad(g, (0, v.shape[1] - len(g)), mode="edge")
        out[ch] = (v[ch] * g).astype(np.float32)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = (out / peak * 0.9).astype(np.float32)
    return out


def _spectral_subtract(y: np.ndarray, sr: int, strength: float = 1.0,
                       floor: float = 0.08) -> np.ndarray:
    """Per-file noise removal: learn the noise profile from the quietest 10%
    of frames, subtract it, keep the phase. ``strength`` 1.0 = normal,
    1.5 = aggressive ("remove ALL background noise")."""
    out = np.zeros_like(y)
    for ch in range(y.shape[0]):
        S = librosa.stft(y[ch])
        mag = np.abs(S) + 1e-12
        phase = S / mag
        frame_e = np.mean(mag, axis=0)
        med_e = float(np.median(frame_e))
        # true noise floor needs real pauses/breaths: frames well below median.
        # Continuous signal (no pauses) -> go gentle so tone isn't eaten.
        genuine_quiet = frame_e[frame_e < 0.5 * med_e]
        eff_strength = strength
        if len(genuine_quiet) >= 4:
            prof_frames = mag[:, frame_e < 0.5 * med_e]
        else:
            q = max(4, int(len(frame_e) * 0.10))
            prof_frames = mag[:, np.argpartition(frame_e, q)[:q]]
            eff_strength = strength * 0.25
        profile = np.mean(prof_frames, axis=1, keepdims=True)
        clean = mag - eff_strength * profile
        clean = np.maximum(clean, floor * mag)
        # temporal smoothing of the mask tames "musical noise" warble
        mask = clean / mag
        if mask.shape[1] >= 3:
            ker = np.ones(3, dtype=np.float32) / 3
            mask = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, mask)
            clean = mask * mag
        out[ch] = librosa.istft((clean * phase).astype(np.complex64),
                                length=y.shape[1])
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = (out / peak * 0.9).astype(np.float32)
    return out


def _denoise_mix(y: np.ndarray, sr: int, aggressive: bool = False) -> tuple[np.ndarray, int]:
    """Whole-mix background-noise removal (no voice named in the request).

    Two stages: transient/disturbance suppression first (horns, drills —
    too loud for profile subtraction), then learned-profile subtraction
    for stationary hiss/hum. Returns (audio, n_disturbances).
    """
    y, n_events = _suppress_disturbances(y, sr, aggressive)
    y = _spectral_subtract(y, sr, strength=1.5 if aggressive else 1.2,
                           floor=0.03 if aggressive else 0.08)
    return y, n_events


def _polish_vocal(v: np.ndarray, sr: int, aggressive: bool = False) -> tuple[np.ndarray, int]:
    """EQ + denoise for an isolated vocal stem: rumble cut, mud cut,
    presence + air lift, de-ess, hiss tame, learned-profile subtraction.

    Disturbance suppression runs on the stem too — horns/sirens often leak
    into the Demucs vocals stem, and boosting the stem without taming them
    would make them louder. Ends with upward compression so quiet words
    stay audible. Returns (audio, n_disturbances).
    """
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v[np.newaxis, :]
    # denoise first so the EQ lifts voice, not hiss
    v = _spectral_subtract(v, sr, strength=1.5 if aggressive else 1.0,
                           floor=0.05 if aggressive else 0.1)
    v, n_events = _suppress_disturbances(v, sr, aggressive)
    out = np.zeros_like(v)
    for ch in range(v.shape[0]):
        S = librosa.stft(v[ch])
        freqs = librosa.fft_frequencies(sr=sr)
        gain = np.ones(len(freqs))
        rumble = 100.0 if aggressive else 80.0
        gain[freqs < rumble] = 0.05
        gain[(freqs >= 250) & (freqs <= 450)] = 0.8  # mud cut
        gain[(freqs >= 2000) & (freqs <= 5000)] = 2.0 if aggressive else 1.7
        gain[(freqs > 5500) & (freqs <= 8000)] = 0.85  # de-ess
        gain[(freqs >= 10000) & (freqs <= 14000)] = 1.25  # air
        gain[freqs > 14000] = 0.5
        out[ch] = librosa.istft(S * gain[:, np.newaxis], length=v.shape[1])
    out = _compress_vocal(out, sr, aggressive)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = (out / peak * 0.9).astype(np.float32)
    return out, n_events


def _enhance_voice(audio_path: str, y: np.ndarray, sr: int,
                   params: dict | None = None) -> tuple[np.ndarray, dict]:
    """Voice-first enhancement. Returns (audio, metadata).

    - ``denoise_mix`` (no voice named): learned-profile subtraction on the mix.
    - otherwise: isolate the vocal stem (Demucs), polish + lift it, remix over
      the accompaniment. Falls back to mix EQ when separation is unavailable.
    """
    params = params or {}
    aggressive = bool(params.get("aggressive"))
    meta: dict = {"aggressive": aggressive}

    if params.get("denoise_mix"):
        y, n_events = _denoise_mix(y, sr, aggressive)
        meta["method"] = "mix_denoise"
        meta["enhanced"] = "mix"
        meta["disturbances_removed"] = n_events
        return y, meta

    vocal_db = 4.0 + (2.0 if aggressive else 0.0) + (2.0 if params.get("level_boost") else 0.0)
    try:
        separator = SourceSeparator()
        stems = separator.separate(audio_path)
        if "vocals" not in stems:
            raise RuntimeError("no vocals stem")
        n = y.shape[1]
        v, _ = librosa.load(stems["vocals"], sr=sr, mono=False)
        if v.ndim == 1:
            v = v[np.newaxis, :]
        v = v[:, :n] if v.shape[1] >= n else np.pad(v, ((0, 0), (0, n - v.shape[1])))
        acc = np.zeros_like(y)
        for name, path in stems.items():
            if name == "vocals":
                continue
            s, _ = librosa.load(path, sr=sr, mono=True)
            if len(s) < n:
                s = np.pad(s, (0, n - len(s)))
            acc += s[:n].astype(np.float32)  # broadcasts over channels
        v, n_events = _polish_vocal(v, sr, aggressive)
        if v.shape[0] == 1 and y.shape[0] > 1:
            v = np.stack([v[0]] * y.shape[0]).astype(np.float32)
        mix = acc + v * _db_to_lin(vocal_db)
        peak = float(np.max(np.abs(mix)))
        if peak > 1e-9:
            mix = (mix / peak * 0.9).astype(np.float32)
        meta["method"] = "stem_remix"
        meta["enhanced"] = "vocals"
        meta["vocal_boost_db"] = round(vocal_db, 1)
        meta["disturbances_removed"] = n_events
        return mix, meta
    except Exception as e:
        meta["stem_error"] = str(e)[:160]
        # fallback without Demucs: tame bursts first (else the presence lift
        # amplifies horns/drills), then EQ, then compress + lift for audibility.
        y, n_events = _suppress_disturbances(y, sr, aggressive)
        meta["disturbances_removed"] = n_events
        y = _enhance_vocals(y, sr, params)
        y = _compress_vocal(y, sr, aggressive)
        if params.get("level_boost"):
            y = _apply_gain(y, 2.0)
            peak = float(np.max(np.abs(y)))
            if peak > 1e-9:
                y = (y / peak * 0.9).astype(np.float32)
        meta["method"] = "eq_fallback"
        meta["enhanced"] = "vocals"
        return y, meta


def _enhance_vocals(y: np.ndarray, sr: int, params: dict | None = None) -> np.ndarray:
    """Vocal clarity: HP <80 Hz, presence 2-5 kHz, spectral gate denoise.

    Tuned voice-first: a stronger presence lift (+4.6 dB) and a lower gate
    floor so background noise/hiss drops away while the lead vocal (male or
    female fundamental + harmonics) stays intact and audible.
    """
    params = params or {}
    out = np.zeros_like(y)
    for ch in range(y.shape[0]):
        S = librosa.stft(y[ch])
        freqs = librosa.fft_frequencies(sr=sr)
        gain = np.ones(len(freqs))
        gain[freqs < 80] = 0.05  # rumble cut
        band = (freqs >= 2000) & (freqs <= 5000)
        gain[band] = 1.7  # presence: consonants + vocal edge
        gain[freqs > 12000] = 0.5  # hiss tame
        S_enh = S * gain[:, np.newaxis]
        frame_rms = np.sqrt(np.mean(np.abs(S_enh) ** 2, axis=0) + 1e-12)
        if params.get("denoise", True):
            thresh = np.median(frame_rms) * 0.25
            mask = np.clip((frame_rms - thresh) / (thresh + 1e-9), 0.15, 1.0)
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
    if "dubstep" in style_l or "wobble" in style_l:
        groove = "dubstep"
        drums, _ = _drum_pattern(tempo, beats, dur, "dubstep", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.45
        root = _detect_key_root(y, sr)
        wob, _ = _synth_layer("dubstep", tempo, beats, dur, sr, root)
        wob = wob / (np.max(np.abs(wob)) + 1e-9) * 0.5
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.78 + drums * 0.5 + wob * 0.55, -1, 1)
    elif "futur" in style_l or "future_bass" in style_l:
        groove = "future" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "future", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.4
        root = _detect_key_root(y, sr)
        saw, _ = _synth_layer("future", tempo, beats, dur, sr, root)
        saw = saw / (np.max(np.abs(saw)) + 1e-9) * 0.45
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.8 + drums * 0.45 + saw * 0.5, -1, 1)
        t = np.arange(y.shape[1]) / sr
        pump = 1 - 0.25 * (0.5 * (1 + np.sin(2 * np.pi * (tempo / 60) * t - np.pi / 2)))
        y = y * pump[np.newaxis, :]
    elif "big_room" in style_l or "bigroom" in style_l or "festival" in style_l or (
        "edm" in style_l and "tropical" not in style_l
    ):
        groove = "big_room" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "big_room", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.5
        root = _detect_key_root(y, sr)
        stab, _ = _synth_layer("edm", tempo, beats, dur, sr, root)
        stab = stab / (np.max(np.abs(stab)) + 1e-9) * 0.4
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.78 + drums * 0.55 + stab * 0.45, -1, 1)
        t = np.arange(y.shape[1]) / sr
        pump = 1 - 0.25 * (0.5 * (1 + np.sin(2 * np.pi * (tempo / 60) * t - np.pi / 2)))
        y = y * pump[np.newaxis, :]
    elif "deep_house" in style_l or "deep-house" in style_l:
        # before the bare-"house" branch: "deep_house" contains "house"
        groove = "deep_house" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "deep_house", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.35
        root = _detect_key_root(y, sr)
        rhodes, _ = _synth_layer("deep_house", tempo, beats, dur, sr, root)
        rhodes = rhodes / (np.max(np.abs(rhodes)) + 1e-9) * 0.4
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.82 + drums * 0.4 + rhodes * 0.45, -1, 1)
        t = np.arange(y.shape[1]) / sr
        pump = 1 - 0.2 * (0.5 * (1 + np.sin(2 * np.pi * (tempo / 60) * t - np.pi / 2)))
        y = y * pump[np.newaxis, :]
    elif "house" in style_l:
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
        drums, _ = _drum_pattern(tempo, beats, dur, "tropical", sr)
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
    elif "techno" in style_l:
        groove = "techno" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "techno", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.5
        root = _detect_key_root(y, sr)
        roll, _ = _synth_layer("techno", tempo, beats, dur, sr, root)
        roll = roll / (np.max(np.abs(roll)) + 1e-9) * 0.4
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.78 + drums * 0.55 + roll * 0.45, -1, 1)
    elif "trance" in style_l or "psytrance" in style_l:
        groove = "trance" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "trance", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.45
        root = _detect_key_root(y, sr)
        gate, _ = _synth_layer("trance", tempo, beats, dur, sr, root)
        gate = gate / (np.max(np.abs(gate)) + 1e-9) * 0.45
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.78 + drums * 0.5 + gate * 0.5, -1, 1)
        y = _add_reverb(y, sr)
    elif "trap" in style_l:
        groove = "trap" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "trap", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.45
        root = _detect_key_root(y, sr)
        bell, _ = _synth_layer("trap", tempo, beats, dur, sr, root)
        bell = bell / (np.max(np.abs(bell)) + 1e-9) * 0.45
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.8 + drums * 0.5 + bell * 0.5, -1, 1)
    elif "dnb" in style_l or "drum_and_bass" in style_l:
        groove = "dnb" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "dnb", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.5
        root = _detect_key_root(y, sr)
        reese, _ = _synth_layer("dnb", tempo, beats, dur, sr, root)
        reese = reese / (np.max(np.abs(reese)) + 1e-9) * 0.45
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.78 + drums * 0.55 + reese * 0.5, -1, 1)
    elif "jungle" in style_l or "amen break" in style_l or style_l.strip() == "amen":
        groove = "jungle" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "jungle", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.5
        root = _detect_key_root(y, sr)
        chop, _ = _synth_layer("jungle", tempo, beats, dur, sr, root)
        chop = chop / (np.max(np.abs(chop)) + 1e-9) * 0.45
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.78 + drums * 0.55 + chop * 0.5, -1, 1)
    elif "grime" in style_l or "eski" in style_l:
        groove = "grime" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "grime", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.45
        root = _detect_key_root(y, sr)
        eski, _ = _synth_layer("grime", tempo, beats, dur, sr, root)
        eski = eski / (np.max(np.abs(eski)) + 1e-9) * 0.4
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.8 + drums * 0.5 + eski * 0.45, -1, 1)
    elif "garage" in style_l or "2-step" in style_l or "2step" in style_l:
        groove = "garage" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "garage", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.4
        root = _detect_key_root(y, sr)
        warp, _ = _synth_layer("garage", tempo, beats, dur, sr, root)
        warp = warp / (np.max(np.abs(warp)) + 1e-9) * 0.45
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.8 + drums * 0.45 + warp * 0.5, -1, 1)
    elif "amapiano" in style_l or "log drum" in style_l or "logdrum" in style_l:
        groove = "amapiano" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "amapiano", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.35
        root = _detect_key_root(y, sr)
        logd, _ = _synth_layer("amapiano", tempo, beats, dur, sr, root)
        logd = logd / (np.max(np.abs(logd)) + 1e-9) * 0.5
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.8 + drums * 0.4 + logd * 0.55, -1, 1)
    elif "afro house" in style_l or "afrohouse" in style_l or "afro-house" in style_l:
        groove = "afro_house" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "afro_house", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.4
        root = _detect_key_root(y, sr)
        perc, _ = _synth_layer("afro_house", tempo, beats, dur, sr, root)
        perc = perc / (np.max(np.abs(perc)) + 1e-9) * 0.45
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.8 + drums * 0.45 + perc * 0.5, -1, 1)
        y = _add_reverb(y, sr)
    elif "hardstyle" in style_l or "rawstyle" in style_l:
        groove = "hardstyle" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "hardstyle", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.55
        root = _detect_key_root(y, sr)
        screech, _ = _synth_layer("hardstyle", tempo, beats, dur, sr, root)
        screech = screech / (np.max(np.abs(screech)) + 1e-9) * 0.4
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.76 + drums * 0.6 + screech * 0.45, -1, 1)
    elif "phonk" in style_l:
        groove = "phonk" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "phonk", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.45
        root = _detect_key_root(y, sr)
        cow, _ = _synth_layer("phonk", tempo, beats, dur, sr, root)
        cow = cow / (np.max(np.abs(cow)) + 1e-9) * 0.4
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.8 + drums * 0.5 + cow * 0.45, -1, 1)
    elif "synthwave" in style_l or "retrowave" in style_l or "outrun" in style_l:
        groove = "synthwave" if groove == "default" else groove
        drums, _ = _drum_pattern(tempo, beats, dur, "synthwave", sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.4
        root = _detect_key_root(y, sr)
        retro, _ = _synth_layer("synthwave", tempo, beats, dur, sr, root)
        retro = retro / (np.max(np.abs(retro)) + 1e-9) * 0.45
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.8 + drums * 0.45 + retro * 0.5, -1, 1)
        y = _add_reverb(y, sr)
    else:
        drums, _ = _drum_pattern(tempo, beats, dur, groove, sr)
        drums = drums / (np.max(np.abs(drums)) + 1e-9) * 0.35
        for ch in range(y.shape[0]):
            y[ch] = np.clip(y[ch] * 0.85 + drums * 0.35, -1, 1)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9
    return y, {"style": style, "tempo_bpm": round(float(tempo), 1), "beat_count": len(beats), "groove": groove}
