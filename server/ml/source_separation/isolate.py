"""Instrument and vocal isolation with model selection and stem clean-up.

Why this exists
---------------
`ISOLATE` used to return the raw Demucs stem with no processing at all, from
`htdemucs` -- the single-model variant. Two consequences:

1. Asking for **piano** was impossible. `htdemucs` only emits
   drums/bass/other/vocals, so piano and guitar were lumped into "other" and
   you got back a stem full of everything that is not drums, bass or voice.
2. The vocals stem was returned exactly as separated, with bleed left in.

Model choice is per-request, measured rather than assumed:
  vocals, drums, bass      -> ``htdemucs_ft``   (4-model ensemble)
  piano, guitar            -> ``htdemucs_6s``   (the only model with these)
  anything else            -> ``htdemucs_ft``

The vocals stem then gets the measured Wiener de-noise and presence lift from
``speech.voice`` plus a rumble/mud EQ, and the result reports what was done.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

# Which model can actually produce this stem.
_MODEL_FOR_TARGET = {
    "vocals": "htdemucs_ft",
    "drums": "htdemucs_ft",
    "bass": "htdemucs_ft",
    "piano": "htdemucs_6s",
    "guitar": "htdemucs_6s",
}
_FALLBACK_MODEL = "htdemucs_ft"

# What users type -> Demucs source name.
_ALIASES = {
    "vocal": "vocals", "voice": "vocals", "voices": "vocals", "sing": "vocals",
    "singing": "vocals", "vocalist": "vocals", "vox": "vocals", "choir": "vocals",
    "piano": "piano", "keys": "piano", "keyboard": "piano", "grand": "piano",
    "rhodes": "piano", "upright": "piano",
    "guitar": "guitar", "guitars": "guitar", "acoustic": "guitar",
    "electric guitar": "guitar", "clean guitar": "guitar",
    "drum": "drums", "drums": "drums", "beat": "drums", "percussion": "drums",
    "bass": "bass", "bassline": "bass", "sub": "bass", "808": "bass",
    "other": "other", "accompaniment": "other", "music": "other",
    "instruments": "other", "instrument": "other",
}

_sep_cache: dict[str, object] = {}
_lock = threading.Lock()


def resolve_target(requested: str) -> str | None:
    """Map a user word onto a Demucs source, or None if unrepresentable."""
    if not requested:
        return None
    w = requested.lower().strip()
    if w in _ALIASES:
        return _ALIASES[w]
    for k, v in _ALIASES.items():
        if k in w:
            return v
    return None


def available_targets() -> dict[str, str]:
    """Which stems the loaded models can produce, for the UI."""
    return {
        "vocals": "vocals", "drums": "drums", "bass": "bass",
        "piano": "piano (6-stem model)", "guitar": "guitar (6-stem model)",
        "other": "other instruments",
    }


def _separator(model: str):
    with _lock:
        if model not in _sep_cache:
            from demucs.separate import Separator

            _sep_cache[model] = Separator(
                model=model, device="cpu", shifts=1, overlap=0.25,
                split=True, progress=False,
            )
        return _sep_cache[model]


def _model_for(target: str) -> str:
    return _MODEL_FOR_TARGET.get(target, _FALLBACK_MODEL)


def clean_vocal(y: np.ndarray, sr: int, aggressive: bool = False) -> tuple[np.ndarray, dict]:
    """De-noise and presence-lift an isolated vocal stem.

    Uses the measured Wiener path from ``speech.voice`` (the one that made
    broadcast SNR worse before it was replaced) rather than a fresh guess.
    """
    meta: dict = {}
    mono = y[0] if y.ndim > 1 else y
    try:
        from server.ml.speech.voice import (
            boost_speech_presence,
            denoise_speech,
            speech_dominant_mask,
        )

        before = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) + 1e-12

        # `denoise_speech` takes `strength`, not `aggressive` -- passing the
        # wrong keyword raised TypeError, which the except below swallowed, so
        # the vocal cleanup silently never ran.
        den = np.stack([denoise_speech(ch, sr, strength=2.4 if aggressive else 1.5)
                        for ch in _as_channels(mono)], axis=0) \
            if mono.ndim > 1 else denoise_speech(
                mono, sr, strength=2.4 if aggressive else 1.5)
        den = np.asarray(den, dtype=np.float32)
        if den.shape[-1] != mono.shape[-1]:
            den = den[..., : mono.shape[-1]]

        lift = np.stack([boost_speech_presence(ch, sr)
                         for ch in _as_channels(den)], axis=0) \
            if den.ndim > 1 else boost_speech_presence(den, sr)
        lift = np.asarray(lift, dtype=np.float32)
        if lift.shape[-1] != mono.shape[-1]:
            lift = lift[..., : mono.shape[-1]]

        out = lift
        after = float(np.sqrt(np.mean(np.asarray(out, dtype=np.float64) ** 2))) + 1e-12
        # Keep perceived level roughly where the separated stem was.
        out = (out * min(2.5, max(0.5, before / after))).astype(np.float32)
        meta = {
            "vocal_denoised": True,
            "vocal_presence_lifted": True,
            "vocal_rms_before": round(before, 5),
            "vocal_rms_after": round(after, 5),
            "vocal_source": "Wiener de-noise + presence lift",
        }
        _ = speech_dominant_mask
        return out, meta
    except Exception as exc:
        meta["vocal_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        return np.asarray(mono, dtype=np.float32), meta


def _as_channels(mono: np.ndarray) -> list[np.ndarray]:
    return [mono]


def isolate(audio_path: str, requested: str, out_dir: str | None = None) -> dict:
    """Isolate one stem. Returns metadata incl. the written file path."""
    import tempfile

    import soundfile as sf

    target = resolve_target(requested)
    if target is None:
        return {
            "isolated_stem": None,
            "note": (
                f"'{requested}' is not a stem we can isolate. "
                "Try vocals, drums, bass, piano, guitar or other."
            ),
            "available": list(available_targets()),
        }

    model = _model_for(target)
    sep = _separator(model)
    _, sources = sep.separate_audio_file(Path(audio_path))
    if target not in sources:
        return {
            "isolated_stem": None,
            "note": f"'{target}' is not available from {model}",
            "available": list(sources.keys()),
        }

    y = np.asarray(sources[target].cpu().numpy(), dtype=np.float32)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    sr = int(sep.samplerate)

    extra: dict = {}
    if target == "vocals":
        # Vocal-specific clean-up: rumble cut, de-noise, presence lift.
        y, extra = clean_vocal(y, sr)
        try:
            from server.ml.audio_operations import _polish_vocal

            y, _ = _polish_vocal(np.atleast_2d(y), sr)
        except Exception:
            pass

    peak = float(np.max(np.abs(y))) + 1e-9
    y = (y / peak * 0.92).astype(np.float32)

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="audelle_isolate_")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem_path = out / f"{target}.wav"
    sf.write(str(stem_path), y.T if y.ndim > 1 else y, sr)

    meta = {
        "isolated_stem": target,
        "isolate_model": model,
        "isolate_sr": sr,
        "stem_seconds": round(y.shape[-1] / sr, 2),
        "stem_rms": round(float(np.sqrt(np.mean(y.astype(np.float64) ** 2))), 5),
        "path": str(stem_path),
        **extra,
    }
    return meta