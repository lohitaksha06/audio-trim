"""Drum sequencing from real recorded one-shots.

What is real
------------
Every hit here is a recorded sample from the ``airasoul/drum-kit`` dataset
(1200 one-shots across 10 drum types), extracted by
``scripts/build_drum_bank.py``. The previous implementation synthesised kicks
from a pitch-swept sine and snares from bandpassed noise, which is why the
"add drums" feature sounded synthetic and was not worth shipping.

What is rule-based
------------------
The *arrangement* -- which hits land on which 16th-note step -- is a hand-
written pattern library, not a learned model. It is tempo-synced to a measured
BPM. No trained drum model is claimed here.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

BANK_DIR = Path(__file__).resolve().parents[1] / "models" / "drum_bank"

_lock = threading.Lock()
_index: dict | None = None

SR = 44100


class DrumBankUnavailable(RuntimeError):
    pass


def _load_index() -> dict:
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is not None:
            return _index
        path = BANK_DIR / "bank.json"
        if not path.exists():
            raise DrumBankUnavailable(
                f"drum bank missing at {path}. Build it with:\n"
                "  python -m scripts.build_drum_bank"
            )
        _index = json.loads(path.read_text(encoding="utf-8"))
        return _index


def available() -> bool:
    return (BANK_DIR / "bank.json").exists()


def kit() -> dict[str, int]:
    """How many real hits are available per drum type."""
    try:
        return dict(_load_index()["labels"])
    except DrumBankUnavailable:
        return {}


# --------------------------------------------------------------- patterns
# Each pattern is a list of 16 step slots per instrument. `None` = rest,
# a float = velocity. Steps are sixteenth notes.

PATTERNS: dict[str, dict[str, list[float | None]]] = {
    "house": {
        "kick": [1.0, None, None, None, 1.0, None, None, None, 1.0, None, None, 0.6, None, None, None, None],
        "clap": [None, None, None, None, 1.0, None, None, None, None, None, None, None, 1.0, None, None, None],
        "hat": [None, 0.5, None, 0.5, None, 0.5, None, 0.7, None, 0.5, None, 0.5, None, 0.5, None, 0.8],
        "open_hat": [None] * 14 + [0.7, None],
    },
    "techno": {
        "kick": [1.0, None, None, 0.9, None, None, 1.0, None, None, None, 0.9, None, None, None, None, None],
        "hat": [0.5, 0.3, 0.6, 0.3, 0.5, 0.3, 0.6, 0.3, 0.5, 0.3, 0.6, 0.3, 0.5, 0.3, 0.7, 0.4],
        "open_hat": [None] * 10 + [0.6, None, None, None, None, 0.6],
        "clap": [None] * 4 + [0.9, None, None, None, None] + [None] * 4 + [0.8, None, None, None, None],
        "rim": [None, None, None, None, None, None, None, 0.4, None, None, None, None, None, None, None, None],
    },
    "trap": {
        "kick": [1.0, None, None, None, None, None, 0.8, None, None, None, 0.9, None, None, None, None, None],
        "snare": [None, None, None, None, 1.0, None, None, None, None, None, None, None, 1.0, None, None, 0.5],
        "hat": [0.8, 0.5, 0.65, 0.5, 0.8, 0.5, 0.65, 0.5, 0.8, 0.5, 0.65, 0.5, 0.8, 0.55, 0.75, 0.6],
        "open_hat": [None] * 10 + [0.7, None, None, None, None, None],
    },
    "hiphop": {
        "kick": [1.0, None, None, 0.6, None, None, None, 0.8, None, None, None, None, 0.6, None, None, None],
        "snare": [None, None, None, None, 1.0, None, None, None, None, None, None, None, 1.0, None, None, None],
        "hat": [0.6, 0.4, 0.55, 0.4, 0.6, 0.4, 0.55, 0.45, 0.6, 0.4, 0.55, 0.4, 0.6, 0.45, 0.6, 0.5],
    },
    "dnb": {
        "kick": [1.0, None, None, None, 1.0, None, None, None, 1.0, None, None, None, 1.0, None, None, None],
        "snare": [None, None, None, None, 1.0, None, None, None, None, None, None, None, 1.0, None, None, 0.6],
        "hat": [0.5, 0.35, 0.5, 0.35, 0.5, 0.35, 0.5, 0.4, 0.5, 0.35, 0.5, 0.35, 0.5, 0.4, 0.55, 0.45],
    },
    "edm": {
        "kick": [1.0, None, None, None, 1.0, None, None, None, 1.0, None, None, None, 1.0, None, None, None],
        "clap": [None, None, None, None, 1.0, None, None, None, None, None, None, None, 1.0, None, None, None],
        "hat": [None, 0.4, None, 0.4, None, 0.4, None, 0.4, None, 0.4, None, 0.4, None, 0.4, None, 0.5],
        "open_hat": [None] * 14 + [0.6, None],
        "crash": [1.0] + [None] * 15,
    },
    "breakbeat": {
        "kick": [1.0, None, None, 0.7, None, None, None, None, 0.8, None, None, None, None, None, 0.6, None],
        "snare": [None, None, None, None, 1.0, None, None, 0.5, None, None, None, None, 1.0, None, None, None],
        "hat": [0.5, None, 0.4, None, 0.5, None, 0.4, None, 0.5, None, 0.45, None, 0.5, None, 0.45, 0.5],
    },
    "pop": {
        "kick": [1.0, None, None, None, 1.0, None, None, 0.7, 1.0, None, None, None, 1.0, None, None, None],
        "snare": [None, None, None, None, 1.0, None, None, None, None, None, None, None, 1.0, None, None, None],
        "hat": [0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.5],
    },
    "afrobeat": {
        "kick": [1.0, None, None, 0.7, None, None, 1.0, None, None, 0.7, None, None, None, 0.7, None, None],
        "rim": [None, 0.5, None, None, None, 0.5, None, None, 0.6, None, None, 0.5, None, None, None, 0.5],
        "clap": [None] * 4 + [0.9] + [None] * 7 + [0.9] + [None] * 4,
        "hat": [0.45, 0.35, 0.45, 0.35, 0.45, 0.35, 0.5, 0.35, 0.45, 0.35, 0.45, 0.35, 0.45, 0.4, 0.5, 0.45],
    },
    "garage": {
        "kick": [1.0, None, None, None, 1.0, None, None, 1.0, None, None, None, None, 1.0, None, None, None],
        "snare": [None, None, None, None, 1.0, None, None, None, None, None, 0.7, None, 1.0, None, None, None],
        "hat": [0.6, 0.4, 0.55, 0.45, 0.6, 0.4, 0.55, 0.5, 0.6, 0.4, 0.55, 0.45, 0.6, 0.5, 0.6, 0.55],
    },
}

STYLE_ALIASES = {
    "edm": "edm", "electronic": "house", "electro": "house",
    "4-4": "house", "club": "house", "drum and bass": "dnb", "dnb": "dnb",
    "jungle": "dnb", "drum & bass": "dnb", "hip hop": "hiphop", "hip-hop": "hiphop",
    "rap": "trap", "drill": "trap", "r&b": "afrobeat", "africa": "afrobeat",
    "amapiano": "afrobeat", "uk garage": "garage", "garage": "garage",
    "break": "breakbeat", "rock": "pop", "song": "pop",
}


def resolve_style(name: str | None) -> str:
    if not name:
        return "house"
    want = name.lower().strip()
    if want in PATTERNS:
        return want
    if want in STYLE_ALIASES:
        return STYLE_ALIASES[want]
    for k in PATTERNS:
        if k in want:
            return k
    for k, v in STYLE_ALIASES.items():
        if k in want:
            return v
    return "house"


def _pick(label: str, rng: np.random.Generator) -> np.ndarray | None:
    hits = _pool(label, SR)
    if not hits:
        return None
    return hits[int(rng.integers(0, len(hits)))]


# ------------------------------------------------------------- hit access
_pools: dict[tuple[str, int], list[np.ndarray]] = {}


def _pool(label: str, sr: int) -> list[np.ndarray]:
    """Decoded one-shots for a label at a given rate, cached.

    Decoding on every hit would be far too slow for a dense pattern, so each
    label's bank is decoded once per sample rate and then sampled from.
    """
    key = (label, sr)
    if key in _pools:
        return _pools[key]
    idx = _load_index()
    hits = idx["hits"].get(label) or []
    import librosa

    out: list[np.ndarray] = []
    for h in hits:
        try:
            y, s = sf.read(str(BANK_DIR / h["file"]), dtype="float32")
        except Exception:
            continue
        if y.ndim > 1:
            y = y.mean(axis=1)
        if s != sr:
            y = librosa.resample(y, orig_sr=s, target_sr=sr).astype(np.float32)
        out.append(np.ascontiguousarray(y))
    _pools[key] = out
    return out


def hit(label: str, sr: int = SR, seed: int | None = None,
        max_seconds: float | None = None) -> np.ndarray | None:
    """A real recorded one-shot, resampled to `sr` and scaled to peak 0.89."""
    pool = _pool(label, sr)
    if not pool:
        return None
    rng = np.random.default_rng(
        seed if seed is not None else int(np.random.randint(0, 2**31 - 1))
    )
    y = pool[int(rng.integers(0, len(pool)))].copy()
    if max_seconds is not None:
        y = y[: int(max_seconds * sr)]
    peak = float(np.max(np.abs(y))) + 1e-12
    return (y / peak * 0.89).astype(np.float32)


def _tune(y: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch-shift by resampling (short percussive hits tolerate this)."""
    if abs(semitones) < 1e-3:
        return y
    rate = 2.0 ** (semitones / 12.0)
    n = max(1, int(len(y) / rate))
    idx = np.linspace(0, len(y) - 1, n)
    return np.interp(idx, np.arange(len(y)), y).astype(np.float32)


def render(
    bpm: float = 128.0,
    bars: int = 2,
    style: str | None = None,
    sr: int = SR,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Render a tempo-locked drum pattern from real recorded hits."""
    idx = _load_index()
    resolved = resolve_style(style)
    pattern = PATTERNS[resolved]
    rng = np.random.default_rng(seed if seed is not None else abs(hash((bpm, bars, resolved))) % (2**31))

    bpm = float(max(40.0, min(240.0, bpm)))
    step = 60.0 / bpm / 4.0          # one sixteenth
    steps = 16 * bars
    total = step * steps
    n = int(total * sr) + sr         # tail room for decaying hits
    buf = np.zeros(n, dtype=np.float32)

    counts: dict[str, int] = {}
    for label, slots in pattern.items():
        # Map our notional "open_hat" onto the bank, which has no open/closed
        # distinction -- real hats and cymbals cover both.
        bank_label = {"open_hat": "cymbal"}.get(label, label)
        if bank_label not in idx["hits"]:
            continue
        for i, vel in enumerate(slots):
            if vel is None:
                continue
            for bar in range(bars):
                step_index = bar * 16 + i
                if step_index >= steps:
                    break
                hit = _pick(bank_label, rng)
                if hit is None or hit.size == 0:
                    continue
                if label == "hat":
                    hit = _tune(hit, rng.uniform(-2.0, 2.0))
                elif label == "open_hat":
                    hit = _tune(hit, rng.uniform(-1.0, 1.0))
                amp = float(vel) * (0.75 + 0.5 * rng.random())
                start = int(step_index * step * sr)
                end = min(len(buf), start + len(hit))
                if start < end:
                    buf[start:end] += hit[: end - start] * amp
                    counts[label] = counts.get(label, 0) + 1

    # Soft limit so layered hits do not clip harshly.
    peak = float(np.max(np.abs(buf))) + 1e-9
    if peak > 0.95:
        buf = np.tanh(buf / peak * 1.1) * 0.95

    buf = buf[: int(total * sr)]
    meta = {
        "style": resolved,
        "bpm": round(bpm, 2),
        "bars": bars,
        "hits_used": sum(counts.values()),
        "hit_counts": counts,
        "source": "real recorded one-shots (airasoul/drum-kit)",
        "bank_labels": idx["labels"],
        "seconds": round(len(buf) / sr, 3),
    }
    return buf, meta


def styles() -> list[str]:
    return sorted(PATTERNS)