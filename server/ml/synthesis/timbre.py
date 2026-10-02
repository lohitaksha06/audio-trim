"""Instrument synthesis driven by a trained timbre model.

What is learned and what is not
-------------------------------
*Learned*: the spectral envelope (32 bands) and envelope shape (attack, decay
ratio, centroid, flatness) as a function of instrument family, pitch and
velocity. These are produced by a small MLP trained on real instrument
recordings -- see ``scripts/train_timbre_model.py``.

*Not learned*: the renderer that turns a predicted envelope into audio is
deterministic additive synthesis. That is the standard hybrid design used by
DDSP and NSF models (network predicts harmonic/noise magnitudes; a
synthesiser renders them), and it is stated here rather than implied to be a
generative model.

This replaces the old hand-written presets, where every "instrument" was a
fixed list of harmonic ratios and an ADSR chosen by hand.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "timbre_model.pt"

_lock = threading.Lock()
_cache: dict | None = None

MAX_PARTIALS = 64


class TimbreModelUnavailable(RuntimeError):
    """Raised when the trained timbre model has not been built."""


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        if not MODEL_PATH.exists():
            raise TimbreModelUnavailable(
                f"timbre model not found at {MODEL_PATH}. Build it with:\n"
                "  python -m scripts.train_timbre_model"
            )
        import torch

        blob = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        blob["torch"] = torch
        _cache = blob
        return _cache


def available() -> bool:
    return MODEL_PATH.exists()


def families() -> list[str]:
    try:
        return list(_load()["families"])
    except TimbreModelUnavailable:
        return []


def _nearest_family(requested: str) -> str | None:
    """Map a user word ('piano', 'guitar') onto a trained family.

    Falls back to the closest trained family by shared keyword, then to the
    nearest family name by character overlap. Returns ``None`` only when the
    request shares nothing with the trained set.
    """
    fams = _load()["families"]
    if not fams:
        return None
    want = requested.lower().strip().replace(" ", "_")
    if want in fams:
        return want

    bases = sorted({f.split("/")[0] for f in fams})
    if want in bases:
        return next(f for f in fams if f.split("/")[0] == want)
    for f in fams:
        if want in f or f.split("/")[0] in want:
            return f

    # Aliases users actually type.
    aliases = {
        "piano": "keyboard",
        "keys": "keyboard",
        "synth": "synth_lead",
        "synths": "synth_lead",
        "lead": "synth_lead",
        "pad": "organ",
        "strings": "string",
        "violin": "string",
        "cello": "string",
        "sax": "reed",
        "saxophone": "reed",
        "clarinet": "reed",
        "horns": "brass",
        "trumpet": "brass",
        "vocals": "vocal",
        "voice": "vocal",
        "bells": "mallet",
        "marimba": "mallet",
        "xylo": "mallet",
        "percussion": "mallet",
        "drums": "mallet",
        "bassline": "bass",
        "lowend": "bass",
    }
    alias = aliases.get(want)
    if alias and alias in bases:
        return next(f for f in fams if f.split("/")[0] == alias)

    # No fuzzy character-overlap fallback on purpose. It mapped unrelated words
    # onto arbitrary families ("acid" -> vocal/acoustic), which is the same
    # substring trap that once made "phonk" resolve to "honk". Returning None
    # sends the caller to its DSP fallback, which is the honest outcome for a
    # sound with no trained equivalent.
    return None


def predict(family: str, pitch: int, velocity: int = 100) -> dict:
    """Predict a timbre for (family, pitch, velocity).

    Returns band magnitudes in linear amplitude plus shape scalars.
    """
    blob = _load()
    torch = blob["torch"]
    fam = _nearest_family(family)
    if fam is None:
        raise KeyError(
            f"no trained timbre for '{family}'; trained families: "
            + ", ".join(sorted({f.split('/')[0] for f in blob['families']}))
        )
    idx = blob["vocab"][fam]
    n_bands = int(blob["n_bands"])
    n_env = int(blob["n_env"])

    pv = torch.tensor([[pitch / 127.0, velocity / 127.0]], dtype=torch.float32)
    fi = torch.tensor([idx], dtype=torch.long)
    with torch.no_grad():
        pred = _net(blob)(fi, pv).numpy()[0]

    mu = np.asarray(blob["mu"], dtype=np.float32)
    sd = np.asarray(blob["sd"], dtype=np.float32)
    target = pred * sd + mu

    bands_log = target[:n_bands]
    # invert the feature transform: log1p(x*100) -> x
    bands = np.expm1(bands_log.astype(np.float64)) / 100.0
    bands = np.clip(bands, 0.0, None)

    return {
        "family": fam,
        "requested_family": family,
        "bands": bands.astype(np.float32),
        "band_edges": _band_edges(n_bands),
        "attack": float(np.clip(target[n_bands], 0.0, 1.0)),
        "decay_ratio": float(np.clip(target[n_bands + 1], 0.0, 1.0)),
        "centroid": float(np.clip(target[n_bands + 2], 0.0, 1.0)),
        "flatness": float(np.clip(target[n_bands + 3], 0.0, 1.0)),
    }


def _net(blob: dict):
    """Rebuild the module and load its weights."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    try:
        from scripts.train_timbre_model import TimbreNet
    except Exception:  # pragma: no cover - fall back to a local definition
        import torch.nn as nn

        class TimbreNet(nn.Module):
            def __init__(self, n_families, hidden=128, emb=24):
                super().__init__()
                self.emb = nn.Embedding(n_families, emb)
                self.body = nn.Sequential(
                    nn.Linear(emb + 2, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden), nn.SiLU(),
                    nn.Linear(hidden, sum(blob["mu"])),
                )

            def forward(self, fam, pv):
                return self.body(torch.cat([self.emb(fam), pv], dim=1))

    net = TimbreNet(max(len(blob["vocab"]), 2))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net


def _band_edges(n_bands: int, sr: int = 16000) -> np.ndarray:
    return np.geomspace(60, min(7800, sr / 2 - 1), n_bands + 1)


def midi_to_hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def render_note(
    family: str,
    pitch: int,
    duration: float = 0.5,
    velocity: int = 100,
    sr: int = 22050,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Render one note. Returns (mono audio, the timbre dict used)."""
    tim = predict(family, pitch, velocity)
    rng = np.random.default_rng(
        seed if seed is not None else (hash((family, pitch, velocity)) & 0xFFFFFFFF)
    )

    f0 = midi_to_hz(pitch)
    n = max(64, int(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr

    # Partial amplitudes: read the predicted envelope at each partial's
    # frequency. This is the DDSP-style step -- learned magnitudes, rendered
    # by a deterministic oscillator bank.
    edges = _band_edges(len(tim["bands"]))
    amps: list[float] = []
    for k in range(1, MAX_PARTIALS + 1):
        fh = f0 * k
        if fh >= sr / 2 * 0.98:
            break
        # log-frequency interpolation between learned bands
        frac = np.interp(np.log(fh), np.log(edges), np.arange(len(edges)))
        i0 = min(int(frac), len(tim["bands"]) - 1)
        i1 = min(i0 + 1, len(tim["bands"]) - 1)
        w = frac - i0
        a = (1 - w) * tim["bands"][i0] + w * tim["bands"][i1]
        # Real instruments roll off with partial number.
        a *= 1.0 / (1.0 + 0.06 * (k - 1))
        # A little per-note variation so repeats are not identical.
        a *= 1.0 + rng.uniform(-0.08, 0.08)
        amps.append(max(float(a), 0.0))

    if not amps:
        return np.zeros(n, dtype=np.float32), tim

    amps_arr = np.asarray(amps)
    top = float(amps_arr.max())
    if top <= 1e-9:
        return np.zeros(n, dtype=np.float32), tim
    amps_arr = amps_arr / top

    sig = np.zeros(n, dtype=np.float64)
    for i, a in enumerate(amps_arr, start=1):
        # Slight inharmonicity keeps it from sounding like a pure organ.
        drift = 1.0 + 0.0006 * i * i
        sig += a * np.sin(2 * np.pi * f0 * i * drift * t + rng.uniform(0, 2 * np.pi))

    # Noise layer for breath / attack character, scaled by measured flatness.
    noise_amt = float(np.clip(0.35 * (1.0 - tim["flatness"]), 0.0, 0.35))
    if noise_amt > 0.01:
        sig += noise_amt * rng.standard_normal(n)

    sig = _adsr(
        sig,
        sr,
        attack_s=0.002 + tim["attack"] * min(0.35, duration * 0.6),
        decay_ratio=0.25 + 0.7 * tim["decay_ratio"],
        sustain=0.45 + 0.4 * tim["decay_ratio"],
    )

    peak = float(np.max(np.abs(sig))) + 1e-12
    level = 0.12 + 0.55 * (velocity / 127.0)
    out = (sig / peak * level).astype(np.float32)

    # Soft-clip rather than hard-limit so heavy layers do not buzz.
    out = np.tanh(out * 1.2) / np.tanh(1.2)
    return out.astype(np.float32), tim


def _adsr(
    x: np.ndarray,
    sr: int,
    attack_s: float,
    decay_ratio: float,
    sustain: float,
    release_s: float = 0.05,
) -> np.ndarray:
    n = len(x)
    a = min(int(attack_s * sr), max(1, n // 4))
    r = min(int(release_s * sr), max(1, n // 4))
    d = min(int((n - a - r) * decay_ratio), max(0, n - a - r))

    env = np.ones(n, dtype=np.float64)
    env[:a] = np.linspace(0.0, 1.0, a)
    if d > 0:
        env[a:a + d] = np.linspace(1.0, sustain, d)
    env[n - r:] *= np.linspace(1.0, 0.0, r)
    return x * env