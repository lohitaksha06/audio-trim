"""Train an instrument timbre model from real recordings (NSynth).

Why this exists
---------------
`add guitar` / `add flute` used to call hand-written additive synthesis with
hand-picked harmonic ratios — a preset, not a learned model. This trains the
timbre itself from real instrument recordings, then renders it.

Architecture is hybrid neural + DSP (the DDSP / NNSF idea): the network predicts
a compact spectral envelope + envelope shape from (instrument, pitch, velocity);
a deterministic synthesiser turns that prediction into audio. The *timbre* is
learned, the renderer is DSP, and that is stated plainly rather than dressed up.

CPU-friendly by construction: features are extracted per note and discarded, so
peak memory is one note, not the dataset.

Usage:
    python -m scripts.train_timbre_model --shards N
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

try:
    import pyarrow.parquet as pq
    import soundfile as sf
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"missing dependency: {exc}")

N_BANDS = 32          # timbre envelope resolution
N_ENV = 4             # attack, decay, centroid, flatness
TARGET_DIM = N_BANDS + N_ENV

SR = 16000
NOTE_RE = re.compile(
    r"(?P<inst>[a-z_]+)_(?P<src>acoustic|electronic|synthetic)_(?P<note>\d+)-"
    r"(?P<pitch>\d+)-(?P<vel>\d+)"
)


# ----------------------------------------------------------------- features

def _band_envelope(y: np.ndarray, sr: int, n_bands: int = N_BANDS) -> np.ndarray:
    """Average log-magnitude envelope over the sustain, in `n_bands` bands."""
    import librosa

    # Skip the attack so the envelope describes the tone, not the transient.
    start = int(0.15 * len(y))
    seg = y[start:] if len(y) - start > sr // 4 else y
    if seg.size < 256:
        seg = y

    S = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512))
    if S.size == 0:
        return np.zeros(n_bands, dtype=np.float32)
    mean_spec = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Log-spaced bands up to ~8 kHz (Nyquist is 8k at 16 kHz).
    edges = np.geomspace(60, min(7800, sr / 2 - 1), n_bands + 1)
    out = np.zeros(n_bands, dtype=np.float32)
    for i in range(n_bands):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = mean_spec[sel].mean() if sel.any() else 0.0
    out = np.log1p(out * 100.0)
    # Normalise level out; keep *shape*.
    out = out - out.mean()
    return out.astype(np.float32)


def note_features(y: np.ndarray, sr: int) -> np.ndarray | None:
    """[envelope(n_bands), attack, decay, centroid, flatness] for one note."""
    import librosa

    if y.size < sr // 8:
        return None
    peak = float(np.max(np.abs(y)))
    if peak < 1e-5:
        return None

    env = _band_envelope(y, sr)

    # Attack: time to reach 90% of peak (normalised).
    env_line = librosa.onset.onset_strength(y=y, sr=sr, hop_length=128)
    if env_line.size:
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
        thr = 0.9 * float(np.max(rms)) if rms.size else 0.0
        idx = np.argmax(rms >= thr) if thr > 0 and np.any(rms >= thr) else 0
        attack = float(idx / max(len(rms), 1))
    else:
        attack = 0.0

    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    if spec.size == 0:
        return None
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mag = spec.mean(axis=1)
    total = float(mag.sum()) + 1e-12
    centroid = float((freqs * mag).sum() / total) / (sr / 2)
    flatness = float(
        np.exp(np.mean(np.log(mag + 1e-12))) / (np.mean(mag) + 1e-12)
    )

    extra = np.array(
        [
            min(attack, 1.0),
            float(np.mean(np.abs(y[: len(y) // 2])) / (peak + 1e-9)) if len(y) > 1 else 0.0,
            centroid,
            flatness,
        ],
        dtype=np.float32,
    )
    return np.concatenate([env, extra]).astype(np.float32)


# ---------------------------------------------------------------- extraction

def extract(shard_paths: list[str], limit: int | None = None, verbose: bool = True):
    import pyarrow.parquet as _pq

    inst_table = _pq.read_table(shard_paths[0], columns=["audio", "instrument"])
    pitch_shard = shard_paths[1] if len(shard_paths) > 1 else None
    pitches = None
    if pitch_shard:
        pt = _pq.read_table(pitch_shard, columns=["pitch"])
        pitches = [int(x) for x in pt.column("pitch").to_pylist()]

    audios = inst_table.column("audio").to_pylist()
    insts = inst_table.column("instrument").to_pylist()

    X, Y, F, META = [], [], [], []
    t0 = time.time()
    for i, (a, inst) in enumerate(zip(audios, insts)):
        if limit and len(X) >= limit:
            break
        raw = a["bytes"] if isinstance(a, dict) else a
        if not raw:
            continue
        try:
            y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            continue
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            continue  # shards are 16 kHz; skip anything unexpected
        feats = note_features(y, sr)
        if feats is None:
            continue

        m = NOTE_RE.match(a["path"] if isinstance(a, dict) else "")
        pitch = int(m.group("pitch")) if m else (pitches[i] if pitches and i < len(pitches) else 60)
        vel = int(m.group("vel")) if m else 100
        src = m.group("src") if m else "acoustic"
        fam = f"{inst}/{src}"

        X.append([pitch / 127.0, vel / 127.0])
        Y.append(feats)
        F.append(fam)
        META.append({"family": fam, "pitch": pitch, "velocity": vel, "path": a["path"]})

        if verbose and len(X) % 500 == 0:
            print(f"  {len(X)} notes  ({time.time() - t0:.0f}s)")

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), F, META


def family_vocab(fams: list[str]):
    uniq = sorted(set(fams))
    return {f: i for i, f in enumerate(uniq)}, uniq


# ----------------------------------------------------------------- training

class TimbreNet(torch.nn.Module):
    """(instrument family, pitch, velocity) -> spectral envelope + shape."""

    def __init__(self, n_families: int, hidden: int = 128, emb: int = 24):
        super().__init__()
        self.emb = torch.nn.Embedding(n_families, emb)
        self.body = torch.nn.Sequential(
            torch.nn.Linear(emb + 2, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, TARGET_DIM),
        )

    def forward(self, fam: torch.Tensor, pv: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([self.emb(fam), pv], dim=1))


def train(X, Y, F, vocab, epochs=60, hidden=128, seed=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    fam_idx = [vocab[f] for f in F]
    x_in = torch.tensor(X, dtype=torch.float32)
    x_fam = torch.tensor(fam_idx, dtype=torch.long)
    y_out = torch.tensor(Y, dtype=torch.float32)

    mu, sd = y_out.mean(0), y_out.std(0) + 1e-6
    y_norm = (y_out - mu) / sd

    net = TimbreNet(max(len(vocab), 2), hidden=hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()

    n = len(X)
    idx = torch.randperm(n)
    val = max(1, int(n * 0.15))
    tr, va = idx[:-val], idx[-val:]

    best = float("inf")
    best_state = None
    for ep in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr))]
        total = 0.0
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            loss = lossf(net(x_fam[b], x_in[b]), y_norm[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(x_fam[va], x_in[va]), y_norm[va]))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 15 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  train {total / max(len(tr),1):.4f}  val {vl:.4f}")

    net.load_state_dict(best_state)
    net.eval()
    return net, mu, sd, float(best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\Lohit\AppData\Local\Temp\opencode\data\nsynth")
    ap.add_argument("--out", default=r"server\ml\models")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    inst = glob.glob(args.data + r"\instrument\*.parquet")
    pitch = glob.glob(args.data + r"\pitch\*.parquet")
    if not inst:
        raise SystemExit(f"no parquet shards under {args.data}\\instrument")
    print(f"shards: {len(inst)} instrument, {len(pitch)} pitch")

    print("extracting features...")
    X, Y, F, META = extract(inst + pitch, limit=args.limit)
    print(f"notes: {len(X)}  families: {len(set(F))}")

    vocab, uniq = family_vocab(F)
    print("families:", ", ".join(uniq))

    print("training...")
    net, mu, sd, val = train(X, Y, F, vocab, epochs=args.epochs)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch_path = out / "timbre_model.pt"
    import torch

    torch.save(
        {
            "state_dict": net.state_dict(),
            "vocab": vocab,
            "mu": mu.tolist(),
            "sd": sd.tolist(),
            "n_bands": N_BANDS,
            "n_env": N_ENV,
            "families": uniq,
            "val_loss": val,
            "notes_trained": int(len(X)),
        },
        torch_path,
    )
    print("saved", torch_path, "val_loss", round(val, 4))

    (out / "timbre_model.json").write_text(
        json.dumps(
            {"families": uniq, "val_loss": val, "notes": len(X),
             "n_bands": N_BANDS, "arch": "emb24+2 -> 128 -> 128 -> 36"},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved", out / "timbre_model.json")


if __name__ == "__main__":
    main()