#!/usr/bin/env python3
"""Diagnoser: small CNN that predicts SNR (dB) and clipped fraction from log-mel.

Two outputs, one head each:

    snr_db           regression, 0-30 dB, predicted as a normalised value
    clipped_fraction regression in [0, 1] via a sigmoid

``--smoke`` proves the loop and prints a real timing number with zero downloads.

    python scripts/train_diagnoser.py --smoke
    python scripts/train_diagnoser.py --manifest data/manifest.json

CPU only, threads capped at 4.

Splits: the manifest carries a SPEAKER-DISJOINT train/val split. Training
respects that split (it never re-splits by clip, which would leak a speaker
into both sides). The script then reports BOTH a seen-speaker score and an
unseen-speaker score, because the gap between them is the overfitting signal.
Per-epoch train/val loss is saved to ``outputs/metrics/`` for a learning curve.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import (  # noqa: E402  - sibling module, shared signal model
    SR,
    hard_clip,
    load_audio,
    mix_at_snr,
    normalise_peak,
    resample_to_16k,
    synth_noise,
    synth_speech,
)

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "metrics"

N_FFT = 512
HOP = 160          # 100 frames per second
N_MELS = 64
SNR_SCALE = 30.0   # dB, so the SNR head works in roughly [-1, 1]


# ---------------------------------------------------------------------------
# Log-mel features (numpy only)
# ---------------------------------------------------------------------------

def mel_filterbank(sr: int, n_fft: int, n_mels: int,
                   fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """HTK-style triangular mel filterbank, shape (n_fft // 2 + 1, n_mels)."""
    fmax = fmax or sr / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        if centre == left:
            centre = left + 1
        if right == centre:
            right = centre + 1
        for k in range(left, min(centre, fb.shape[0])):
            fb[k, m - 1] = (k - left) / max(1, centre - left)
        for k in range(centre, min(right, fb.shape[0])):
            fb[k, m - 1] = (right - k) / max(1, right - centre)
    return fb


MEL_FB = torch.from_numpy(mel_filterbank(SR, N_FFT, N_MELS))
WINDOW = torch.hann_window(N_FFT)


def log_mel_batch(clips: np.ndarray) -> torch.Tensor:
    """(N, T) float32 audio -> (N, 1, n_mels, frames) standardised log-mel."""
    x = torch.from_numpy(np.ascontiguousarray(clips, dtype=np.float32))
    x = x - x.mean(dim=1, keepdim=True)          # drop DC / offset
    spec = torch.stft(
        x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=WINDOW,
        center=True, return_complex=True,
    ).abs().pow(2.0)                              # (N, freq, F)
    mel = torch.matmul(MEL_FB.T, spec)            # (N, n_mels, F)
    logmel = torch.log(mel + 1e-8)
    logmel = (logmel - logmel.mean(dim=(1, 2), keepdim=True)) / (
        logmel.std(dim=(1, 2), keepdim=True) + 1e-5)
    return logmel.unsqueeze(1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Diagnoser(nn.Module):
    """~150k parameters. Three conv blocks, pooled, two regression heads."""

    def __init__(self, n_mels: int = N_MELS, width: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 16, 128), nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.snr_head = nn.Linear(128, 1)
        self.clip_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.body(x))
        snr = self.snr_head(h).squeeze(1) * SNR_SCALE     # dB
        clip = torch.sigmoid(self.clip_head(h)).squeeze(1)
        return snr, clip


def loss_fn(snr_pred, clip_pred, snr_true, clip_true) -> torch.Tensor:
    """L1 on both, with the SNR term normalised so the two are comparable."""
    return (torch.nn.functional.l1_loss(snr_pred / SNR_SCALE, snr_true / SNR_SCALE)
            + torch.nn.functional.l1_loss(clip_pred, clip_true))


@torch.no_grad()
def evaluate(model, feats, snr_true, clip_true, batch: int = 128):
    model.eval()
    total, n = 0.0, 0
    mae_snr, mae_clip = 0.0, 0.0
    for i in range(0, feats.shape[0], batch):
        f = feats[i:i + batch]
        s, c = model(f)
        ts, tc = snr_true[i:i + batch], clip_true[i:i + batch]
        total += float(loss_fn(s, c, ts, tc)) * f.shape[0]
        mae_snr += float((s - ts).abs().sum())
        mae_clip += float((c - tc).abs().sum())
        n += f.shape[0]
    model.train()
    return total / max(1, n), mae_snr / max(1, n), mae_clip / max(1, n)


@torch.no_grad()
def predict(model, feats, batch: int = 128):
    """Return (snr_pred, clip_pred) tensors for all rows, for thresholding."""
    model.eval()
    snrs, clips = [], []
    for i in range(0, feats.shape[0], batch):
        s, c = model(feats[i:i + batch])
        snrs.append(s.detach().cpu())
        clips.append(c.detach().cpu())
    model.train()
    return torch.cat(snrs), torch.cat(clips)


def clip_classification_metrics(pred_frac, true_frac, frac_threshold: float = 0.001):
    """Treat "is this clip clipped?" as binary classification.

    Returns precision/recall/F1 (macro of the two classes) so the report can
    talk about class separability even though the primary task is regression.
    """
    y_true = (true_frac >= frac_threshold).astype(int)
    y_pred = (pred_frac >= frac_threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "positive_rate": float(y_true.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def make_smoke_data(count: int, seed: int, seconds: float = 1.0):
    """Tone + noise at a known SNR, hard-clipped at a known level."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    clips = np.empty((count, n), dtype=np.float32)
    snr = np.empty(count, dtype=np.float32)
    frac = np.empty(count, dtype=np.float32)

    for i in range(count):
        speech = synth_speech(n, rng)
        noise = synth_noise(n, rng)
        s = float(rng.uniform(0.0, 30.0))
        mixed = normalise_peak(mix_at_snr(speech, noise, s, rng),
                               float(rng.uniform(0.6, 1.0)))
        threshold = float(np.max(np.abs(mixed)) * rng.uniform(0.15, 0.9))
        clips[i], frac[i] = hard_clip(mixed, threshold)
        snr[i] = s
    return clips, snr, frac


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

def load_manifest_data(manifest_path: Path, limit: int | None):
    """Read a preprocess.py manifest and rebuild features + labels.

    Returns clips, snr, frac, speakers, is_val. The speaker-disjoint split from
    the manifest is preserved; it is NOT re-split here.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tracks = manifest.get("tracks", [])
    if not tracks:
        raise SystemExit(f"{manifest_path}: no 'tracks' in manifest")

    clips, snr, frac, speakers, is_val = [], [], [], [], []
    for t in tracks[:limit] if limit else tracks:
        path = Path(t["file"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            print(f"  skip missing {path}", file=sys.stderr)
            continue
        x, sr = load_audio(path)
        clips.append(resample_to_16k(x, sr))
        snr.append(t["snr_db"])
        frac.append(t["clipped_fraction"])
        speakers.append(t.get("speaker", "?"))
        is_val.append(t.get("split") == "val")
    if not clips:
        raise SystemExit("no usable audio in manifest")
    length = min(c.size for c in clips)
    return (np.stack([c[:length] for c in clips]),
            np.asarray(snr, np.float32),
            np.asarray(frac, np.float32),
            np.asarray(speakers, dtype=object),
            np.asarray(is_val, dtype=bool))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(feats, snr, frac, *, epochs: int, batch: int, lr: float,
          val_mask: np.ndarray, seed: int, label: str,
          speakers: np.ndarray | None = None) -> dict:
    torch.manual_seed(seed)
    n = feats.shape[0]
    val_mask = np.asarray(val_mask, dtype=bool)
    train_idx = torch.from_numpy(np.where(~val_mask)[0])
    val_idx = torch.from_numpy(np.where(val_mask)[0])

    feats = feats.float()
    snr_t = torch.from_numpy(snr).float()
    frac_t = torch.from_numpy(frac).float()

    model = Diagnoser()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n[{label}]")
    print(f"  clips          {n}  (train {len(train_idx)}, val {len(val_idx)})")
    print(f"  labels         snr {snr.min():.1f}-{snr.max():.1f} dB, "
          f"clipped {frac.min():.4f}-{frac.max():.4f}")
    print(f"  features       {tuple(feats.shape[1:])}  = "
          f"{int(np.prod(feats.shape[1:]))} values per clip")
    print(f"  parameters     {n_params:,}")
    print(f"  threads        {torch.get_num_threads()}")

    before = evaluate(model, feats[val_idx], snr_t[val_idx], frac_t[val_idx])
    print(f"  loss before    {before[0]:.4f}   "
          f"(MAE snr {before[1]:.2f} dB, MAE clip {before[2]:.4f})")

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "val_mae_snr": [], "val_mae_clip": []}
    step_times: list[float] = []
    epoch_times: list[float] = []

    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        order = train_idx[torch.randperm(len(train_idx))]
        running, seen = 0.0, 0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            t0 = time.perf_counter()
            snr_pred, clip_pred = model(feats[idx])
            loss = loss_fn(snr_pred, clip_pred, snr_t[idx], frac_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step_times.append(time.perf_counter() - t0)
            running += float(loss.detach()) * len(idx)
            seen += len(idx)
        train_loss = running / max(1, seen)
        dt = time.perf_counter() - t_epoch
        epoch_times.append(dt)

        val_loss, val_snr, val_clip = evaluate(
            model, feats[val_idx], snr_t[val_idx], frac_t[val_idx])
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_mae_snr"].append(round(val_snr, 4))
        history["val_mae_clip"].append(round(val_clip, 6))
        print(f"  epoch {epoch + 1:2d}/{epochs}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  (MAE snr {val_snr:.2f} dB, "
              f"clip {val_clip:.4f})  {dt:5.2f}s")

    # Trivial baseline: always predict the training mean.
    mean_snr = float(snr_t[train_idx].mean())
    mean_frac = float(frac_t[train_idx].mean())
    base_snr = float((snr_t[val_idx] - mean_snr).abs().mean())
    base_frac = float((frac_t[val_idx] - mean_frac).abs().mean())

    # Seen (a subset of train speakers) vs unseen (val speakers): the gap is
    # the overfitting signal.
    seen_loss, seen_snr, seen_clip = evaluate(
        model, feats[train_idx], snr_t[train_idx], frac_t[train_idx])
    unseen_loss, unseen_snr, unseen_clip = evaluate(
        model, feats[val_idx], snr_t[val_idx], frac_t[val_idx])

    after = (unseen_loss, unseen_snr, unseen_clip)

    # Clipping as a binary classification (for the separability section).
    val_snr_pred, val_clip_pred = predict(model, feats[val_idx])
    clip_cls = clip_classification_metrics(val_clip_pred.numpy(), frac_t[val_idx].numpy())

    print(f"  loss after     {after[0]:.4f}   "
          f"(MAE snr {after[1]:.2f} dB, MAE clip {after[2]:.4f})")
    print(f"  mean-baseline  MAE snr {base_snr:.2f} dB, "
          f"MAE clip {base_frac:.4f}")
    print(f"  seen/unseen    MAE snr {seen_snr:.2f} dB vs {unseen_snr:.2f} dB "
          f"(gap {unseen_snr - seen_snr:+.2f}); "
          f"clip {seen_clip:.4f} vs {unseen_clip:.4f}")
    print(f"  clip (cls)     P {clip_cls['precision']:.3f}  "
          f"R {clip_cls['recall']:.3f}  F1 {clip_cls['f1']:.3f}  "
          f"acc {clip_cls['accuracy']:.3f}  "
          f"(clip+ {clip_cls['positive_rate']:.1%})")
    print(f"  loss change    {before[0]:.4f} -> {after[0]:.4f}  "
          f"({(after[0] - before[0]) / before[0] * 100:+.1f}%)")
    if step_times:
        print(f"  time/step      {float(np.mean(step_times)) * 1000:.1f} ms  "
              f"({len(step_times)} steps)")
        print(f"  time/epoch     {float(np.mean(epoch_times)):.2f} s  "
              f"({epochs} epochs)")

    history["seen_mae_snr"] = round(seen_snr, 4)
    history["unseen_mae_snr"] = round(unseen_snr, 4)
    history["seen_mae_clip"] = round(seen_clip, 6)
    history["unseen_mae_clip"] = round(unseen_clip, 6)
    history["baseline_mae_snr"] = round(base_snr, 4)
    history["baseline_mae_clip"] = round(base_frac, 6)
    history["n_params"] = n_params
    history["n_train"] = int(len(train_idx))
    history["n_val"] = int(len(val_idx))
    history["clip_cls"] = clip_cls
    if speakers is not None:
        history["val_speakers"] = sorted({s for s, m in zip(speakers, val_mask) if m})
    return history, model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Train the SNR/clipping diagnoser (CPU, 4 threads).")
    p.add_argument("--smoke", action="store_true",
                   help="synthetic clips in memory, no downloads")
    p.add_argument("--manifest", type=Path,
                   help="manifest from preprocess.py (real audio)")
    p.add_argument("--count", type=int, default=500,
                   help="smoke clips (default 500)")
    p.add_argument("--limit", type=int, default=None,
                   help="max clips from a real manifest")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--steps", type=int, default=None,
                   help="cap total optimiser steps (smoke defaults to 100)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="smoke-only: fraction of clips held out")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-metrics", type=Path, default=None,
                   help="where to save the per-epoch history (JSON)")
    args = p.parse_args(argv)

    if not args.smoke and not args.manifest:
        p.print_help()
        print("\nNothing to do: pass --smoke (synthetic) or --manifest <path>.")
        return 0

    started = time.perf_counter()

    if args.smoke:
        t0 = time.perf_counter()
        clips, snr, frac = make_smoke_data(args.count, args.seed)
        print(f"generated {args.count} synthetic 1 s clips in "
              f"{time.perf_counter() - t0:.2f}s")
        label = "smoke: synthetic tone + noise"
        epochs = args.epochs
        if args.steps:
            per_epoch = max(1, int(len(clips) * (1 - args.val_frac)) // args.batch)
            epochs = max(1, min(epochs, args.steps // per_epoch or 1))
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(clips))
        n_val = max(1, int(round(len(clips) * args.val_frac)))
        val_mask = np.zeros(len(clips), dtype=bool)
        val_mask[perm[:n_val]] = True
        speakers = None
    else:
        clips, snr, frac, speakers, is_val = load_manifest_data(
            args.manifest, args.limit)
        print(f"loaded {len(clips)} clips from {args.manifest} "
              f"(val speakers: {sorted({s for s, m in zip(speakers, is_val) if m})})")
        label = f"real: {args.manifest.name}"
        epochs = args.epochs
        val_mask = is_val

    t0 = time.perf_counter()
    feats = log_mel_batch(clips)
    print(f"log-mel: {tuple(feats.shape)} in {time.perf_counter() - t0:.2f}s")

    history, model = train(feats, snr, frac, epochs=epochs, batch=args.batch, lr=args.lr,
                           val_mask=val_mask, seed=args.seed, label=label,
                           speakers=speakers)

    out = args.out_metrics or OUT_DIR / f"{label.replace(' ', '_').replace(':', '-')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"\n  metrics saved  {out}")

    # Save the trained model so the demo can load it later.
    model_dir = ROOT / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = "diagnoser.pt" if args.manifest else "diagnoser_smoke.pt"
    torch.save(model.state_dict(), model_dir / model_name)
    config = {
        "model": "Diagnoser",
        "task": "SNR (dB) + clipped-fraction regression on 16 kHz log-mel",
        "sample_rate": SR,
        "n_fft": N_FFT,
        "hop": HOP,
        "n_mels": N_MELS,
        "snr_scale": SNR_SCALE,
        "n_params": history["n_params"],
        "feature_norm": "per-clip standardisation (mean/std), as in training",
        "note": "Load with: Diagnoser(); model.load_state_dict(torch.load(path))",
    }
    (model_dir / (model_name.replace(".pt", ".json"))).write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"  model saved    {model_dir / model_name}")

    print(f"  total wall     {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
