#!/usr/bin/env python3
"""Evaluate the trained diagnoser against ground truth AND the DSP baseline.

The DSP baseline is a numpy port of the shipped browser code
(`website/src/lib/audio/clean.ts`): frame-percentile SNR (10th vs 90th
percentile of 20 ms frame energy) and plateau-based clipping detection. This is
the same estimator the product shows in `/features/clean`, so the comparison is
apples-to-apples with what a user actually sees.

    python scripts/evaluate.py --model outputs/models/diagnoser.pt \
                               --manifest data/manifest.test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import SR  # noqa: E402
from train_diagnoser import (  # noqa: E402
    Diagnoser,
    log_mel_batch,
    load_manifest_data,
    predict,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "metrics"
CLIP_THRESHOLD = 0.9999
CLIP_MIN_RUN = 3


# ---------------------------------------------------------------------------
# DSP baseline (port of clean.ts)
# ---------------------------------------------------------------------------

def _percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p))


def dsp_snr(clip: np.ndarray, sr: int = SR) -> tuple[float, bool]:
    """Frame-percentile SNR estimate. Returns (snr_db, reliable)."""
    frame = max(64, int(0.02 * sr))
    if clip.size < frame:
        rms = float(np.sqrt(np.mean(clip * clip) + 1e-12))
        level = 20.0 * np.log10(rms + 1e-12)
        return 0.0, False
    hop = max(1, frame // 2)
    levels = []
    for start in range(0, clip.size - frame + 1, hop):
        seg = clip[start:start + frame]
        levels.append(20.0 * np.log10(np.sqrt(np.mean(seg * seg)) + 1e-12))
    levels = np.asarray(levels, dtype=np.float32)
    noise = _percentile(levels, 10)
    active = _percentile(levels, 90)
    snr = active - noise
    reliable = (active - noise) >= 6 and active > -70
    return float(np.clip(snr, 0.0, 80.0)), bool(reliable)


def dsp_clip_present(clip: np.ndarray) -> bool:
    """True if there is a flat-topped plateau at full scale (>=3 samples)."""
    tolerance = 1e-6
    run_start = -1
    flat = True
    found = False
    for i in range(len(clip)):
        value = clip[i]
        if abs(value) >= CLIP_THRESHOLD:
            if run_start < 0:
                run_start = i
                flat = True
            elif abs(value - clip[run_start]) > tolerance:
                flat = False
        else:
            if run_start >= 0 and (i - run_start) >= CLIP_MIN_RUN and flat:
                found = True
                break
            run_start = -1
            flat = True
    return found


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def binary_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1(precision, recall),
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "positive_rate": float(y_true.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the diagnoser + DSP baseline.")
    p.add_argument("--model", type=Path, default=ROOT / "outputs/models/diagnoser.pt")
    p.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.test.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    torch.set_num_threads(4)

    model = Diagnoser()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    clips, snr_true, frac_true, speakers, is_val = load_manifest_data(
        args.manifest, args.limit)
    n = clips.shape[0]
    print(f"evaluating {n} clips from {args.manifest}")

    feats = log_mel_batch(clips)
    snr_pred, frac_pred = predict(model, feats)
    snr_pred_np = snr_pred.numpy()
    frac_pred_np = frac_pred.numpy()

    dsp_snr_arr = np.empty(n, dtype=np.float32)
    dsp_rel = np.empty(n, dtype=bool)
    dsp_clip_arr = np.empty(n, dtype=bool)
    for i in range(n):
        dsp_snr_arr[i], dsp_rel[i] = dsp_snr(clips[i])
        dsp_clip_arr[i] = dsp_clip_present(clips[i])

    # SNR regression (dB). The DSP estimate uses a different SNR definition, so
    # it carries a systematic bias; both are reported against the same label.
    mae_snr_model = float(np.mean(np.abs(snr_pred_np - snr_true)))
    mae_snr_dsp = float(np.mean(np.abs(dsp_snr_arr - snr_true)))
    mae_clip_model = float(np.mean(np.abs(frac_pred_np - frac_true)))

    # Binary clipping detection (F1). Model threshold vs DSP plateau detector.
    clip_bin_true = (frac_true >= 0.001).astype(int)
    clip_bin_model = (frac_pred_np >= 0.001).astype(int)
    clip_bin_dsp = dsp_clip_arr.astype(int)
    cls_model = binary_metrics(clip_bin_model, clip_bin_true)
    cls_dsp = binary_metrics(clip_bin_dsp, clip_bin_true)

    # Pearson correlation of each predictor with the true label (a rough but
    # readable "how informative is this estimator" number).
    corr_snr_model = float(np.corrcoef(snr_pred_np, snr_true)[0, 1])
    corr_snr_dsp = float(np.corrcoef(dsp_snr_arr, snr_true)[0, 1])

    print("\n== Diagnoser vs DSP baseline (ground truth = the labels we built) ==")
    print(f"  SNR MAE        model {mae_snr_model:.2f} dB   "
          f"DSP {mae_snr_dsp:.2f} dB   (lower is better)")
    print(f"  SNR corr       model {corr_snr_model:.3f}   "
          f"DSP {corr_snr_dsp:.3f}   (higher is better)")
    print(f"  clip MAE       model {mae_clip_model:.4f}")
    print(f"  clip F1        model {cls_model['f1']:.3f}   "
          f"DSP {cls_dsp['f1']:.3f}")
    print(f"  clip P/R       model {cls_model['precision']:.3f}/{cls_model['recall']:.3f}   "
          f"DSP {cls_dsp['precision']:.3f}/{cls_dsp['recall']:.3f}")
    print(f"  clip positive  {cls_model['positive_rate']:.1%}")

    result = {
        "split": "test" if "test" in args.manifest.stem else args.manifest.stem,
        "n": n,
        "speakers": sorted({s for s in speakers}),
        "mae_snr_model_db": round(mae_snr_model, 3),
        "mae_snr_dsp_db": round(mae_snr_dsp, 3),
        "corr_snr_model": round(corr_snr_model, 4),
        "corr_snr_dsp": round(corr_snr_dsp, 4),
        "mae_clip_model": round(mae_clip_model, 5),
        "clip_cls_model": {k: round(v, 4) for k, v in cls_model.items()},
        "clip_cls_dsp": {k: round(v, 4) for k, v in cls_dsp.items()},
    }

    out = args.out or OUT_DIR / f"test_eval_{args.manifest.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
