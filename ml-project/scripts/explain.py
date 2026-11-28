#!/usr/bin/env python3
"""Explain the diagnoser: gradient saliency (always) + LIME (if installed).

    python scripts/explain.py --smoke
    python scripts/explain.py --model outputs/models/diagnoser.pt \
                              --manifest data/manifest.json

Method 1 - gradient saliency, native PyTorch, no extra package
--------------------------------------------------------------
Backprop the SNR head output w.r.t. the input log-mel and take |gradient|.
This answers "which time-frequency cells does the SNR prediction depend on?"
and is reported as top mel bands (with approximate Hz), top time frames (with
seconds) and a concentration number. It always works.

Method 2 - LIME, optional
-------------------------
If ``lime`` is importable, a tabular LIME surrogate is fit on an 8x8 pooled
log-mel grid. The surrogate's predict function upsamples the grid back to the
model's input shape, so the explanation is still about this model. If lime is
missing the script says so and falls back to saliency only. Nothing is
installed by this script.

``--smoke`` trains a tiny diagnoser on synthetic clips in memory, then runs
both paths on four of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_diagnoser import (  # noqa: E402
    HOP,
    N_MELS,
    SR,
    Diagnoser,
    load_manifest_data,
    log_mel_batch,
    make_smoke_data,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

DEFAULT_MODEL = OUT_DIR / "models" / "diagnoser.pt"
FALLBACK_MODEL = OUT_DIR / "models" / "diagnoser_smoke.pt"


# ---------------------------------------------------------------------------
# Mel geometry (for human-readable band labels)
# ---------------------------------------------------------------------------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_edges(n_mels: int = N_MELS, sr: int = SR) -> np.ndarray:
    pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), n_mels + 2)
    return _mel_to_hz(pts)


def band_label(b: int, edges: np.ndarray) -> str:
    lo, hi = float(edges[b]), float(edges[b + 2])
    return f"mel {b:>2d} ({lo:>5.0f}-{hi:>5.0f} Hz)"


# ---------------------------------------------------------------------------
# Model loading / tiny training
# ---------------------------------------------------------------------------

def load_saved_model(model_path: Path, config_path: Path | None = None):
    if not model_path.is_file():
        return None, None, f"model not found: {model_path}"
    cfg_path = config_path or model_path.with_suffix(".json")
    cfg = {}
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model = Diagnoser(n_mels=int(cfg.get("n_mels", N_MELS)))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, cfg, None


def train_tiny(clips: np.ndarray, snr: np.ndarray, frac: np.ndarray, *,
               steps: int = 15, batch: int = 32, seed: int = 7):
    """A short optimisation pass so --smoke has a non-random model."""
    torch.manual_seed(seed)
    feats = log_mel_batch(clips)
    snr_t = torch.from_numpy(np.asarray(snr, dtype=np.float32))
    frac_t = torch.from_numpy(np.asarray(frac, dtype=np.float32))
    model = Diagnoser()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    n = feats.shape[0]
    last = float("nan")
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        s, c = model(feats[idx])
        loss = (F.l1_loss(s / 30.0, snr_t[idx] / 30.0)
                + F.l1_loss(c, frac_t[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    model.eval()
    return model, feats, last


# ---------------------------------------------------------------------------
# Gradient saliency
# ---------------------------------------------------------------------------

def gradient_saliency(model, feats: torch.Tensor, *, top_k: int = 5) -> dict:
    """|d SNR / d log-mel|, averaged over the batch."""
    model.eval()
    x = feats.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    snr_pred, clip_pred = model(x)
    snr_pred.sum().backward()

    grad = x.grad.detach().abs()                 # (N, 1, n_mels, frames)
    sal = grad.mean(dim=0)[0].cpu().numpy()      # (n_mels, frames)
    per_clip = grad.mean(dim=(1, 2, 3)).cpu().numpy()

    edges = mel_edges(sal.shape[0])
    freq_energy = sal.mean(axis=1)
    time_energy = sal.mean(axis=0)
    freq_share = freq_energy / (float(freq_energy.sum()) + 1e-12)
    time_share = time_energy / (float(time_energy.sum()) + 1e-12)

    bands = []
    for b in np.argsort(freq_energy)[::-1][:top_k]:
        bands.append({
            "mel": int(b),
            "hz_lo": float(edges[b]),
            "hz_hi": float(edges[b + 2]),
            "share": float(freq_share[b]),
            "label": band_label(int(b), edges),
        })
    frames = []
    for t in np.argsort(time_energy)[::-1][:top_k]:
        frames.append({
            "frame": int(t),
            "seconds": float(t * HOP / SR),
            "share": float(time_share[t]),
        })

    flat = np.sort(sal.ravel())[::-1]
    k = max(1, int(0.10 * flat.size))
    concentration = float(flat[:k].sum() / (flat.sum() + 1e-12))

    centres = 0.5 * (edges[1:-1] + edges[2:])
    low = float(freq_share[centres < 500.0].sum())
    high = float(freq_share[centres >= 2000.0].sum())
    mid = max(0.0, 1.0 - low - high)

    return {
        "saliency": sal,
        "per_clip": per_clip,
        "bands": bands,
        "frames": frames,
        "concentration_top10pct": concentration,
        "band_groups": {"low_lt500": low, "mid_500_2000": mid,
                        "high_ge2000": high},
        "predictions": {
            "snr_db": [float(v) for v in snr_pred.detach().cpu().numpy()],
            "clipped_fraction": [float(v) for v in clip_pred.detach().cpu().numpy()],
        },
    }


# ---------------------------------------------------------------------------
# LIME (optional)
# ---------------------------------------------------------------------------

def lime_explain(model, feats: torch.Tensor, *, grid: int = 8,
                 n_samples: int = 400, seed: int = 7):
    """Return (items, note). items is None when LIME is unavailable."""
    try:
        from lime.lime_tabular import LimeTabularExplainer  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - missing package is the expected path
        return None, "LIME not installed - run: pip install lime"

    try:
        feats = feats.detach()
        pooled = F.adaptive_avg_pool2d(feats, (grid, grid))
        pooled_np = pooled.numpy().reshape(feats.shape[0], -1).astype(np.float64)
        target_frames = int(feats.shape[-1])
        mel = int(feats.shape[2])

        def predict_fn(z):
            t = torch.from_numpy(np.asarray(z, dtype=np.float32))
            t = t.reshape(-1, 1, grid, grid)
            t = F.interpolate(t, size=(mel, target_frames), mode="bilinear",
                              align_corners=False)
            with torch.no_grad():
                s, _ = model(t)
            return s.numpy().reshape(-1, 1)

        names = [f"mel{b}_t{t}" for b in range(grid) for t in range(grid)]
        explainer = LimeTabularExplainer(
            pooled_np,
            mode="regression",
            feature_names=names,
            discretize_continuous=True,
            random_state=seed,
        )
        exp = explainer.explain_instance(
            pooled_np[0], predict_fn, num_features=10, num_samples=n_samples)
        return exp.as_list(), None
    except Exception as exc:  # noqa: BLE001 - never let an optional path crash
        return None, (f"LIME is installed but the explanation failed "
                      f"({type(exc).__name__}: {exc})")


def lime_item_region(item_name: str, mel: int, frames: int) -> str:
    """Turn 'mel3_t5 <= 0.12' into a readable band/time region."""
    try:
        head = item_name.split()[0]
        b_part, t_part = head.split("_")
        b = int(b_part.replace("mel", ""))
        t = int(t_part.replace("t", ""))
        edges = mel_edges(mel)
        lo, hi = int(edges[b]), int(edges[b + 2])
        sec = t / 8.0 * (frames * HOP / SR)
        return f"mel {b} ({lo}-{hi} Hz), time ~{sec:.2f}s"
    except Exception:  # noqa: BLE001
        return item_name


# ---------------------------------------------------------------------------
# Figure (optional)
# ---------------------------------------------------------------------------

def save_heatmap(sal: np.ndarray, path: Path) -> str | None:
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return f"matplotlib not available ({type(exc).__name__})"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    im = ax.imshow(sal, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("frame (100 per second)")
    ax.set_ylabel("mel band")
    ax.set_title("Gradient saliency of the SNR head")
    fig.colorbar(im, ax=ax, label="|d SNR / d log-mel|")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_explanation(model, feats: torch.Tensor, *, snr_true=None,
                    frac_true=None, model_label: str = "",
                    prefix: str = "", fig_dir: Path = FIG_DIR,
                    lime_samples: int = 400, seed: int = 7):
    sal = gradient_saliency(model, feats)
    n, _, mel, frames = feats.shape
    lime_items, lime_note = lime_explain(
        model, feats, n_samples=lime_samples, seed=seed)

    try:
        import shap  # noqa: F401,PLC0415 - availability check only
        shap_note = "SHAP is installed (not used by this script)."
    except Exception:  # noqa: BLE001
        shap_note = "SHAP not installed - optional: pip install shap"

    md: list[str] = []
    md.append("# Explainability - diagnoser")
    md.append("")
    md.append(f"- model: {model_label or 'in-memory tiny diagnoser'}")
    md.append(f"- inputs: {n} clips, log-mel {mel} x {frames}")
    md.append("- method: |d SNR_head / d input log-mel|, averaged over clips")
    md.append("")

    md.append("## Predictions on the explained clips")
    md.append("| clip | pred SNR (dB) | true SNR (dB) | pred clipped | true clipped |")
    md.append("| ---: | ---: | ---: | ---: | ---: |")
    for i in range(n):
        true_s = f"{snr_true[i]:.2f}" if snr_true is not None else "-"
        true_c = f"{frac_true[i]:.4f}" if frac_true is not None else "-"
        md.append(f"| {i} | {sal['predictions']['snr_db'][i]:.2f} | {true_s} | "
                  f"{sal['predictions']['clipped_fraction'][i]:.4f} | {true_c} |")
    md.append("")

    md.append("## Most influential mel bands")
    for b in sal["bands"]:
        md.append(f"- {b['label']}: {b['share']:.1%} of saliency")
    md.append("")

    md.append("## Most influential time regions")
    for f in sal["frames"]:
        md.append(f"- frame {f['frame']} (~{f['seconds']:.2f}s): "
                  f"{f['share']:.1%} of saliency")
    md.append("")

    md.append("## Concentration and band groups")
    md.append(f"- top 10% of time-frequency cells hold "
              f"{sal['concentration_top10pct']:.1%} of total saliency")
    g = sal["band_groups"]
    md.append(f"- low (<500 Hz): {g['low_lt500']:.1%}; "
              f"mid (500-2000 Hz): {g['mid_500_2000']:.1%}; "
              f"high (>=2000 Hz): {g['high_ge2000']:.1%}")
    md.append("")

    md.append("## LIME")
    if lime_items is None:
        md.append(f"- {lime_note}")
        md.append("- Falling back to gradient saliency only.")
    else:
        md.append("- tabular LIME on an 8x8 pooled log-mel grid "
                  "(regression surrogate for the SNR head)")
        for name, weight in lime_items[:10]:
            region = lime_item_region(name, mel, frames)
            md.append(f"- {region}: weight {weight:+.3f}")
    md.append("")

    md.append("## SHAP")
    md.append(f"- {shap_note}")
    md.append("")

    fig_path = fig_dir / f"{prefix}explain_saliency.png" if prefix else \
        fig_dir / "explain_saliency.png"
    fig_note = save_heatmap(sal["saliency"], fig_path)
    md.append("## Figure")
    if fig_note is None:
        try:
            fig_rel = str(fig_path.relative_to(ROOT))
        except ValueError:
            fig_rel = str(fig_path)
        md.append(f"- {fig_rel}")
    else:
        fig_rel = None
        md.append(f"- not saved: {fig_note}")
    md.append("")

    summary = {
        "n_clips": int(n),
        "feature_shape": [int(mel), int(frames)],
        "top_bands": sal["bands"],
        "top_frames": sal["frames"],
        "concentration_top10pct": sal["concentration_top10pct"],
        "band_groups": sal["band_groups"],
        "predictions": sal["predictions"],
        "lime": {
            "available": lime_items is not None,
            "note": lime_note,
            "items": ([{"item": name, "weight": float(w)}
                       for name, w in lime_items[:10]]
                      if lime_items is not None else []),
        },
        "figure": (fig_rel if fig_note is None else None),
        "figure_note": fig_note,
    }
    return "\n".join(md), summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Gradient saliency + optional LIME for the diagnoser.")
    p.add_argument("--smoke", action="store_true",
                   help="train a tiny model on synthetic clips in memory")
    p.add_argument("--model", type=Path, default=None,
                   help=f"saved state dict (default {DEFAULT_MODEL})")
    p.add_argument("--config", type=Path, default=None,
                   help="model JSON (default: same stem as --model)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="manifest for real explanation inputs")
    p.add_argument("--limit", type=int, default=4,
                   help="clips to explain (default 4)")
    p.add_argument("--count", type=int, default=96,
                   help="smoke training clips (default 96)")
    p.add_argument("--steps", type=int, default=15,
                   help="smoke training steps (default 15)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--report-out", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    snr_true = None
    frac_true = None
    source = ""

    if args.smoke:
        clips, snr, frac = make_smoke_data(args.count, args.seed)
        model, feats_all, last_loss = train_tiny(
            clips, snr, frac, steps=args.steps, seed=args.seed)
        model_label = f"in-memory Diagnoser, {args.steps} synthetic steps " \
                      f"(last loss {last_loss:.4f})"
        feats = feats_all[:args.limit].detach()
        snr_true = np.asarray(snr[:args.limit], dtype=np.float64)
        frac_true = np.asarray(frac[:args.limit], dtype=np.float64)
        source = f"synthetic ({args.count} in-memory clips, {args.limit} explained)"
        prefix = "smoke_"
    else:
        model_path = args.model or DEFAULT_MODEL
        used_fallback = False
        if not model_path.is_file() and args.model is None and FALLBACK_MODEL.is_file():
            model_path = FALLBACK_MODEL
            used_fallback = True
        model, cfg, err = load_saved_model(model_path, args.config)
        if err:
            print(f"{err}\nTrain one first: python scripts/train_diagnoser.py --smoke\n"
                  f"or point --model at a state dict.", file=sys.stderr)
            return 2
        model_label = f"{model_path} ({cfg.get('n_params', '?')} params)"
        if used_fallback:
            model_label += " [smoke model fallback]"

        if args.manifest:
            clips, snr, frac, _spk, _val = load_manifest_data(
                args.manifest, args.limit)
            snr_true = np.asarray(snr[:args.limit], dtype=np.float64)
            frac_true = np.asarray(frac[:args.limit], dtype=np.float64)
            source = f"{args.manifest} ({len(clips)} clips)"
        else:
            clips, snr, frac = make_smoke_data(max(args.limit, 8), args.seed)
            snr_true = np.asarray(snr[:args.limit], dtype=np.float64)
            frac_true = np.asarray(frac[:args.limit], dtype=np.float64)
            source = "synthetic (no --manifest given; explanation inputs only)"
        feats = log_mel_batch(clips[:args.limit]).detach()
        prefix = ""

    text, summary = run_explanation(
        model, feats, snr_true=snr_true, frac_true=frac_true,
        model_label=model_label, prefix=prefix, seed=args.seed)
    print(text)
    print(f"\nsource  {source}")

    suffix = "_smoke" if args.smoke else ""
    report_out = args.report_out or (OUT_DIR / f"explainability_report{suffix}.md")
    json_out = args.json_out or (OUT_DIR / f"explainability_summary{suffix}.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(text + "\n", encoding="utf-8")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"report  {report_out}")
    print(f"summary {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
