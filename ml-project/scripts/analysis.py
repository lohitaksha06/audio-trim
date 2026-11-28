#!/usr/bin/env python3
"""Dimensionality, correlation and class-separability analysis.

Everything here is numpy + torch; scikit-learn is optional and only used for
t-SNE. PCA is implemented with ``numpy.linalg.eigh`` so no scikit-learn is
needed for the dimensionality reduction itself.

    python scripts/analysis.py --smoke
    python scripts/analysis.py --manifest data/manifest.json --limit 300

What it does
------------
1. Extracts the same log-mel features the diagnoser trains on
   (``train_diagnoser.log_mel_batch``).
2. PCA to 2D, reporting the explained-variance ratio of the top components.
   For D > N the exact dual form is used: eigh on the N x N sample-space
   covariance, whose non-zero eigenvalues and principal directions match the
   D x D covariance exactly. It is still ``numpy.linalg.eigh``, not sklearn.
3. t-SNE on the top-50 PCA components if scikit-learn is installed. If it is
   not, the script says so and falls back to the 2D PCA projection (which is
   already computed). It never hand-rolls t-SNE.
4. Pearson correlations between six hand features and the two labels.
5. Class separability: 2D projections coloured by clipped/not and by SNR bin,
   Fisher ratios, and a nearest-centroid clip classifier scored against
   ``clipped_fraction >= 0.001`` (a separability probe, not a trained model).

Matplotlib is optional. When present, scatter plots go to ``outputs/figures/``;
the 2D coordinates are always printed as text as well.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_diagnoser import (  # noqa: E402
    N_MELS,
    SR,
    clip_classification_metrics,
    load_manifest_data,
    log_mel_batch,
    make_smoke_data,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

CLIP_THRESHOLD = 0.001
SNR_EDGES = (10.0, 20.0)  # bins: <10, 10-20, >=20 dB


# ---------------------------------------------------------------------------
# PCA via numpy.linalg.eigh
# ---------------------------------------------------------------------------

def pca_eigh(X: np.ndarray, n_components: int = 2):
    """Principal components with ``numpy.linalg.eigh`` only.

    Returns (scores, explained_variance_ratio, components).

    When D <= N the covariance is formed directly and eigendecomposed. When
    D > N (the usual case: 6464 log-mel values, a few hundred clips) the
    N x N sample-space covariance is used instead. That is the exact dual:
    the non-zero eigenvalues are identical and the principal directions are
    recovered as ``X^T v``. Either way the eigensolver is numpy, not sklearn.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    if n < 2:
        raise ValueError("PCA needs at least two rows")
    k = int(min(n_components, n - 1, d))
    Xc = X - X.mean(axis=0, keepdims=True)

    if d <= n:
        cov = Xc.T @ Xc / (n - 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        comps = evecs[:, order][:, :k]
        scores = Xc @ comps
    else:
        gram = Xc @ Xc.T / (n - 1)
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1]
        evals = np.clip(evals[order], 0.0, None)
        comps = Xc.T @ evecs[:, order][:, :k]
        norms = np.linalg.norm(comps, axis=0, keepdims=True)
        comps = comps / np.maximum(norms, 1e-12)
        scores = Xc @ comps

    # Stable sign convention: largest-magnitude loading points positive.
    for j in range(comps.shape[1]):
        pivot = int(np.argmax(np.abs(comps[:, j])))
        if comps[pivot, j] < 0:
            comps[:, j] *= -1.0
            scores[:, j] *= -1.0

    total = float(evals.sum())
    ratio = evals / total if total > 0 else np.zeros_like(evals)
    return scores, ratio, comps


# ---------------------------------------------------------------------------
# t-SNE (optional scikit-learn, never hand-rolled)
# ---------------------------------------------------------------------------

def compute_tsne(pca_scores: np.ndarray, seed: int):
    """Return (coords, note). coords is None when sklearn is unavailable."""
    try:
        from sklearn.manifold import TSNE  # noqa: PLC0415 - optional dependency
    except Exception as exc:  # noqa: BLE001 - any import failure means fallback
        return None, (f"t-SNE requires scikit-learn (import failed: "
                      f"{type(exc).__name__}). Showing the 2D PCA projection "
                      f"instead. Install with: pip install scikit-learn")
    try:
        n = pca_scores.shape[0]
        perplexity = float(min(30.0, max(5.0, (n - 1) / 3.0)))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=500,
            random_state=seed,
        )
        return tsne.fit_transform(pca_scores), None
    except Exception as exc:  # noqa: BLE001 - fall back rather than crash
        return None, (f"t-SNE failed ({type(exc).__name__}: {exc}). "
                      f"Showing the 2D PCA projection instead.")


# ---------------------------------------------------------------------------
# Hand features and correlation
# ---------------------------------------------------------------------------

def hand_features(clips: np.ndarray, sr: int = SR) -> dict[str, np.ndarray]:
    """Six interpretable per-clip features. All numpy, all cheap."""
    x = np.asarray(clips, dtype=np.float64)
    n, length = x.shape

    frame = 400  # 25 ms at 16 kHz
    nf = max(1, length // frame)
    frames = x[:, :nf * frame].reshape(n, nf, frame)

    signs = np.signbit(frames).astype(np.int8)
    zcr = np.mean(np.abs(np.diff(signs, axis=2)), axis=2).mean(axis=1)

    window = np.hanning(length)
    spectrum = np.abs(np.fft.rfft(x * window, axis=1))
    freqs = np.fft.rfftfreq(length, 1.0 / sr)
    centroid = (spectrum * freqs).sum(axis=1) / (spectrum.sum(axis=1) + 1e-12)

    peak = np.max(np.abs(x), axis=1)
    rms = np.sqrt(np.mean(x ** 2, axis=1) + 1e-12)
    return {
        "rms": rms,
        "zcr": zcr,
        "centroid_hz": centroid,
        "peak": peak,
        "crest": peak / (rms + 1e-12),
        "dc_offset": x.mean(axis=1),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def correlation_matrix(features: dict[str, np.ndarray], snr: np.ndarray,
                       frac: np.ndarray):
    names = list(features.keys()) + ["snr_db", "clipped_fraction"]
    cols = [features[k] for k in features] + [snr, frac]
    m = np.eye(len(cols), dtype=np.float64)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = pearson(cols[i], cols[j])
            m[i, j] = m[j, i] = r
    return names, m


# ---------------------------------------------------------------------------
# Separability
# ---------------------------------------------------------------------------

def fisher_ratio(Y: np.ndarray, labels: np.ndarray) -> float:
    """Between-class scatter / within-class scatter in the 2D projection."""
    overall = Y.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in np.unique(labels):
        Yc = Y[labels == c]
        mu = Yc.mean(axis=0)
        between += len(Yc) * float(((mu - overall) ** 2).sum())
        within += float(((Yc - mu) ** 2).sum())
    return between / max(within, 1e-12)


def nearest_centroid_predict(Y: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = np.unique(labels)
    centroids = np.stack([Y[labels == c].mean(axis=0) for c in classes])
    dist = ((Y[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    return classes[dist.argmin(axis=1)]


def snr_bin_labels(snr: np.ndarray) -> np.ndarray:
    return np.digitize(snr, SNR_EDGES).astype(int)


# ---------------------------------------------------------------------------
# Figures (optional matplotlib)
# ---------------------------------------------------------------------------

def save_scatter(Y: np.ndarray, values: np.ndarray, kind: str,
                 path: Path, seed: int = 0, projection: str = "PCA") -> str | None:
    """Save one coloured scatter. Returns None on success, else a reason."""
    try:
        import matplotlib  # noqa: PLC0415 - optional dependency
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return f"matplotlib not available ({type(exc).__name__})"

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    if kind == "clip":
        clipped = values.astype(bool)
        groups = [
            (~clipped, "not clipped", "#4C78A8"),
            (clipped, "clipped (>= 0.001)", "#E45756"),
        ]
        for mask, label, colour in groups:
            if mask.any():
                ax.scatter(Y[mask, 0], Y[mask, 1], s=14, alpha=0.75,
                           label=label, color=colour, edgecolors="none")
        ax.set_title(f"Log-mel {projection} - clipped vs not")
    else:
        bins = values.astype(int)
        groups = [
            (0, "SNR < 10 dB", "#4C78A8"),
            (1, "10-20 dB", "#F2A93B"),
            (2, ">= 20 dB", "#54A24B"),
        ]
        for b, label, colour in groups:
            mask = bins == b
            if mask.any():
                ax.scatter(Y[mask, 0], Y[mask, 1], s=14, alpha=0.75,
                           label=label, color=colour, edgecolors="none")
        ax.set_title(f"Log-mel {projection} - SNR bins")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_coords(name: str, Y: np.ndarray, snr: np.ndarray,
                  clipped: np.ndarray, n: int = 10) -> list[str]:
    lines = [f"  {name} 2D coordinates (first {min(n, len(Y))}):",
             "    idx          x          y     snr_db  clipped"]
    for i in range(min(n, len(Y))):
        lines.append(f"    {i:>3d}  {Y[i, 0]:>9.4f}  {Y[i, 1]:>9.4f}  "
                     f"{snr[i]:>7.2f}  {str(bool(clipped[i])):>7s}")
    return lines


def run_analysis(clips: np.ndarray, snr: np.ndarray, frac: np.ndarray, *,
                 seed: int = 7, prefix: str = "", fig_dir: Path = FIG_DIR,
                 max_figure_points: int = 3000) -> tuple[str, dict]:
    """Run every analysis and return (markdown, json-safe summary)."""
    n = len(clips)
    snr = np.asarray(snr, dtype=np.float64)
    frac = np.asarray(frac, dtype=np.float64)
    clipped = frac >= CLIP_THRESHOLD
    snr_bins = snr_bin_labels(snr)

    feats = log_mel_batch(clips)
    X = feats.numpy().reshape(n, -1).astype(np.float64)

    # ---- PCA -------------------------------------------------------------
    scores, ratio, _ = pca_eigh(X, n_components=50)
    cum = np.cumsum(ratio)
    top = ratio[:2]
    Y_pca = scores[:, :2]

    # ---- t-SNE -----------------------------------------------------------
    Y_tsne, tsne_note = compute_tsne(scores, seed)
    projection_name = "t-SNE" if Y_tsne is not None else "PCA"
    Y_plot = Y_tsne if Y_tsne is not None else Y_pca

    # ---- correlation -----------------------------------------------------
    features = hand_features(clips)
    corr_names, corr = correlation_matrix(features, snr, frac)

    # ---- separability ----------------------------------------------------
    clip_probe = nearest_centroid_predict(Y_pca, clipped.astype(int))
    clip_metrics = clip_classification_metrics(
        clip_probe.astype(np.float64), clipped.astype(np.float64), 0.5)
    snr_probe = nearest_centroid_predict(Y_pca, snr_bins)
    snr_acc = float(np.mean(snr_probe == snr_bins))

    # ---- figures ---------------------------------------------------------
    fig_paths: list[str] = []
    fig_notes: list[str] = []
    if n <= max_figure_points:
        tag = f"{prefix}analysis" if prefix else "analysis"
        for proj_name, proj in (("pca", Y_pca), ("tsne", Y_tsne)):
            if proj is None:
                continue
            p1 = fig_dir / f"{tag}_{proj_name}_clip.png"
            p2 = fig_dir / f"{tag}_{proj_name}_snr.png"
            for path, vals, kind in ((p1, clipped, "clip"), (p2, snr_bins, "snr")):
                note = save_scatter(proj, vals, kind, path, seed)
                if note is None:
                    try:
                        fig_paths.append(str(path.relative_to(ROOT)))
                    except ValueError:
                        fig_paths.append(str(path))
                else:
                    fig_notes.append(note)

    # ---- markdown --------------------------------------------------------
    md: list[str] = []
    md.append("# Dimensionality, correlation and separability")
    md.append("")
    md.append(f"- clips: {n}")
    md.append(f"- log-mel features: {X.shape[1]} values per clip "
              f"({N_MELS} mels x {feats.shape[-1]} frames)")
    md.append(f"- projection: {projection_name}")
    md.append("")

    md.append("## PCA (numpy.linalg.eigh)")
    md.append(f"- component 1 explains {ratio[0]:.1%} of variance")
    md.append(f"- component 2 explains {ratio[1]:.1%} of variance")
    md.append(f"- top-2 together: {cum[1]:.1%}")
    for k in (5, 10, 20, 50):
        if k <= len(ratio):
            md.append(f"- top-{k}: {cum[k - 1]:.1%}")
    md.append("")

    md.append("## t-SNE")
    if Y_tsne is None:
        md.append(f"- {tsne_note}")
    else:
        md.append("- scikit-learn TSNE on the top-50 PCA components "
                  "(perplexity capped at 30, 500 iterations)")
    md.append("")

    md.append("## Pearson correlation matrix")
    header = "  " + " ".join(f"{name[:10]:>10s}" for name in corr_names)
    md.append(header)
    for i, name in enumerate(corr_names):
        row = " ".join(f"{corr[i, j]:>10.3f}" for j in range(len(corr_names)))
        md.append(f"  {name[:10]:>10s} {row}")
    md.append("")
    md.append("  Labels are the last two rows/columns: snr_db and "
              "clipped_fraction.")
    md.append("")

    md.append("## Class separability")
    md.append(f"- clip class balance: {int(clipped.sum())}/{n} clipped "
              f"({clipped.mean():.1%}) at fraction >= {CLIP_THRESHOLD}")
    md.append(f"- Fisher ratio, clipped vs not (2D PCA): "
              f"{fisher_ratio(Y_pca, clipped.astype(int)):.3f}")
    md.append(f"- Fisher ratio, three SNR bins (2D PCA): "
              f"{fisher_ratio(Y_pca, snr_bins):.3f}")
    md.append(f"- nearest-centroid clip probe (in-sample): "
              f"P {clip_metrics['precision']:.3f}  "
              f"R {clip_metrics['recall']:.3f}  "
              f"F1 {clip_metrics['f1']:.3f}  "
              f"acc {clip_metrics['accuracy']:.3f}")
    md.append(f"- nearest-centroid SNR-bin probe (in-sample): "
              f"accuracy {snr_acc:.3f} over 3 bins")
    md.append("")
    md.append("  The probes are a separability check, not a trained model: "
              "centroids are fit on the same points they score. They answer "
              "'is the class visible in 2D?', not 'how well does it generalise?'")
    md.append("")

    md.extend(_print_coords(projection_name, Y_plot, snr, clipped))
    md.append("")

    md.append("## Figures")
    if fig_paths:
        for p in fig_paths:
            md.append(f"- {p}")
    else:
        reason = fig_notes[0] if fig_notes else "no projection to plot"
        md.append(f"- not saved: {reason}")
    md.append("")

    summary = {
        "n_clips": int(n),
        "feature_dim": int(X.shape[1]),
        "projection": projection_name,
        "pca": {
            "explained_variance_ratio": [float(x) for x in ratio[:10]],
            "cumulative": {str(k): float(cum[k - 1])
                           for k in (2, 5, 10, 20, 50) if k <= len(ratio)},
        },
        "tsne": {"available": Y_tsne is not None, "note": tsne_note},
        "correlation": {
            "names": corr_names,
            "matrix": [[float(x) for x in row] for row in corr],
        },
        "separability": {
            "clip": clip_metrics,
            "fisher_clip": fisher_ratio(Y_pca, clipped.astype(int)),
            "fisher_snr_bins": fisher_ratio(Y_pca, snr_bins),
            "snr_bin_accuracy": snr_acc,
            "snr_bin_counts": {str(b): int((snr_bins == b).sum())
                               for b in np.unique(snr_bins)},
        },
        "figures": fig_paths,
        "figure_notes": fig_notes,
    }
    return "\n".join(md), summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="PCA / t-SNE / correlation / separability on log-mel.")
    p.add_argument("--smoke", action="store_true",
                   help="synthesise ~300 clips in memory, no data")
    p.add_argument("--manifest", type=Path, default=None,
                   help="manifest from preprocess.py")
    p.add_argument("--limit", type=int, default=300,
                   help="max clips from a real manifest (default 300)")
    p.add_argument("--count", type=int, default=300,
                   help="smoke clips (default 300)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--figures-dir", type=Path, default=FIG_DIR)
    p.add_argument("--report-out", type=Path, default=None)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.smoke:
        clips, snr, frac = make_smoke_data(args.count, args.seed)
        source = f"synthetic ({args.count} in-memory clips)"
        prefix = "smoke_"
    else:
        if not args.manifest:
            p.print_help()
            print("\nNothing to do: pass --smoke or --manifest <path>.")
            return 0
        clips, snr, frac, _speakers, _is_val = load_manifest_data(
            args.manifest, args.limit)
        source = f"{args.manifest} ({len(clips)} clips)"
        prefix = ""

    text, summary = run_analysis(
        clips, snr, frac, seed=args.seed, prefix=prefix,
        fig_dir=args.figures_dir)
    print(text)
    print(f"\nsource  {source}")

    suffix = "_smoke" if args.smoke else ""
    report_out = args.report_out or (OUT_DIR / f"analysis_report{suffix}.md")
    json_out = args.json_out or (OUT_DIR / f"analysis_summary{suffix}.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(text + "\n", encoding="utf-8")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"report  {report_out}")
    print(f"summary {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
