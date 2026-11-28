#!/usr/bin/env python3
"""Assemble ``outputs/RESULTS.md`` from the EDA, metrics and analysis outputs.

Sections rendered:

  (a) results by split (train / val / test)
  (b) baseline comparison
  (c) seen-vs-unseen gap
  (d) clip-classification metrics
  (e) learning-curve numbers
  (f) problem and motivation
  (g) literature review (short; full text in docs/LITERATURE_REVIEW.md)
  (h) explainability
  (i) limitations

    python scripts/report.py
    python scripts/report.py --smoke      # synthetic numbers, template proof

The smoke run writes ``outputs/smoke/RESULTS.md`` so it never overwrites a
real report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eda import make_smoke_manifest, summarise_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
METRICS_DIR = OUT_DIR / "metrics"
LIT_PATH = ROOT / "docs" / "LITERATURE_REVIEW.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _last(seq, default=None):
    if isinstance(seq, (list, tuple)) and seq:
        return seq[-1]
    return default


def _pct(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def load_runs(metrics_dir: Path) -> list[dict]:
    """Every metrics JSON that looks like a train_diagnoser.py history."""
    runs: list[dict] = []
    if not metrics_dir.is_dir():
        return runs
    for path in sorted(metrics_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - skip a corrupt/foreign JSON
            continue
        if not isinstance(data, dict) or "val_mae_snr" not in data:
            continue
        runs.append({
            "file": path.name,
            "data": data,
            "mtime": path.stat().st_mtime,
        })
    # Prefer real runs over smoke runs, then the most recent.
    runs.sort(key=lambda r: ("smoke" in r["file"].lower(), -r["mtime"]))
    return runs


def synthetic_metrics(seed: int = 7) -> dict:
    """Numbers with the same shape as train_diagnoser.py's history.

    Used only by --smoke, so the report template is exercised without data.
    """
    return {
        "epoch": [1, 2, 3, 4],
        "train_loss": [1.0176, 0.3112, 0.1942, 0.1822],
        "val_loss": [0.3015, 0.3372, 0.0988, 0.1559],
        "val_mae_snr": [7.6346, 9.0746, 1.9101, 3.6058],
        "val_mae_clip": [0.047029, 0.034682, 0.035104, 0.035749],
        "seen_mae_snr": 3.3677,
        "unseen_mae_snr": 3.6058,
        "seen_mae_clip": 0.039287,
        "unseen_mae_clip": 0.035749,
        "baseline_mae_snr": 7.0527,
        "baseline_mae_clip": 0.041954,
        "n_params": 154978,
        "n_train": 240,
        "n_val": 60,
        "clip_cls": {
            "positive_rate": 0.9333333333333333,
            "precision": 0.9310344827586207,
            "recall": 0.9642857142857143,
            "f1": 0.9473684210526316,
            "accuracy": 0.9,
        },
        "synthetic": True,
        "note": "Synthetic smoke numbers, not a real result.",
        "seed": seed,
    }


def excerpt(text: str | None, max_lines: int = 45, path_hint: str = "") -> str:
    if not text:
        return ""
    lines = text.strip().splitlines()
    if len(lines) <= max_lines:
        return text.strip()
    more = len(lines) - max_lines
    tail = f"\n\n... ({more} more lines"
    tail += f"; full text in {path_hint})" if path_hint else ")"
    return "\n".join(lines[:max_lines]) + tail


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def eda_section(summary: dict | None, source: str) -> list[str]:
    lines = ["## 3. Data (EDA summary)", ""]
    if summary is None:
        lines.append("No manifest found. Run `python scripts/preprocess.py --demo` "
                     "or `python scripts/eda.py --smoke`.")
        lines.append("")
        return lines

    s = summary
    split_txt = " / ".join(f"{k} {v}" for k, v in s["split_counts"].items())
    snr = s["snr"]
    frac = s["clipped_fraction"]
    dur = s["duration"]
    leak = s["leakage"]
    dc = s["dc"]

    lines.append(f"Source: `{source}`.")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Tracks | {s['num_tracks']} |")
    lines.append(f"| Splits | {split_txt} |")
    lines.append(f"| Speakers | {s['n_speakers']} |")
    lines.append(f"| Clipped tracks (>= {s['clip_threshold']}) | "
                 f"{s['clipped_count']} ({s['clip_rate']:.1%}) |")
    if snr.get("n"):
        lines.append(f"| SNR dB (mean / median / range) | "
                     f"{snr['mean']:.2f} / {snr['median']:.2f} / "
                     f"{snr['min']:.2f}-{snr['max']:.2f} |")
    if frac.get("n"):
        lines.append(f"| Clipped fraction (mean / max) | "
                     f"{frac['mean']:.4f} / {frac['max']:.4f} |")
    if dur.get("n"):
        lines.append(f"| Duration s (mean / min / max) | "
                     f"{dur['mean']:.2f} / {dur['min']:.2f} / {dur['max']:.2f} |")
    if s["noise_files"] is None:
        lines.append("| Noise files | not recorded in this manifest |")
    else:
        lines.append(f"| Noise files | {s['n_noise_files']} distinct |")
    if dc is None:
        lines.append("| DC offset | not measured (no audio reachable) |")
    else:
        lines.append(f"| DC offset (mean / max abs) | "
                     f"{dc['mean']:+.5f} / {dc['max_abs']:.5f} |")
    if leak["ok"]:
        lines.append(f"| Leakage check | PASS - {leak['n_speakers']} speakers, "
                     f"none in two splits |")
    else:
        lines.append(f"| Leakage check | **FAIL** - "
                     f"{leak['speakers_in_multiple_splits']} |")
    lines.append("")
    lines.append("Full ASCII histograms: `outputs/eda_report.md`.")
    lines.append("")
    return lines


def results_section(metrics: dict | None) -> list[str]:
    lines = ["## 4. Results by split", ""]
    if metrics is None:
        lines.append("No metrics JSON found in `outputs/metrics/`. "
                     "Run `python scripts/train_diagnoser.py --smoke` first.")
        lines.append("")
        return lines

    m = metrics
    n_train = m.get("n_train", "-")
    n_val = m.get("n_val", "-")
    seen_snr = m.get("seen_mae_snr", m.get("val_mae_snr"))
    seen_clip = m.get("seen_mae_clip", m.get("val_mae_clip"))
    unseen_snr = m.get("unseen_mae_snr", m.get("val_mae_snr"))
    unseen_clip = m.get("unseen_mae_clip", m.get("val_mae_clip"))
    train_loss = _last(m.get("train_loss"))
    val_loss = _last(m.get("val_loss"))

    lines.append("| Split | n | MAE SNR (dB) | MAE clipped fraction | Loss |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(f"| Train (seen speakers) | {n_train} | {_fmt(seen_snr)} | "
                 f"{_fmt(seen_clip, 4)} | {_fmt(train_loss, 4)} |")
    lines.append(f"| Validation (unseen speakers) | {n_val} | {_fmt(unseen_snr)} | "
                 f"{_fmt(unseen_clip, 4)} | {_fmt(val_loss, 4)} |")
    lines.append("| Test | - | not run | not run | - |")
    lines.append("")
    lines.append("The manifest has no test split. The official VoiceBank-DEMAND "
                 "test speakers (2 unseen speakers) are reserved for a later run; "
                 "the validation split here is speaker-disjoint by construction.")
    lines.append("")
    return lines


def baseline_section(metrics: dict | None) -> list[str]:
    lines = ["## 5. Baseline comparison", ""]
    if metrics is None:
        lines.append("No metrics available.")
        lines.append("")
        return lines

    m = metrics
    base_snr = m.get("baseline_mae_snr")
    base_clip = m.get("baseline_mae_clip")
    model_snr = m.get("unseen_mae_snr", m.get("val_mae_snr"))
    model_clip = m.get("unseen_mae_clip", m.get("val_mae_clip"))

    def improvement(base, model):
        if base is None or model is None or float(base) == 0:
            return "-"
        return f"{(float(base) - float(model)) / float(base) * 100:+.1f}%"

    lines.append("Baseline: always predict the training mean. "
                 "Model: unseen-speaker validation score.")
    lines.append("")
    lines.append("| Target | Baseline MAE | Diagnoser MAE | Relative improvement |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| SNR (dB) | {_fmt(base_snr)} | {_fmt(model_snr)} | "
                 f"{improvement(base_snr, model_snr)} |")
    lines.append(f"| Clipped fraction | {_fmt(base_clip, 4)} | "
                 f"{_fmt(model_clip, 4)} | {improvement(base_clip, model_clip)} |")
    lines.append("")
    return lines


def gap_section(metrics: dict | None) -> list[str]:
    lines = ["## 6. Seen-vs-unseen gap", ""]
    if metrics is None:
        lines.append("No metrics available.")
        lines.append("")
        return lines

    m = metrics
    seen_snr = m.get("seen_mae_snr", m.get("val_mae_snr"))
    unseen_snr = m.get("unseen_mae_snr", m.get("val_mae_snr"))
    seen_clip = m.get("seen_mae_clip", m.get("val_mae_clip"))
    unseen_clip = m.get("unseen_mae_clip", m.get("val_mae_clip"))

    def gap(a, b):
        if a is None or b is None:
            return "-"
        return f"{float(b) - float(a):+.4f}"

    lines.append("| Target | Seen speakers | Unseen speakers | Gap (unseen - seen) |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| MAE SNR (dB) | {_fmt(seen_snr)} | {_fmt(unseen_snr)} | "
                 f"{gap(seen_snr, unseen_snr)} |")
    lines.append(f"| MAE clipped fraction | {_fmt(seen_clip, 4)} | "
                 f"{_fmt(unseen_clip, 4)} | {gap(seen_clip, unseen_clip)} |")
    lines.append("")
    lines.append("A small gap means the model learned the damage, not the voice. "
                 "A large positive gap is the overfitting signal.")
    lines.append("")
    return lines


def clip_section(metrics: dict | None) -> list[str]:
    lines = ["## 7. Clip classification", ""]
    if metrics is None:
        lines.append("No metrics available.")
        lines.append("")
        return lines
    c = metrics.get("clip_cls") or {}
    if not c:
        lines.append("No `clip_cls` block in this metrics file.")
        lines.append("")
        return lines
    lines.append("Binary task: `clipped_fraction >= 0.001`. "
                 "Computed from the model's clipped-fraction head.")
    lines.append("")
    lines.append("| Precision | Recall | F1 | Accuracy | Positive rate |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    lines.append(f"| {_fmt(c.get('precision'), 3)} | {_fmt(c.get('recall'), 3)} | "
                 f"{_fmt(c.get('f1'), 3)} | {_fmt(c.get('accuracy'), 3)} | "
                 f"{_pct(c.get('positive_rate'))} |")
    lines.append("")
    return lines


def curve_section(metrics: dict | None) -> list[str]:
    lines = ["## 8. Learning curve", ""]
    if metrics is None:
        lines.append("No metrics available.")
        lines.append("")
        return lines
    m = metrics
    epochs = m.get("epoch") or []
    if not epochs:
        lines.append("No per-epoch history in this metrics file.")
        lines.append("")
        return lines
    tr = m.get("train_loss") or []
    va = m.get("val_loss") or []
    ms = m.get("val_mae_snr") or []
    mc = m.get("val_mae_clip") or []

    lines.append("| Epoch | Train loss | Val loss | Val MAE SNR (dB) | "
                 "Val MAE clipped |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for i, ep in enumerate(epochs):
        lines.append(f"| {ep} | {_fmt(tr[i] if i < len(tr) else None, 4)} | "
                     f"{_fmt(va[i] if i < len(va) else None, 4)} | "
                     f"{_fmt(ms[i] if i < len(ms) else None)} | "
                     f"{_fmt(mc[i] if i < len(mc) else None, 4)} |")
    lines.append("")
    if va:
        best = int(min(range(len(va)), key=lambda i: va[i])) + 1
        lines.append(f"- Best validation loss at epoch {best} "
                     f"({float(min(va)):.4f}).")
        lines.append(f"- Final train/val loss: {_fmt(_last(tr), 4)} / "
                     f"{_fmt(_last(va), 4)}.")
    lines.append("")
    return lines


def analysis_section(text: str | None) -> list[str]:
    lines = ["## 9. Dimensionality, correlation and separability", ""]
    if text:
        lines.append(excerpt(text, max_lines=50,
                             path_hint="outputs/analysis_report.md"))
    else:
        lines.append("Not run. `python scripts/analysis.py --smoke` (or "
                     "`--manifest data/manifest.json`) writes this section.")
    lines.append("")
    return lines


def explain_section(text: str | None) -> list[str]:
    lines = ["## 10. Explainability", ""]
    if text:
        lines.append(excerpt(text, max_lines=50,
                             path_hint="outputs/explainability_report.md"))
    else:
        lines.append("Not run. `python scripts/explain.py --smoke` (or "
                     "`--model outputs/models/diagnoser.pt`) writes this section.")
        lines.append("")
        lines.append("Planned methods: gradient saliency of the SNR head "
                     "(always available) and LIME (if installed).")
    lines.append("")
    return lines


def literature_section() -> list[str]:
    return [
        "## 2. Literature review (summary)",
        "",
        "- **Classical speech enhancement.** Spectral subtraction (Boll, 1979) "
        "and Wiener filtering (Lim & Oppenheim, 1979) estimate a noise spectrum "
        "from silence and subtract or weight it. Statistical MMSE estimators "
        "(Ephraim & Malah, 1984) replaced the ad-hoc subtraction with a "
        "cost function; the decision-directed a-priori SNR (Scalart & Filho, "
        "1996) is still a strong DSP baseline.",
        "- **Neural enhancement.** DNN regression on log-power spectra "
        "(Xu et al., 2014), fully convolutional / U-Net mask predictors "
        "(Park & Lee, 2017) and conditional GANs (SEGAN, Pascual et al., 2017) "
        "moved the field from hand-designed estimators to learned mappings.",
        "- **Benchmark.** VoiceBank-DEMAND (Valentini-Botinhao et al., 2016) "
        "pairs 28 training speakers with 2 unseen test speakers at 0, 5, 10 "
        "and 15 dB, using DEMAND noises (Thiemann et al., 2013). Its "
        "speaker-disjoint protocol is the reason our split is by speaker, not "
        "by utterance.",
        "- **Declipping.** Cubic-spline interpolation is the standard cheap "
        "baseline; sparse/co-sparse methods (A-SPADE, Kitic et al., 2015) and "
        "DNN declippers improve on it. The survey by Zaviska et al. (2021) "
        "compares the families and is the reference we follow for baselines.",
        "- **Explainability.** LIME (Ribeiro et al., 2016) fits a local "
        "interpretable surrogate; SHAP (Lundberg & Lee, 2017) uses Shapley "
        "values. Gradient saliency needs no package and is our guaranteed "
        "fallback.",
        "",
        "Full review with venues and honest confidence notes: "
        "`docs/LITERATURE_REVIEW.md`.",
        "",
    ]


def limitations_section(synthetic: bool) -> list[str]:
    lines = ["## 11. Limitations", ""]
    if synthetic:
        lines.append("> **This is a smoke report.** Every number below is "
                     "synthetic and exists only to prove the template. It is "
                     "not evidence about the model.")
        lines.append("")
    lines.extend([
        "- The manifest has **no held-out test split**. Validation is "
        "speaker-disjoint, but the official VoiceBank-DEMAND test speakers "
        "have not been run yet.",
        "- Splits are speaker-disjoint but **not noise-disjoint**. A noise "
        "recording can appear in both train and validation, so the score is "
        "optimistic for unseen noise types.",
        "- Our mixtures use **continuous SNR 0-30 dB** and clipping applied "
        "after mixing. That is deliberate (the labels are exact), but it means "
        "the numbers are **not comparable** to published PESQ/STOI results on "
        "the fixed 0/5/10/15 dB benchmark.",
        "- The diagnoser is ~155k parameters, trained on CPU with one seed. "
        "No seed variance, confidence interval or hyper-parameter sweep is "
        "reported.",
        "- Clipping is modelled as hard clipping at a random threshold. Real "
        "recordings may clip in the ADC, after resampling, or with soft "
        "limiting; those cases are not represented.",
        "- The denoiser, declipper and router are not evaluated here. This "
        "report covers the diagnoser only.",
        "- EDA histograms are ASCII; the 2D projections are PCA/t-SNE pictures, "
        "which are descriptive, not a statistical test.",
        "- t-SNE is optional (scikit-learn). When it is missing the analysis "
        "falls back to PCA and says so.",
        "- LIME and SHAP are optional. The guaranteed explanation is gradient "
        "saliency, which is a local linear approximation and can be noisy for "
        "a BatchNorm CNN.",
        "- DC offset is measured only for audio files that are reachable from "
        "the manifest; demo manifests use `memory://` paths and report no DC.",
        "",
    ])
    return lines


def reproduction_section(smoke: bool) -> list[str]:
    lines = ["## 12. Reproduction", ""]
    lines.append("```bash")
    lines.append("python scripts/preprocess.py --demo")
    lines.append("python scripts/train_diagnoser.py --smoke")
    lines.append("python scripts/eda.py --smoke")
    lines.append("python scripts/analysis.py --smoke")
    lines.append("python scripts/explain.py --smoke")
    lines.append("python scripts/report.py --smoke")
    lines.append("```")
    lines.append("")
    lines.append("Real runs replace `--smoke` with `--manifest data/manifest.json` "
                 "and, for explain.py, `--model outputs/models/diagnoser.pt`.")
    if smoke:
        lines.append("")
        lines.append("This file was produced by `--smoke` with synthetic data.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def render_results(metrics: dict | None, runs: list[dict],
                   eda_summary: dict | None, eda_source: str,
                   analysis_text: str | None, explain_text: str | None,
                   *, synthetic: bool = False) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Results - Fix my recording (diagnoser)")
    lines.append("")
    lines.append(f"_Generated {stamp} by `scripts/report.py`._")
    if synthetic:
        lines.append("")
        lines.append("> **SMOKE RUN - synthetic in-memory numbers.** "
                     "The layout is real; the values are not.")
    lines.append("")

    lines.append("## 1. Problem and motivation")
    lines.append("")
    lines.append("Recordings made on phones and laptops arrive damaged in two "
                 "common ways: steady background noise and hard clipping at the "
                 "ADC or after gain. The product already repairs them with "
                 "fixed signal processing, but it does not know **how much** "
                 "damage is present, so it cannot choose the right repair or "
                 "tell the user when a recording is beyond saving.")
    lines.append("")
    lines.append("This project trains small models from scratch to (1) diagnose "
                 "noise level and clipped fraction, (2) denoise, (3) declip and "
                 "(4) route a recording to the right repair chain. The "
                 "diagnoser is the first stage: its outputs are the router's "
                 "inputs. Everything is measured against a no-repair baseline, "
                 "the shipped spectral-gating chain and a cubic-spline "
                 "declipper, because a metric without its baseline is not a "
                 "result.")
    lines.append("")

    lines.extend(literature_section())
    lines.extend(eda_section(eda_summary, eda_source))
    lines.extend(results_section(metrics))
    lines.extend(baseline_section(metrics))
    lines.extend(gap_section(metrics))
    lines.extend(clip_section(metrics))
    lines.extend(curve_section(metrics))

    if runs:
        lines.append("### All metrics files found")
        lines.append("")
        lines.append("| File | n train | n val | Val MAE SNR | Val MAE clip | F1 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for r in runs:
            d = r["data"]
            c = d.get("clip_cls") or {}
            lines.append(f"| `{r['file']}` | {d.get('n_train', '-')} | "
                         f"{d.get('n_val', '-')} | "
                         f"{_fmt(d.get('unseen_mae_snr', d.get('val_mae_snr')))} | "
                         f"{_fmt(d.get('unseen_mae_clip', d.get('val_mae_clip')), 4)} | "
                         f"{_fmt(c.get('f1'), 3)} |")
        lines.append("")

    lines.extend(analysis_section(analysis_text))
    lines.extend(explain_section(explain_text))
    lines.extend(limitations_section(synthetic))
    lines.extend(reproduction_section(synthetic))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Assemble outputs/RESULTS.md from EDA, metrics and analysis.")
    p.add_argument("--smoke", action="store_true",
                   help="synthetic numbers; writes outputs/smoke/RESULTS.md")
    p.add_argument("--manifest", type=Path, default=None,
                   help="manifest for the EDA section")
    p.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.smoke:
        manifest, audio = make_smoke_manifest(count=12, seed=1234)
        eda_summary = summarise_manifest(manifest, audio)
        eda_source = "synthetic manifest in memory (smoke)"
        metrics = synthetic_metrics()
        runs = [{"file": "synthetic-smoke.json", "data": metrics}]
        analysis_text = read_text(OUT_DIR / "analysis_report_smoke.md")
        explain_text = read_text(OUT_DIR / "explainability_report_smoke.md")
        out = args.out or (OUT_DIR / "smoke" / "RESULTS.md")
        synthetic = True
    else:
        manifest_path = args.manifest
        if manifest_path is None:
            for candidate in (DATA_DIR / "manifest.json",
                              DATA_DIR / "manifest.demo.json"):
                if candidate.is_file():
                    manifest_path = candidate
                    break
        if manifest_path is not None and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            eda_summary = summarise_manifest(manifest)
            try:
                eda_source = str(manifest_path.relative_to(ROOT)).replace("\\", "/")
            except ValueError:
                eda_source = str(manifest_path)
        else:
            eda_summary = None
            eda_source = "(no manifest)"
        runs = load_runs(args.metrics_dir)
        metrics = runs[0]["data"] if runs else None
        analysis_text = read_text(OUT_DIR / "analysis_report.md")
        explain_text = read_text(OUT_DIR / "explainability_report.md")
        out = args.out or (OUT_DIR / "RESULTS.md")
        synthetic = False

    text = render_results(metrics, runs, eda_summary, eda_source,
                          analysis_text, explain_text, synthetic=synthetic)
    print(text)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nreport  {out}")
    if synthetic:
        print("note    synthetic smoke numbers; real report is outputs/RESULTS.md")
    if LIT_PATH.is_file():
        print(f"lit     {LIT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
