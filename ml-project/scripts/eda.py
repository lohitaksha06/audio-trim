#!/usr/bin/env python3
"""Exploratory data analysis for a preprocess.py manifest.

Reports, without matplotlib or scikit-learn:

  * track and clip counts, per-split counts,
  * per-speaker counts and a speaker leakage check,
  * duration / SNR / clipped-fraction distributions as ASCII histograms,
  * noise-file usage (when the manifest records it),
  * DC offset measured from the audio, not assumed.

    python scripts/eda.py --manifest data/manifest.json
    python scripts/eda.py --smoke          # synthetic manifest in memory

``--smoke`` builds ~40 one-second synthetic clips with the same helpers
preprocess.py uses, so every branch runs with no data and no downloads.
The report is printed and also written to ``outputs/eda_report[_smoke].md``
with a machine-readable summary in ``outputs/eda_summary[_smoke].json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# A track counts as clipped at the same threshold train_diagnoser.py uses for
# its binary clipping metrics.
CLIP_THRESHOLD = 0.001
DC_FLAG = 1e-3  # -60 dBFS; above this is worth mentioning


# ---------------------------------------------------------------------------
# Small reporting helpers
# ---------------------------------------------------------------------------

def ascii_hist(values, *, bins: int = 10, width: int = 40, label: str = "",
               digits: int = 3) -> str:
    """Return an ASCII histogram as text. No plotting library involved."""
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return f"  {label}: (no values)"

    counts, edges = np.histogram(a, bins=bins)
    peak = int(counts.max())
    lines = [f"  {label}  n={a.size}  bins={bins}"]
    for i, c in enumerate(counts):
        bar = "#" * int(round(c / peak * width)) if peak else ""
        lo, hi = float(edges[i]), float(edges[i + 1])
        lines.append(f"    [{lo:>9.{digits}f}, {hi:>9.{digits}f})  {c:>4d}  {bar}")
    return "\n".join(lines)


def describe(values) -> dict:
    """Min/max/mean/median/std/percentiles as plain Python floats."""
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def leakage_check(tracks: list[dict]) -> dict:
    """A speaker must live in exactly one split; train+val overlap is leakage."""
    by_speaker: dict[str, set[str]] = {}
    for t in tracks:
        by_speaker.setdefault(str(t.get("speaker", "?")), set()).add(
            str(t.get("split", "?")))
    multi = sorted(s for s, splits in by_speaker.items() if len(splits) > 1)
    both = sorted(s for s, splits in by_speaker.items()
                  if "train" in splits and "val" in splits)
    return {
        "ok": not multi,
        "n_speakers": len(by_speaker),
        "speakers_in_multiple_splits": multi,
        "speakers_in_train_and_val": both,
    }


def dc_offsets(manifest: dict, audio_lookup: dict | None = None) -> dict | None:
    """Mean sample value per clip, read from memory or from the manifest files."""
    from preprocess import load_audio  # numpy-only, no heavy imports

    rows: list[tuple[dict, float]] = []
    skipped = 0
    for t in manifest.get("tracks", []):
        x = None
        if audio_lookup is not None and t.get("id") in audio_lookup:
            x = audio_lookup[t["id"]]
        else:
            raw = str(t.get("file", ""))
            if raw and not raw.startswith("memory://"):
                path = _resolve(raw)
                if path.is_file():
                    try:
                        x, _ = load_audio(path)
                    except Exception:  # noqa: BLE001 - one bad file is not fatal
                        x = None
        if x is None:
            skipped += 1
            continue
        rows.append((t, float(np.mean(np.asarray(x, dtype=np.float64)))))

    if not rows:
        return None

    vals = np.asarray([v for _, v in rows], dtype=np.float64)
    per_split: dict[str, list[float]] = {}
    for t, v in rows:
        per_split.setdefault(str(t.get("split", "?")), []).append(v)
    return {
        "n_loaded": len(rows),
        "n_skipped": skipped,
        "mean": float(vals.mean()),
        "mean_abs": float(np.abs(vals).mean()),
        "max_abs": float(np.abs(vals).max()),
        "frac_above_flag": float(np.mean(np.abs(vals) > DC_FLAG)),
        "per_split": {k: describe(v) for k, v in sorted(per_split.items())},
    }


def summarise_manifest(manifest: dict, audio_lookup: dict | None = None) -> dict:
    """Compute every number the report needs. Returns JSON-safe Python types."""
    tracks = list(manifest.get("tracks", []))
    snr = [float(t["snr_db"]) for t in tracks if "snr_db" in t]
    frac = [float(t["clipped_fraction"]) for t in tracks if "clipped_fraction" in t]
    dur = [float(t["duration_s"]) for t in tracks if "duration_s" in t]

    split_counts: dict[str, int] = {}
    speaker_counts: dict[str, int] = {}
    noise_counts: dict[str, int] = {}
    has_noise = False
    for t in tracks:
        split_counts[str(t.get("split", "?"))] = split_counts.get(
            str(t.get("split", "?")), 0) + 1
        speaker_counts[str(t.get("speaker", "?"))] = speaker_counts.get(
            str(t.get("speaker", "?")), 0) + 1
        if "noise_file" in t:
            has_noise = True
            noise_counts[str(t["noise_file"])] = noise_counts.get(
                str(t["noise_file"]), 0) + 1

    clipped = [f >= CLIP_THRESHOLD for f in frac]
    return {
        "name": manifest.get("name", "(unnamed)"),
        "num_tracks": len(tracks),
        "declared_num_tracks": manifest.get("num_tracks"),
        "split_counts": dict(sorted(split_counts.items())),
        "declared_split_counts": manifest.get("split_counts"),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "n_speakers": len(speaker_counts),
        "snr": describe(snr),
        "clipped_fraction": describe(frac),
        "clipped_count": int(sum(clipped)),
        "clip_rate": float(np.mean(clipped)) if clipped else 0.0,
        "duration": describe(dur),
        "noise_files": dict(sorted(noise_counts.items())) if has_noise else None,
        "n_noise_files": len(noise_counts) if has_noise else 0,
        "dc": dc_offsets(manifest, audio_lookup),
        "leakage": leakage_check(tracks),
        "clip_threshold": CLIP_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(summary: dict) -> str:
    """Render the summary as ASCII-safe Markdown."""
    s = summary
    lines: list[str] = []
    lines.append(f"# EDA - {s['name']}")
    lines.append("")
    lines.append(f"- tracks: {s['num_tracks']} "
                 f"(declared {s['declared_num_tracks']})")
    lines.append(f"- splits: {s['split_counts']}")
    lines.append(f"- speakers: {s['n_speakers']}")
    lines.append(f"- clipped tracks (fraction >= {s['clip_threshold']}): "
                 f"{s['clipped_count']} ({s['clip_rate']:.1%})")

    lines.append("")
    lines.append("## Per-speaker counts")
    items = [f"{k}:{v}" for k, v in s["speaker_counts"].items()]
    for i in range(0, len(items), 6):
        lines.append("  " + "  ".join(items[i:i + 6]))

    lines.append("")
    lines.append("## Duration")
    d = s["duration"]
    if d.get("n"):
        lines.append(f"  min {d['min']:.2f}s  median {d['median']:.2f}s  "
                     f"mean {d['mean']:.2f}s  max {d['max']:.2f}s")
    else:
        lines.append("  (no duration_s in manifest)")

    lines.append("")
    lines.append("## SNR (dB)")
    if s["snr"].get("n"):
        a = s["snr"]
        lines.append(f"  min {a['min']:.2f}  p10 {a['p10']:.2f}  "
                     f"median {a['median']:.2f}  mean {a['mean']:.2f}  "
                     f"p90 {a['p90']:.2f}  max {a['max']:.2f}  "
                     f"std {a['std']:.2f}")

    lines.append("")
    lines.append("## Clipped fraction")
    if s["clipped_fraction"].get("n"):
        a = s["clipped_fraction"]
        lines.append(f"  min {a['min']:.4f}  median {a['median']:.4f}  "
                     f"mean {a['mean']:.4f}  p90 {a['p90']:.4f}  "
                     f"max {a['max']:.4f}")

    lines.append("")
    lines.append("## Noise files")
    if s["noise_files"] is None:
        lines.append("  not recorded in this manifest "
                     "(demo manifests only carry speaker/split)")
    else:
        lines.append(f"  {s['n_noise_files']} distinct noise files")
        for k, v in list(s["noise_files"].items())[:20]:
            lines.append(f"    {k}: {v}")
        if len(s["noise_files"]) > 20:
            lines.append(f"    ... {len(s['noise_files']) - 20} more")

    lines.append("")
    lines.append("## DC offset")
    dc = s["dc"]
    if dc is None:
        lines.append("  not measured (no audio files reachable)")
    else:
        lines.append(f"  clips measured {dc['n_loaded']} "
                     f"(skipped {dc['n_skipped']})")
        lines.append(f"  mean {dc['mean']:+.5f}  mean|DC| {dc['mean_abs']:.5f}  "
                     f"max|DC| {dc['max_abs']:.5f}")
        lines.append(f"  clips with |DC| > {DC_FLAG:g}: "
                     f"{dc['frac_above_flag']:.1%}")

    lines.append("")
    lines.append("## Leakage check (speaker-disjoint split)")
    lk = s["leakage"]
    if lk["ok"]:
        lines.append(f"  PASS - {lk['n_speakers']} speakers, none appears in "
                     f"more than one split")
    else:
        lines.append(f"  FAIL - {lk['n_speakers']} speakers, overlapping: "
                     f"{lk['speakers_in_multiple_splits']}")
        lines.append(f"  train/val overlap: {lk['speakers_in_train_and_val']}")
    lines.append("")
    return "\n".join(lines)


def render_with_histograms(summary: dict, snr_values, frac_values,
                           dur_values) -> str:
    """Full report including the ASCII histograms (printed, not in RESULTS.md)."""
    base = render(summary)
    parts = [base]
    if len(snr_values):
        parts.append(ascii_hist(snr_values, bins=10, label="SNR (dB)", digits=2))
    if len(frac_values):
        parts.append(ascii_hist(frac_values, bins=10, label="clipped fraction",
                                digits=4))
    if len(dur_values):
        parts.append(ascii_hist(dur_values, bins=8, label="duration (s)",
                                digits=2))
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Synthetic manifest for --smoke
# ---------------------------------------------------------------------------

def make_smoke_manifest(count: int = 40, seed: int = 1234):
    """Build a synthetic manifest in memory using preprocess.py's own helpers.

    Returns (manifest, audio_lookup) where audio_lookup maps track id to the
    float32 samples, so the DC-offset branch runs without touching disk.
    """
    from preprocess import (
        build_manifest,
        make_demo_clip,
        sha256_bytes,
        speaker_split,
        wav_bytes,
    )

    rng = np.random.default_rng(seed)
    speakers = [f"demo{chr(ord('a') + i % 6)}" for i in range(count)]
    assignment = speaker_split(speakers, val_fraction=0.2, seed=seed)

    tracks: list[dict] = []
    audio: dict[str, np.ndarray] = {}
    for i in range(count):
        clip = make_demo_clip(rng)
        tid = f"demo-{i:04d}"
        audio[tid] = clip["audio"]
        tracks.append({
            "id": tid,
            "file": f"memory://{tid}.wav",
            "split": assignment[speakers[i]],
            "speaker": speakers[i],
            "snr_db": round(clip["snr_db"], 4),
            "clipped_fraction": round(clip["clipped_fraction"], 6),
            "clip_threshold": round(clip["clip_threshold"], 6),
            "duration_s": clip["duration_s"],
            "sha256": sha256_bytes(wav_bytes(clip["audio"])),
        })

    manifest = build_manifest(
        "smoke-synthetic-eda", "https://audelle.local/synthetic", tracks,
        extra={"note": "In-memory synthetic clips for scripts/eda.py --smoke."},
    )
    return manifest, audio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="EDA on a preprocess.py manifest (ASCII only).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="manifest path (default data/manifest.json)")
    p.add_argument("--smoke", action="store_true",
                   help="build a synthetic manifest in memory, no data")
    p.add_argument("--count", type=int, default=40,
                   help="smoke clips (default 40)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--report-out", type=Path, default=None)
    args = p.parse_args(argv)

    audio_lookup = None
    if args.smoke:
        manifest, audio_lookup = make_smoke_manifest(args.count, args.seed)
        manifest_path = Path("memory://synthetic")
    else:
        manifest_path = args.manifest or (DATA_DIR / "manifest.json")
        if not manifest_path.is_file():
            print(f"manifest not found: {manifest_path}\n"
                  f"Run: python scripts/preprocess.py --demo\n"
                  f"or:  python scripts/eda.py --smoke", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    summary = summarise_manifest(manifest, audio_lookup)

    tracks = manifest.get("tracks", [])
    snr_values = [float(t["snr_db"]) for t in tracks if "snr_db" in t]
    frac_values = [float(t["clipped_fraction"]) for t in tracks
                   if "clipped_fraction" in t]
    dur_values = [float(t["duration_s"]) for t in tracks if "duration_s" in t]

    text = render_with_histograms(summary, snr_values, frac_values, dur_values)
    print(text)

    suffix = "_smoke" if args.smoke else ""
    report_out = args.report_out or (OUT_DIR / f"eda_report{suffix}.md")
    json_out = args.json_out or (OUT_DIR / f"eda_summary{suffix}.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(text + "\n", encoding="utf-8")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"report  {report_out}")
    print(f"summary {json_out}")
    if args.smoke:
        print(f"source  synthetic manifest in memory ({manifest_path})")
    else:
        print(f"source  {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
