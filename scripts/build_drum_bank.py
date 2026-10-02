"""Build a drum bank from real recorded one-shots.

Why
---
`add drums` used to synthesise a kick from a pitch-swept sine and a snare from
bandpassed noise. That is not a drum recording, and it is why the feature was
worthless. This extracts the real one-shots from the `airasoul/drum-kit`
dataset so the sequencer plays actual recorded drums.

Scope, stated plainly: the *hits* are real recordings; the *arrangement* on the
grid is rule-based. There is no trained drum model here, and none is claimed.

Usage:
    python -m scripts.build_drum_bank --per-label 120
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 22050
MAX_SECONDS = 4.0
PEAK = 0.89


def _trim(y: np.ndarray, sr: int, thresh_db: float = -45.0) -> np.ndarray:
    """Trim leading/trailing near-silence, keeping a short pre-roll."""
    if y.size == 0:
        return y
    amp = np.abs(y)
    ref = float(amp.max()) + 1e-12
    keep = amp > ref * (10 ** (thresh_db / 20.0))
    if not keep.any():
        return y
    first, last = int(np.argmax(keep)), int(len(keep) - np.argmax(keep[::-1]))
    pad = int(0.005 * sr)
    return y[max(0, first - pad):min(len(y), last + pad)]


def _fade_edges(y: np.ndarray, sr: int, ms: float = 3.0) -> np.ndarray:
    """Short fades so one-shots do not click when placed on the grid."""
    if y.size == 0:
        return y
    n = min(int(sr * ms / 1000), y.size // 2)
    if n < 2:
        return y
    out = y.copy()
    out[:n] *= np.linspace(0.0, 1.0, n)
    out[-n:] *= np.linspace(1.0, 0.0, n)
    return out


def build(data_dir: str, out_dir: str, per_label: int = 120, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    files = sorted(Path(data_dir).glob("**/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet found under {data_dir}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index: dict[str, list[dict]] = {}
    seen: dict[str, int] = {}

    for pf in files:
        table = pq.read_table(str(pf), columns=["audio", "label"])
        rows = table.to_pylist()
        # Shuffle file order so a cap does not only take the first N per source.
        order = rng.permutation(len(rows))
        for i in order:
            row = rows[int(i)]
            label = str(row["label"]).strip().lower()
            if not label or label == "nan":
                continue
            if seen.get(label, 0) >= per_label:
                continue
            raw = row["audio"]["bytes"] if isinstance(row["audio"], dict) else row["audio"]
            if not raw:
                continue
            try:
                y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            except Exception:
                continue
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SR:
                import librosa

                y = librosa.resample(y, orig_sr=sr, target_sr=SR)
            if y.size < int(0.02 * SR):
                continue

            y = _trim(y, SR)
            if y.size > int(MAX_SECONDS * SR):
                y = y[: int(MAX_SECONDS * SR)]
            y = _fade_edges(y, SR)
            peak = float(np.max(np.abs(y))) + 1e-12
            y = (y / peak * PEAK).astype(np.float32)

            n = seen.get(label, 0)
            # FLAC is lossless and ~62% smaller than WAV here — the bank is
            # committed to the repo, so that is a real saving (60 MB -> 23 MB).
            name = f"{label}_{n:03d}.flac"
            sf.write(str(out / name), y, SR, format="FLAC", subtype="PCM_16")
            index.setdefault(label, []).append(
                {
                    "file": name,
                    "duration": round(float(y.size) / SR, 4),
                    # Measured, so the sequencer can pick hits by character.
                    "rms": round(float(np.sqrt(np.mean(y ** 2))), 5),
                    "peak": round(peak, 5),
                }
            )
            seen[label] = n + 1

    meta = {
        "sample_rate": SR,
        "per_label_cap": per_label,
        "labels": {k: len(v) for k, v in sorted(index.items())},
        "total_hits": sum(len(v) for v in index.values()),
        "hits": index,
        "source": "airasoul/drum-kit (real recorded one-shots)",
    }
    (out / "bank.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=r"C:\Users\Lohit\AppData\Local\Temp\opencode\data\drums",
    )
    ap.add_argument("--out", default=r"server\ml\models\drum_bank")
    ap.add_argument("--per-label", type=int, default=120)
    args = ap.parse_args()

    meta = build(args.data, args.out, per_label=args.per_label)
    print(f"total one-shots: {meta['total_hits']}")
    for k, v in meta["labels"].items():
        print(f"  {k:<10} {v}")


if __name__ == "__main__":
    main()