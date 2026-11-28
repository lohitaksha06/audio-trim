#!/usr/bin/env python3
"""Build 16 kHz noisy + clipped mixtures and write a dataset manifest.

The diagnoser needs clips whose damage is known exactly, so the damage is
applied here rather than measured later:

  * clean speech is resampled to 16 kHz,
  * DEMAND noise is mixed in at an SNR drawn uniformly from 0-30 dB,
  * the mixture is hard-clipped at a random threshold,
  * SNR and clipped fraction are recorded as ground-truth labels.

Splits are speaker-disjoint: every utterance of a speaker lands in exactly one
of train/val, so the validation score is not inflated by the same voice.

    python scripts/preprocess.py --demo
    python scripts/preprocess.py --clean-dir data/voicebank/clean \
                                 --noise-dir data/demand/OHALLWAY_16k --limit 200

``--demo`` needs no data: it synthesises ~40 one-second clips in memory and
writes ``data/manifest.demo.json`` so the pipeline can be proven end to end.

Only numpy + soundfile are required; the stdlib ``wave`` module is the fallback
when soundfile is missing. librosa is not used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

SR = 16_000
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".mp3"}

VOICEBANK_URL = "https://datashare.ed.ac.uk/handle/10283/2791"
DEMAND_URL = "https://zenodo.org/records/1227121"
LICENSE = "CC BY 4.0"

try:  # soundfile is optional
    import soundfile as sf
except Exception:  # noqa: BLE001 - any import failure means "use wave"
    sf = None


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read a file as mono float32. Returns (samples, sample_rate)."""
    if sf is not None:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return x.mean(axis=1).astype(np.float32), int(sr)

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: unsupported sample width {width} bytes")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x.astype(np.float32), sr


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    """Write 16-bit PCM mono. Falls back to wave when soundfile is absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if sf is not None:
        sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")
        return
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    """Encode to WAV in memory, for hashing demo clips without touching disk."""
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Resampling (no scipy, no librosa)
# ---------------------------------------------------------------------------

def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz. Integer ratios decimate, others interpolate.

    Decimation is preceded by a boxcar average, a crude anti-alias filter. It
    is honest about being crude: enough to keep the pipeline moving, not a
    mastering-grade resampler.
    """
    if sr == SR or x.size == 0:
        return x.astype(np.float32)

    if sr % SR == 0:  # 48k -> 16k, 32k -> 16k
        k = sr // SR
        kernel = np.ones(k, dtype=np.float32) / k
        filtered = np.convolve(x, kernel, mode="same")
        return filtered[::k].astype(np.float32)

    n_out = int(round(x.size * SR / sr))
    t_in = np.arange(x.size, dtype=np.float64) / sr
    t_out = np.arange(n_out, dtype=np.float64) / SR
    return np.interp(t_out, t_in, x).astype(np.float32)


# ---------------------------------------------------------------------------
# Levels, mixing, clipping
# ---------------------------------------------------------------------------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def speech_level(x: np.ndarray) -> float:
    """Level of the loud part of a clip, ignoring silence between words."""
    frame = max(1, int(0.02 * SR))
    n = (x.size // frame) * frame
    if n < frame:
        return rms(x)
    frames = np.square(x[:n].reshape(-1, frame), dtype=np.float64)
    energies = frames.mean(axis=1)
    thresh = np.quantile(energies, 0.75)
    loud = energies[energies >= thresh]
    return float(np.sqrt(loud.mean() + 1e-12))


def fit_length(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Crop or tile x to exactly n samples, starting at a random offset."""
    if x.size == 0:
        return np.zeros(n, dtype=np.float32)
    if x.size >= n:
        start = int(rng.integers(0, x.size - n + 1))
        return x[start:start + n]
    reps = int(np.ceil(n / x.size))
    return np.tile(x, reps)[:n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.Generator) -> np.ndarray:
    """Scale noise to an exact SNR relative to the speech's active level."""
    s_level = speech_level(speech)
    n_level = rms(noise)
    if n_level < 1e-9:
        return speech.copy()
    target = s_level / (10.0 ** (snr_db / 20.0))
    return (speech + noise * (target / n_level)).astype(np.float32)


def hard_clip(x: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    """Clip at +-threshold. Returns (clipped, fraction of samples clipped)."""
    clipped = np.clip(x, -threshold, threshold).astype(np.float32)
    fraction = float(np.mean(np.abs(x) >= threshold))
    return clipped, fraction


def normalise_peak(x: np.ndarray, target: float) -> np.ndarray:
    peak = float(np.max(np.abs(x)) + 1e-12)
    return (x * (target / peak)).astype(np.float32)


# ---------------------------------------------------------------------------
# Demo synthesis (in memory, no downloads)
# ---------------------------------------------------------------------------

def synth_speech(n: int, rng: np.random.Generator) -> np.ndarray:
    """A voiced, formant-ish buzz with syllable gating. Not speech, but it has
    the spectral and temporal shape speech has, which is all the smoke test
    needs."""
    t = np.arange(n, dtype=np.float64) / SR
    f0 = rng.uniform(95.0, 210.0)
    vib = 1.0 + 0.02 * np.sin(2 * np.pi * rng.uniform(4.0, 6.5) * t)
    phase = 2 * np.pi * f0 * np.cumsum(vib) / SR
    x = np.zeros(n, dtype=np.float64)
    for k in range(1, int(rng.integers(8, 16)) + 1):
        if k * f0 > 7600:
            break
        x += (1.0 / k ** 1.15) * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    # crude formant tilt
    x *= np.exp(-t / rng.uniform(0.25, 0.6))
    # syllables
    gate = np.zeros(n, dtype=np.float64)
    pos = 0
    while pos < n:
        length = int(rng.uniform(0.08, 0.28) * SR)
        gap = int(rng.uniform(0.02, 0.12) * SR)
        end = min(n, pos + length)
        env = np.sin(np.linspace(0, np.pi, end - pos)) ** 2
        gate[pos:end] = env
        pos = end + gap
    x = x * gate + 1e-4 * rng.standard_normal(n)
    return normalise_peak(x.astype(np.float32), 0.9)


def synth_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Coloured noise plus optional mains hum."""
    white = rng.standard_normal(n)
    smooth = np.convolve(white, np.ones(9) / 9.0, mode="same")
    x = 0.7 * white + 0.3 * smooth * 3.0
    if rng.random() < 0.5:
        t = np.arange(n) / SR
        x += 0.15 * np.sin(2 * np.pi * rng.choice([50.0, 60.0]) * t)
    return normalise_peak(x.astype(np.float32), 0.5)


def make_demo_clip(rng: np.random.Generator, seconds: float = 1.0) -> dict:
    n = int(seconds * SR)
    speech = synth_speech(n, rng)
    noise = synth_noise(n, rng)
    snr_db = float(rng.uniform(0.0, 30.0))
    mixed = mix_at_snr(speech, noise, snr_db, rng)
    mixed = normalise_peak(mixed, float(rng.uniform(0.6, 1.0)))
    threshold = float(np.max(np.abs(mixed)) * rng.uniform(0.15, 0.9))
    clipped, fraction = hard_clip(mixed, threshold)
    return {
        "audio": clipped,
        "snr_db": snr_db,
        "clipped_fraction": fraction,
        "clip_threshold": threshold,
        "duration_s": seconds,
    }


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def speaker_of(path: Path) -> str:
    """VoiceBank files are p226_001.wav; fall back to the parent folder."""
    m = re.match(r"^(p\d+)[_-]", path.stem)
    if m:
        return m.group(1)
    m = re.match(r"^(p\d+)$", path.parent.name)
    if m:
        return m.group(1)
    return path.parent.name or "unknown"


def speaker_split(speakers: list[str], val_fraction: float,
                  seed: int) -> dict[str, str]:
    """Assign whole speakers to train or val, never both."""
    unique = sorted(set(speakers))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(unique)))
    n_val = max(1, int(round(len(unique) * val_fraction))) if len(unique) > 1 else 0
    val_speakers = {unique[i] for i in order[:n_val]}
    return {s: ("val" if s in val_speakers else "train") for s in unique}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(name: str, source_url: str, tracks: list[dict],
                   extra: dict | None = None) -> dict:
    """Assemble a schema-shaped manifest with per-track checksums."""
    counts: dict[str, int] = {}
    for t in tracks:
        counts[t["split"]] = counts.get(t["split"], 0) + 1

    # Top-level split is the primary one; per-track split is authoritative.
    primary = max(counts, key=counts.get) if counts else "train"

    payload = json.dumps(tracks, sort_keys=True, separators=(",", ":"))
    manifest = {
        "name": name,
        "source_url": source_url,
        "license": LICENSE,
        "num_tracks": len(tracks),
        "labels": ["snr_db", "clipped_fraction", "clip_threshold", "speaker"],
        "split": primary,
        "checksum": sha256_bytes(payload.encode("utf-8")),
        "sample_rate": SR,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "scripts/preprocess.py",
        "split_counts": counts,
        "tracks": tracks,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def summarise(manifest: dict, out: Path, elapsed: float) -> None:
    tracks = manifest["tracks"]
    print(f"\nmanifest  {out}")
    print(f"  name        {manifest['name']}")
    print(f"  tracks      {manifest['num_tracks']}")
    print(f"  splits      {manifest['split_counts']}")
    if tracks:
        snrs = [t["snr_db"] for t in tracks]
        frs = [t["clipped_fraction"] for t in tracks]
        print(f"  snr dB      min {min(snrs):5.2f}  max {max(snrs):5.2f}  "
              f"mean {float(np.mean(snrs)):5.2f}")
        print(f"  clipped     min {min(frs):.4f}  max {max(frs):.4f}  "
              f"mean {float(np.mean(frs)):.4f}")
    print(f"  checksum    {manifest['checksum'][:16]}...")
    print(f"  elapsed     {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_demo(count: int, seed: int, val_fraction: float, out: Path) -> int:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    speakers = [f"demo{chr(ord('a') + i % 6)}" for i in range(count)]
    assignment = speaker_split(speakers, val_fraction, seed)

    tracks = []
    for i in range(count):
        clip = make_demo_clip(rng)
        spk = speakers[i]
        split = assignment[spk]
        data = wav_bytes(clip["audio"])
        tracks.append({
            "id": f"demo-{i:04d}",
            "file": f"memory://demo-{i:04d}.wav",
            "split": split,
            "speaker": spk,
            "snr_db": round(clip["snr_db"], 4),
            "clipped_fraction": round(clip["clipped_fraction"], 6),
            "clip_threshold": round(clip["clip_threshold"], 6),
            "duration_s": clip["duration_s"],
            "sha256": sha256_bytes(data),
        })

    manifest = build_manifest(
        "demo-synthetic-1s", "https://audelle.local/synthetic", tracks,
        extra={"note": "Synthetic 1-second clips generated in memory. No real "
                       "audio was downloaded; files are not written to disk."},
    )
    write_manifest(manifest, out)
    summarise(manifest, out, time.perf_counter() - started)
    return 0


def find_audio(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def run_real(clean_dir: Path, noise_dir: Path, limit: int, seed: int,
             val_fraction: float, out: Path,
             force_split: str | None = None) -> int:
    if not clean_dir.is_dir():
        print(f"clean dir not found: {clean_dir}\n"
              f"Run: python scripts/download_data.py --source voicebank "
              f"--check\n(then fetch the subset you want)", file=sys.stderr)
        return 2
    if not noise_dir.is_dir():
        print(f"noise dir not found: {noise_dir}\n"
              f"Run: python scripts/download_data.py --source demand --check\n"
              f"(then fetch e.g. OHALLWAY_16k and unpack it here)",
              file=sys.stderr)
        return 2

    clean_files = find_audio(clean_dir)
    noise_files = find_audio(noise_dir)
    if not clean_files or not noise_files:
        print(f"no audio found (clean={len(clean_files)}, "
              f"noise={len(noise_files)})", file=sys.stderr)
        return 2

    rng = np.random.default_rng(seed)
    if force_split:
        assignment = {speaker_of(p): force_split for p in clean_files}
    else:
        assignment = speaker_split([speaker_of(p) for p in clean_files],
                                   val_fraction, seed)

    started = time.perf_counter()
    tracks = []
    for i, path in enumerate(clean_files[:limit]):
        try:
            speech, sr = load_audio(path)
        except Exception as e:  # noqa: BLE001 - skip a bad file, keep going
            print(f"  skip {path.name}: {e}", file=sys.stderr)
            continue
        speech = resample_to_16k(speech, sr)
        if speech.size < SR // 4:
            continue

        noise_path = noise_files[int(rng.integers(0, len(noise_files)))]
        noise, noise_sr = load_audio(noise_path)
        noise = resample_to_16k(noise, noise_sr)

        snr_db = float(rng.uniform(0.0, 30.0))          # continuous, not 4 levels
        n = speech.size
        mixed = mix_at_snr(speech, fit_length(noise, n, rng), snr_db, rng)
        mixed = normalise_peak(mixed, float(rng.uniform(0.6, 1.0)))

        threshold = float(np.max(np.abs(mixed)) * rng.uniform(0.15, 0.9))
        clipped, fraction = hard_clip(mixed, threshold)

        spk = speaker_of(path)
        split = assignment[spk]
        dest = DATA_DIR / "processed" / split / f"{spk}-{i:05d}.wav"
        write_wav(dest, clipped)

        tracks.append({
            "id": f"{spk}-{i:05d}",
            "file": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "split": split,
            "speaker": spk,
            "snr_db": round(snr_db, 4),
            "clipped_fraction": round(fraction, 6),
            "clip_threshold": round(threshold, 6),
            "duration_s": round(n / SR, 3),
            "noise_file": noise_path.name,
            "sha256": sha256_file(dest),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} clips...")

    manifest = build_manifest(
        "voicebank-demand-mixtures", VOICEBANK_URL, tracks,
        extra={"noise_source_url": DEMAND_URL},
    )
    write_manifest(manifest, out)
    summarise(manifest, out, time.perf_counter() - started)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build 16 kHz noisy/clipped mixtures + a manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="--demo needs no data and writes data/manifest.demo.json.",
    )
    p.add_argument("--demo", action="store_true",
                   help="synthesise ~40 one-second clips in memory (no data)")
    p.add_argument("--count", type=int, default=40,
                   help="number of demo clips (default 40)")
    p.add_argument("--clean-dir", type=Path, default=DATA_DIR / "voicebank",
                   help="folder of clean speech (default data/voicebank)")
    p.add_argument("--noise-dir", type=Path, default=DATA_DIR / "demand",
                   help="folder of DEMAND noise wavs (default data/demand)")
    p.add_argument("--limit", type=int, default=200,
                   help="max clips to process in a real run (default 200)")
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="fraction of SPEAKERS held out (default 0.15)")
    p.add_argument("--force-split", type=str, default=None,
                   choices=["train", "val", "test"],
                   help="assign every track this split (used to build the "
                        "held-out test set from unseen speakers)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path,
                   help="manifest path (default data/manifest.json, or "
                        "data/manifest.demo.json with --demo)")
    args = p.parse_args(argv)

    out = args.out or (DATA_DIR / ("manifest.demo.json" if args.demo
                                   else "manifest.json"))

    if args.demo:
        return run_demo(max(1, args.count), args.seed, args.val_fraction, out)
    return run_real(args.clean_dir, args.noise_dir, args.limit, args.seed,
                    args.val_fraction, out, args.force_split)
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
