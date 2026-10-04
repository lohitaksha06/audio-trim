"""Train the phrase model on real note sequences.

Pipeline per track:
  Demucs (isolate a mostly-monophonic stem) -> librosa.pyin pitch track ->
  segment into notes -> quantise to the detected grid -> (state, action) pairs

Then a small MLP learns state -> (semitone delta, note length in steps, velocity).

We learn *rhythm and phrasing* from real music. The audio we ship is rendered
from our own trained timbre model, so nothing from the training data is
distributed.

Usage:
    python -m scripts.train_phrasing_model --audio-dir <folder of real music>
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa

from server.ml.synthesis.phrasing import (
    DELTA_BINS,
    DUR_BINS,
    PHRASE_STEPS,
    STATE_DIM,
    _build,
)

MIN_NOTE_S = 0.06
MAX_NOTE_S = 2.0
GAP_S = 0.05
MAX_ABS_DELTA = 7
MAX_DUR_STEPS = 16


_sep = None


def _separator():
    """One cached Separator. Reloading Demucs per track dominated the runtime."""
    global _sep
    if _sep is None:
        from server.ml.source_separation.separator import Separator

        _sep = Separator()
    return _sep


def _isolate_mono(path: str) -> tuple[np.ndarray, int]:
    """Best-effort monophonic-ish stem; falls back to the raw mix."""
    y, sr = librosa.load(path, sr=22050, mono=True)
    try:
        sep = _separator()
    except Exception:
        return y, sr
    for target in ("other", "bass"):
        try:
            p = sep.isolate(path, target)
            if p:
                ys, srs = librosa.load(p, sr=22050, mono=True)
                if ys.size > sr:
                    return ys, srs
        except Exception:
            continue
    return y, sr


def _track_notes(y: np.ndarray, sr: int) -> list[tuple[float, float, int, float]]:
    """Return [(onset_s, duration_s, midi, rms)] from a pitch track."""
    fmin, fmax = librosa.note_to_hz("C2"), librosa.note_to_hz("C6")
    f0, voiced, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=256
    )
    if f0 is None or not np.any(voiced):
        return []

    hop = 256
    times = librosa.times_like(voiced, sr=sr, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    ok = np.nan_to_num(voiced, nan=0.0).astype(bool) & np.isfinite(midi)

    gap = max(1, int(GAP_S * sr / hop))
    notes: list[tuple[float, float, int, float]] = []
    i, n = 0, len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = last = i
        while j < n:
            if ok[j]:
                last = j
            elif j - last > gap:
                break
            j += 1
        seg = midi[i:last + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            onset = float(times[i])
            dur = float(times[last] - times[i] + hop / sr)
            if MIN_NOTE_S <= dur <= MAX_NOTE_S:
                a = int(onset * sr)
                b = min(len(y), int((onset + dur) * sr))
                rms = float(np.sqrt(np.mean(y[a:b] ** 2))) if b > a else 0.0
                notes.append((onset, dur, int(round(float(np.median(seg)))), rms))
        i = last + 1
    return notes


def _tempo(y: np.ndarray, sr: int) -> float:
    try:
        return float(np.median(librosa.beat.tempo(y=y, sr=sr, aggregate=None)))
    except Exception:
        return 120.0


def _pairs_from_notes(
    notes: list[tuple[float, float, int, float]], bpm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one track's notes into (state, action) rows. Rests are included.

    State layout (must match `phrasing.generate_part`):
      0  semitones moved since the start of the current phrase, /24
      1  position through the phrase, 0..1
      2  previous pitch relative to the track's lowest note, /24
      3  loudness of the sounding note (0 when resting)
      4  local note density in the surrounding bar, 0..1
    Action:
      0  semitone delta to the next note (0 when resting)
      1  note length in sixteenth-note steps (0 when resting)
      2  velocity 0..1
    """
    if len(notes) < 6:
        empty = (np.zeros((0, STATE_DIM), np.float32),
                 np.zeros((0, ACT_DIM), np.float32))
        return empty

    step = 60.0 / bpm / 4.0
    grid = sorted(((int(round(on / step)), dur, mid, rms)
                   for on, dur, mid, rms in notes), key=lambda x: x[0])
    clean: list[tuple[int, float, int, float]] = []
    for gs, dur, mid, rms in grid:
        if clean and gs <= clean[-1][0] + 1:
            continue  # pitch trackers double-report onsets
        clean.append((gs, dur, mid, rms))
    if len(clean) < 6:
        return (np.zeros((0, STATE_DIM), np.float32),
                np.zeros((0, ACT_DIM), np.float32))

    root = min(m for _, _, m, _ in clean)
    peak_rms = max((r for *_, r in clean), default=1.0) or 1.0
    last_step = clean[-1][0]

    X: list[list[float]] = []
    Y: list[list[float]] = []

    prev_mid = clean[0][2]
    phrase_start_mid = prev_mid
    since_reset = 0

    for idx, (gs, dur, mid, rms) in enumerate(clean):
        # Rests: one row per skipped step, up to a bar, so the model learns to
        # stop without learning to emit thousands of consecutive rests.
        skipped = gs - (clean[idx - 1][0] + int(round(clean[idx - 1][1] / step))) \
            if idx else 0
        for k in range(min(max(skipped, 0), 16)):
            at = (clean[idx - 1][0] + k + 1) if idx else k
            X.append([
                (prev_mid - phrase_start_mid) / 24.0,
                (at % PHRASE_STEPS) / PHRASE_STEPS,
                (prev_mid - root) / 24.0,
                0.0,
                min(len(clean) / max(last_step / 16.0, 1.0) / 4.0, 1.0),
            ])
            Y.append([0.0, 0.0, 0.0])
            since_reset += 1

        if since_reset >= PHRASE_STEPS:
            phrase_start_mid = prev_mid
            since_reset = 0

        X.append([
            (prev_mid - phrase_start_mid) / 24.0,
            (gs % PHRASE_STEPS) / PHRASE_STEPS,
            (prev_mid - root) / 24.0,
            float(rms / peak_rms),
            min(len(clean) / max(last_step / 16.0, 1.0) / 4.0, 1.0),
        ])
        Y.append([
            float(np.clip(mid - prev_mid, -MAX_ABS_DELTA, MAX_ABS_DELTA)),
            float(np.clip(dur / step, 1.0, MAX_DUR_STEPS)),
            float(np.clip(rms / peak_rms, 0.05, 1.0)),
        ])

        prev_mid = mid
        since_reset += int(round(dur / step))

    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def build(audio_dir: str, limit: int | None = None,
          cache: str | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract (state, action) pairs, caching so retraining skips Demucs."""
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        print(f"using cached pairs: {len(z['X'])} from {int(z['used'])} tracks")
        return z["X"], z["Y"], int(z["used"])

    files = [p for p in Path(audio_dir).rglob("*")
             if p.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a")]
    if not files:
        raise SystemExit(f"no audio files under {audio_dir}")
    if limit:
        files = files[:limit]

    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    used = 0
    for i, f in enumerate(files, 1):
        try:
            y, sr = _isolate_mono(str(f))
            if y.size < sr * 5:
                continue
            notes = _track_notes(y, sr)
            if len(notes) < 8:
                continue
            X, Y = _pairs_from_notes(notes, _tempo(y, sr))
            if len(X) < 8:
                continue
            Xs.append(X)
            Ys.append(Y)
            used += 1
            print(f"  [{i}/{len(files)}] {f.name}: {len(X)} pairs")
        except Exception:
            if "--trace" in sys.argv:
                traceback.print_exc()
            continue
    if not Xs:
        raise SystemExit(
            "no usable tracks. Check the path, and that Demucs + pyin work."
        )
    X, Y = np.concatenate(Xs), np.concatenate(Ys)
    if cache:
        np.savez_compressed(cache, X=X, Y=Y, used=np.asarray(used))
        print(f"cached pairs to {cache}")
    return X, Y, used


def train(X: np.ndarray, Y: np.ndarray, epochs: int = 120, seed: int = 0):
    """Interval and length are classified; only velocity is regressed."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)

    # Y columns: [delta, dur_steps, velocity] -> bins
    delta_bin = np.clip(
        np.rint(Y[:, 0]).astype(int) + (DELTA_BINS // 2), 0, DELTA_BINS - 1
    )
    dur_bin = np.clip(np.rint(Y[:, 1]).astype(int), 0, DUR_BINS - 1)
    vel = Y[:, 2].astype(np.float32)

    d_t = torch.tensor(delta_bin, dtype=torch.long)
    u_t = torch.tensor(dur_bin, dtype=torch.long)
    v_t = torch.tensor(vel, dtype=torch.float32)

    net = _build()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    def loss_of(xb, db, ub, vb):
        o = net(xb)
        return (
            ce(o[:, :DELTA_BINS], db)
            + ce(o[:, DELTA_BINS:DELTA_BINS + DUR_BINS], ub)
            + 0.5 * mse(o[:, -1], vb)
        )

    n = len(X)
    idx = torch.randperm(n)
    val = max(1, int(n * 0.15))
    tr, va = idx[:-val], idx[-val:]

    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            loss = loss_of(x[b], d_t[b], u_t[b], v_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        net.eval()
        with torch.no_grad():
            vl = float(loss_of(x[va], d_t[va], u_t[va], v_t[va]))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  train {tot / max(len(tr), 1):.4f}  val {vl:.4f}")
    net.load_state_dict(best_state)
    net.eval()
    return net, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True,
                    help="folder of real music to learn phrasing from")
    ap.add_argument("--out", default=r"server\ml\models")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--cache", default=r"server\ml\models\phrase_pairs.npz",
                    help="cache extracted pairs so retraining skips Demucs")
    args = ap.parse_args()

    print(f"extracting notes from {args.audio_dir} ...")
    X, Y, used = build(args.audio_dir, limit=args.limit, cache=args.cache)
    print(f"tracks used: {used}   pairs: {len(X)}")

    print("training ...")
    net, val = train(X, Y, epochs=args.epochs)

    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "hidden": 96,
            "val_loss": val,
            "pairs": int(len(X)),
            "tracks": used,
        },
        out / "phrase_model.pt",
    )
    (out / "phrase_model.json").write_text(json.dumps({
        "val_loss": val, "pairs": int(len(X)), "tracks": used,
        "arch": "5 -> 96 -> 96 -> 32",
        "heads": f"delta {DELTA_BINS} bins (-7..+7), "
                 f"duration {DUR_BINS} bins (0=rest..15), velocity 1",
    }, indent=2), encoding="utf-8")
    print("saved", out / "phrase_model.pt", "val_loss", round(val, 4))


if __name__ == "__main__":
    main()
