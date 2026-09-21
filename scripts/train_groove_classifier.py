"""Train the groove (electronic-subgenre feel) classifier.

Supervised training, zero storage footprint: the training loops are
synthesized in RAM (no downloads, no dataset folder), each with musical
variation (BPM, key, swing, humanization, noise, mastering). Real songs can be
mixed in with --songs-dir <root> containing <label>/ audio files — retraining
with your own tracks is how the model keeps learning your sound.

Labels: house techno trance bigroom dubstep trap dnb phonk hiphop pop

Usage:
    python scripts/train_groove_classifier.py [--songs-dir data/songs] [--out ...]

Artifact (~a few hundred KB) loads lazily at request time, only when the user
asks for drums that match the song.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.ml.rhythm.analyzer import LABEL_TO_GROOVE  # noqa: E402

SR = 22050
CLIP_SECONDS = 8.0
LABELS = sorted(LABEL_TO_GROOVE.keys() - {"disco"})  # disco maps to house; trained as house feel
DEFAULT_OUT = ROOT / "server" / "ml" / "models" / "groove_classifier.joblib"


# ----------------------------------------------------------------------------
# Independent synthesizer (shares NO code with the drum patterns it trains).
# Raw, slightly gritty loops: mastered with noise + soft clip so the model
# learns feel, not clean sine waves.
# ----------------------------------------------------------------------------

def _kick(n=None):
    nn = n or int(0.16 * SR)
    t = np.arange(nn) / SR
    f = 150 * np.exp(-t * 30) + 48
    phase = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(phase) * np.exp(-t * 14)).astype(np.float32)


def _snare(n=None, body=190.0):
    nn = n or int(0.2 * SR)
    rng = np.random.default_rng(5)
    t = np.arange(nn) / SR
    noise = rng.standard_normal(nn).astype(np.float32) * np.exp(-t * 22)
    tone = np.sin(2 * np.pi * body * t).astype(np.float32) * np.exp(-t * 30)
    return (0.6 * noise + 0.5 * tone).astype(np.float32)


def _hat(open_=False, amp=0.3):
    nn = int((0.14 if open_ else 0.045) * SR)
    rng = np.random.default_rng(9)
    t = np.arange(nn) / SR
    noise = rng.standard_normal(nn).astype(np.float32)
    return (amp * noise * np.exp(-t * (40 if open_ else 130))).astype(np.float32)


def _cowbell(freq=820.0):
    nn = int(0.12 * SR)
    t = np.arange(nn) / SR
    tone = np.sign(np.sin(2 * np.pi * freq * t)) + 0.6 * np.sign(np.sin(2 * np.pi * freq * 1.48 * t))
    return (0.22 * tone * np.exp(-t * 28)).astype(np.float32)


def _bass_note(freq, dur):
    nn = max(1, int(dur * SR))
    t = np.arange(nn) / SR
    tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    return (0.4 * tone * np.minimum(1.0, t * 60) * np.exp(-t * 3)).astype(np.float32)


def _master(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = y + rng.standard_normal(len(y)).astype(np.float32) * 0.004  # room/hiss
    y = np.tanh(y * 1.4) * 0.85  # glue + soft clip
    peak = float(np.max(np.abs(y))) + 1e-9
    return (y / peak * 0.89).astype(np.float32)


class Loop:
    """Event-list renderer. Events: (beat_pos, kind, opts)."""

    def __init__(self, bpm: float, seed: int, swing: float = 0.0, bare: bool = False):
        self.bpm = bpm
        self.step = 60.0 / bpm
        self.rng = np.random.default_rng(seed)
        self.swing = swing
        self.bare = bare  # drums-only: skip bass/cowbell/mastering
        self.n = int(CLIP_SECONDS * SR)
        self.y = np.zeros(self.n, dtype=np.float32)
        self.K, self.S = _kick(), _snare()
        self.H, self.OH = _hat(), _hat(open_=True)

    def _at(self, beat: float) -> float:
        t = beat * self.step
        # swing pushes offbeat 8ths late; humanization jitters everything
        if self.swing and abs((beat * 2) % 2 - 1) < 0.01:
            t += self.swing * 0.12 * self.step
        t += float(self.rng.normal(0, 0.004))
        return t

    def put(self, beat: float, hit: np.ndarray, gain: float = 1.0):
        if beat < 0 or len(hit) == 0:
            return
        s = int(self._at(beat) * SR)
        if s >= self.n:
            return
        s0 = max(s, 0)
        skip = s0 - s
        e = min(s + len(hit), self.n)
        if e <= s0:
            return
        self.y[s0:e] += (gain * hit[skip:skip + (e - s0)]).astype(np.float32)

    def kick(self, beat: float, gain: float = 1.0):
        self.put(beat, self.K, gain)

    def snare(self, beat: float, gain: float = 0.9):
        self.put(beat, self.S, gain)

    def hat(self, beat: float, gain: float = 0.5, open_=False):
        self.put(beat, self.OH if open_ else self.H, gain)

    def cowbell(self, beat: float, freq: float = 820.0):
        if self.bare:
            return
        self.put(beat, _cowbell(freq))

    def bass(self, beat: float, freq: float, dur: float):
        if self.bare:
            return
        self.put(beat, _bass_note(freq, dur))

    def done(self, seed: int, bare: bool = False) -> np.ndarray:
        if bare:
            peak = float(np.max(np.abs(self.y))) + 1e-9
            return (self.y / peak * 0.89).astype(np.float32)
        return _master(self.y, seed)


def render_loops(label: str, seed: int, bare: bool = False) -> list[tuple[np.ndarray, str]]:
    """Several BPM/key/swing variants of one feel. Returns (audio, label).

    ``bare`` renders drums-only without bass/cowbell/mastering so the model
    learns feel across production styles, not one polished timbre.
    """
    out: list[tuple[np.ndarray, str]] = []
    cfgs: dict[str, list[tuple[float, float, float]]] = {
        # (bpm, root_hz, swing)
        "house": [(124, 55.0, 0.0), (128, 49.0, 0.0), (126, 58.3, 0.0), (130, 55.0, 0.0)],
        "techno": [(132, 55.0, 0.0), (138, 49.0, 0.0), (136, 58.3, 0.0)],
        "trance": [(140, 55.0, 0.0), (142, 65.4, 0.0)],
        "bigroom": [(128, 55.0, 0.0), (130, 49.0, 0.0)],
        "dubstep": [(140, 55.0, 0.0), (150, 49.0, 0.0)],
        "trap": [(142, 55.0, 0.0), (155, 49.0, 0.0), (148, 65.4, 0.0), (150, 55.0, 0.0)],
        "dnb": [(174, 55.0, 0.0), (176, 49.0, 0.0), (170, 58.3, 0.0)],
        "phonk": [(134, 55.0, 0.6), (142, 49.0, 0.7), (138, 58.3, 0.5), (140, 55.0, 0.65)],
        "hiphop": [(92, 55.0, 0.65), (96, 49.0, 0.55), (94, 58.3, 0.6)],
        "pop": [(104, 55.0, 0.0), (116, 65.4, 0.0)],
        "garage": [(132, 55.0, 0.6), (138, 49.0, 0.7), (135, 58.3, 0.5)],
        "amapiano": [(112, 55.0, 0.0), (115, 49.0, 0.0), (113, 58.3, 0.0)],
        "afro_house": [(122, 55.0, 0.2), (124, 49.0, 0.25)],
        "jungle": [(172, 55.0, 0.0), (168, 49.0, 0.0)],
        "grime": [(140, 55.0, 0.0), (142, 49.0, 0.0)],
    }
    for i, (bpm, root, swing) in enumerate(cfgs[label]):
        L = Loop(bpm, seed * 100 + i, swing, bare)
        beats = int(CLIP_SECONDS / L.step)
        for b in range(beats):
            pos = b % 4
            if label in ("house", "bigroom"):
                L.kick(b)
                L.hat(b + 0.5, 0.45, open_=True)
                if pos in (1, 3):
                    L.snare(b)
                L.hat(b + 0.25, 0.2)
                L.hat(b + 0.75, 0.2)
                if label == "bigroom":
                    L.bass(b, root / 2, L.step * 0.9)
            elif label in ("techno", "trance"):
                L.kick(b)
                for off in (0.25, 0.5, 0.75):
                    L.hat(b + off, 0.28)
                if pos in (1, 3):
                    L.snare(b, 0.8)
                if label == "trance":
                    L.bass(b, root, L.step * 0.45)
            elif label == "dubstep":
                if pos == 0:
                    L.kick(b)
                if pos == 2:
                    L.snare(b)
                    L.kick(b, 0.5)
                L.hat(b + 0.5, 0.3)
                L.bass(b, root / 2, L.step * 1.4)
            elif label == "trap":
                if pos == 0 or (b % 8 == 6):
                    L.kick(b)
                if pos == 2:
                    L.snare(b)
                for off in (1 / 3, 2 / 3, 0.5):
                    L.hat(b + off, 0.3)
                L.bass(b if pos == 0 else -1, root / 2, L.step * 1.8)
            elif label == "dnb":
                if pos in (0, 2) or (b % 8 == 5):
                    L.kick(b, 0.9)
                if pos in (1, 3):
                    L.snare(b)
                L.hat(b + 0.5, 0.35)
            elif label == "phonk":
                if pos in (0, 2) or (b % 8 == 6):
                    L.kick(b)
                if pos in (1, 3):
                    L.snare(b, 0.85)
                L.hat(b + 0.5, 0.32)
                if b % 2 == 0:
                    L.cowbell(b + 0.5, 820.0 if pos == 0 else 660.0)
                L.bass(b if pos == 0 else -1, root / 2, L.step)
            elif label == "hiphop":
                if pos in (0, 2):
                    L.kick(b)
                if pos in (1, 3):
                    L.snare(b)
                L.hat(b + 0.5, 0.35)
                L.bass(b, root / 2, L.step * 0.8)
            elif label == "pop":
                L.kick(b)
                L.hat(b + 0.5, 0.4)
                if pos in (1, 3):
                    L.snare(b)
            elif label == "garage":
                # shuffled 2-step: kick 1, skipped snare, offbeat licks
                if pos == 0:
                    L.kick(b)
                if pos == 2:
                    L.kick(b + 0.7, 0.6)
                if pos in (1, 3):
                    L.snare(b, 0.8)
                L.hat(b + 0.5, 0.32)
                L.hat(b + 0.25, 0.2)
                L.bass(b if pos == 0 else -1, root / 2, L.step * 0.9)
            elif label == "amapiano":
                # soft four-floor + shaker 16ths + log drum
                L.kick(b, 0.75)
                for off in (0.25, 0.5, 0.75):
                    L.hat(b + off, 0.32)
                L.hat(b + 0.5, 0.4, open_=True)
                if pos in (1, 3):
                    L.snare(b, 0.7)
                if pos in (0, 2):
                    L.bass(b, root / 2, L.step * 0.8)
            elif label == "afro_house":
                L.kick(b, 0.9)
                L.hat(b + 0.5, 0.4, open_=True)
                L.hat(b + 0.25, 0.28)
                L.hat(b + 0.75, 0.3)
                if pos in (1, 3):
                    L.snare(b, 0.7)
                L.bass(b if pos in (0, 2) else -1, root / 2, L.step * 0.7)
            elif label == "jungle":
                if pos in (0, 2) or (b % 8 in (3, 6)):
                    L.kick(b, 0.9)
                if pos in (1, 3) or (b % 8 == 7):
                    L.snare(b)
                for off in (0.25, 0.5, 0.75):
                    L.hat(b + off, 0.3)
                if pos == 0:
                    L.bass(b, root / 2, L.step)
            elif label == "grime":
                if pos == 0:
                    L.kick(b)
                if pos == 2:
                    L.snare(b)
                    L.kick(b, 0.5)
                L.hat(b + 0.5, 0.28)
                L.bass(b if pos == 0 else -1, root / 2, L.step * 1.5)
        out.append((L.done(seed * 7 + i, bare), label))
    return out


# ----------------------------------------------------------------------------
# Features mirror server/ml/rhythm/analyzer.py::rhythm_feature_vector exactly
# (single source of truth lives there; this reuses it).
# ----------------------------------------------------------------------------

def extract_features(y: np.ndarray) -> np.ndarray:
    from server.ml.rhythm.analyzer import rhythm_feature_vector

    feats, _ = rhythm_feature_vector(y, SR)
    return feats


def load_songs_dir(root: Path) -> list[tuple[np.ndarray, str]]:
    """Real songs: <root>/<label>/*.wav|mp3|... — label must be known."""
    from server.ml.rhythm.analyzer import LABEL_TO_GROOVE

    out: list[tuple[np.ndarray, str]] = []
    if not root.is_dir():
        print(f"--songs-dir {root} not found, skipping.")
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        label = sub.name.lower()
        if label not in LABEL_TO_GROOVE or label == "disco":
            print(f"  skip dir '{sub.name}' (unknown label)")
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                continue
            try:
                y, _ = librosa.load(str(f), sr=SR, mono=True, duration=30.0)
                if y.size:
                    out.append((y.astype(np.float32), label))
                    print(f"  + {f.name} -> {label}")
            except Exception as e:
                print(f"  ! {f.name}: {str(e)[:100]}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Audelle groove classifier (RAM-only dataset).")
    parser.add_argument("--songs-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    X: list[np.ndarray] = []
    y: list[str] = []
    print("Synthesizing training loops in RAM ...")
    for label in LABELS:
        # two humanization seeds x polished/bare production styles
        for rep in range(2):
            for bare in (False, True):
                for audio, lab in render_loops(label, args.seed + rep * 977, bare):
                    try:
                        X.append(extract_features(audio))
                        y.append(lab)
                    except Exception as e:
                        print(f"  ! synth {lab}: {str(e)[:100]}")
    print(f"  synth clips: {len(y)}")

    if args.songs_dir:
        print(f"Loading real songs from {args.songs_dir} ...")
        for audio, lab in load_songs_dir(args.songs_dir):
            try:
                X.append(extract_features(audio))
                y.append(lab)
            except Exception as e:
                print(f"  ! song {lab}: {str(e)[:100]}")
        print(f"  total clips: {len(y)}")

    X_arr = np.vstack(X)
    y_arr = np.asarray(y)
    print(f"Feature matrix: {X_arr.shape}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_arr)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    clf = RandomForestClassifier(n_estimators=150, max_depth=12,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_test, y_test))
    print(f"Test accuracy: {acc:.3f} ({len(y_test)} clips held out)")
    print(classification_report(y_test, clf.predict(X_test),
                                target_names=le.classes_.tolist(), zero_division=0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": le.classes_.tolist(),
                 "feature_size": int(X_arr.shape[1]),
                 "test_accuracy": round(acc, 4)}, args.out)
    print(f"Saved {args.out} ({args.out.stat().st_size / 1024:.0f} KB) — loads lazily on demand.")


if __name__ == "__main__":
    main()
