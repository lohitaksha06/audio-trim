"""Train a lightweight music-genre classifier on a small GTZAN subset.

Downloads a small balanced subset (10 clips x 10 genres, ~130 MB) from the
Hugging Face `storylinez/gtzan-music-genre-dataset` mirror, extracts librosa
features, trains a scikit-learn classifier, and saves the artifact so the
server can predict genre for uploaded audio.

Usage:
    python scripts/train_genre_classifier.py [--clips-per-genre 10] [--out ...]

The raw dataset is downloaded to a temp dir and deleted after training by
default (--keep-data keeps it for debugging).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_OWNER = "storylinez"
DATASET_NAME = "gtzan-music-genre-dataset"
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
BASE = f"https://huggingface.co/datasets/{DATASET_OWNER}/{DATASET_NAME}/resolve/main"

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "server" / "ml" / "models" / "genre_classifier.joblib"


def download_subset(root: Path, clips_per_genre: int) -> dict[str, list[Path]]:
    """Download `clips_per_genre` clips per genre into `root/<genre>/<file>.wav`."""
    paths: dict[str, list[Path]] = {}
    session = requests.Session()
    for genre in GENRES:
        gdir = root / genre
        gdir.mkdir(parents=True, exist_ok=True)
        # Reuse the known indexing of the mirror dataset (files: <genre>.<NNNNN>.wav).
        # Download exactly clips_per_genre entries.
        n = 0
        i = 0
        while n < clips_per_genre:
            name = f"{genre}.{i:05d}.wav"
            url = f"{BASE}/{genre}/{name}"
            dest = gdir / name
            if not dest.exists():
                r = session.get(url, timeout=60)
                if r.status_code != 200:
                    if r.status_code == 404:
                        i += 1
                        continue
                    raise RuntimeError(f"download {url}: HTTP {r.status_code}")
                dest.write_bytes(r.content)
            paths.setdefault(genre, []).append(dest)
            n += 1
            i += 1
            # Guard against infinite loops if the mirror naming differs.
            if i > 500:
                raise RuntimeError(f"could not find {clips_per_genre} clips for {genre}")
    return paths


def extract_features(audio_path: Path) -> np.ndarray:
    """Extract a fixed-length feature vector from an audio file."""
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=10.0)
    if y.size == 0:
        return np.zeros(1)
    features: list[float] = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([float(np.mean(spectral_centroid)), float(np.std(spectral_centroid))])

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth))])

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend([float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff))])

    zero_crossing = librosa.feature.zero_crossing_rate(y)
    features.extend([float(np.mean(zero_crossing)), float(np.std(zero_crossing))])

    rms = librosa.feature.rms(y=y)
    features.extend([float(np.mean(rms)), float(np.std(rms))])

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(np.atleast_1d(tempo)[0]))

    return np.asarray(features, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Audelle genre classifier on a small GTZAN subset.")
    parser.add_argument("--clips-per-genre", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--keep-data", action="store_true", help="keep raw clips in temp (for debugging)")
    args = parser.parse_args()

    clips_per_genre = args.clips_per_genre
    tmp = Path(tempfile.mkdtemp(prefix="audelle-gtzan-"))
    print(f"Downloading {clips_per_genre} clips x {len(GENRES)} genres into {tmp} ...")
    paths = download_subset(tmp, clips_per_genre)

    X: list[np.ndarray] = []
    y: list[str] = []
    for genre, files in paths.items():
        for f in files:
            X.append(extract_features(f))
            y.append(genre)

    X = np.vstack(X)
    y_arr = np.asarray(y)
    print(f"Extracted {X.shape[0]} samples, {X.shape[1]} features.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_arr)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_test, y_test))
    cv = float(np.mean(cross_val_score(clf, X, y_enc, cv=5)))
    print(f"Test accuracy: {acc:.3f} | 5-fold CV accuracy: {cv:.3f}")

    # Persist pipeline: classes + model.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": clf, "classes": le.classes_.tolist(), "feature_size": int(X.shape[1])}
    joblib.dump(payload, args.out)
    print(f"Saved model artifact to {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")

    if not args.keep_data:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"Discarded raw dataset at {tmp}")


if __name__ == "__main__":
    main()