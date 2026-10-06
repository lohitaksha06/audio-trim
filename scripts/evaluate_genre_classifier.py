"""Reproducible evaluation for the genre classifier.

The training script prints numbers to stdout and then throws the audio away,
so nothing in the repository records what the shipped artefact actually scores.
This script re-runs the same pipeline over a cached copy of the data and writes
the result to `server/ml/models/genre_eval.json`, which is what the paper cites.

    python scripts/evaluate_genre_classifier.py --clips-per-genre 10

Cache: `--cache <dir>` keeps the downloaded clips so a re-run costs nothing.
The default lives under the system temp directory, not in git.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_genre_classifier import (  # noqa: E402
    DEFAULT_OUT,
    GENRES,
    download_subset,
    extract_features,
)

SEED = 7  # fixed so the number is reproducible


def evaluate(cache: Path, clips_per_genre: int, out_path: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    paths = download_subset(cache, clips_per_genre)

    X, y, names = [], [], []
    for genre in GENRES:
        for f in sorted(paths[genre]):
            X.append(extract_features(f))
            y.append(genre)
            names.append(f.name)
    X = np.vstack(X)
    y = np.asarray(y)

    counts = {g: int((y == g).sum()) for g in GENRES}
    print(f"clips: {len(y)}  features: {X.shape[1]}  per genre: {counts}")

    clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
    print(f"5-fold CV accuracy: {cv_scores.mean() * 100:.1f}% "
          f"(+/- {cv_scores.std() * 100:.1f}, folds "
          f"{[round(s * 100, 1) for s in cv_scores]})")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    clf.fit(X_tr, y_tr)
    holdout = float(clf.score(X_te, y_te))
    print(f"holdout accuracy:  {holdout * 100:.1f}%  ({len(y_te)} clips)")

    per_genre = {}
    for g in GENRES:
        mask = y_te == g
        if mask.any():
            per_genre[g] = round(float(clf.score(X_te[mask], y_te[mask])), 4)

    result = {
        "dataset": "GTZAN (storylinez/gtzan-music-genre-dataset mirror)",
        "clips": int(len(y)),
        "clips_per_genre": clips_per_genre,
        "genres": GENRES,
        "feature_size": int(X.shape[1]),
        "model": "RandomForestClassifier(n_estimators=300)",
        "seed": SEED,
        "cv_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "cv_folds": [round(float(s), 4) for s in cv_scores],
        "holdout_accuracy": round(holdout, 4),
        "holdout_clips": int(len(y_te)),
        "per_genre_holdout": per_genre,
        "chance": round(1.0 / len(GENRES), 4),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": str(DEFAULT_OUT),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"written -> {out_path}")
    return result


def main() -> None:
    import tempfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-per-genre", type=int, default=10)
    ap.add_argument("--cache", type=Path,
                    default=Path(tempfile.gettempdir()) / "audelle_gtzan")
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_OUT.parent / "genre_eval.json")
    args = ap.parse_args()
    evaluate(args.cache, args.clips_per_genre, args.out)


if __name__ == "__main__":
    main()