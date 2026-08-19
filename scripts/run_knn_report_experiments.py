"""Run the report's kNN Noisy/Clean experiments on a small GTZAN subset.

Constructs a binary dataset: Clean = GTZAN music segments, Noisy = the same
segments with additive Gaussian noise. Produces the four plots and all the
metrics required by the report's Section VI [Insert: ...] placeholders.

Outputs PNG figures into report_figs/ and prints the numbers.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

BASE = "https://huggingface.co/datasets/storylinez/gtzan-music-genre-dataset/resolve/main"
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
CLIPS_PER_GENRE = 8
SEGMENT_SECONDS = 2.0
SNR_DB = float(__import__("os").environ.get("REPORT_SNR_DB", "15.0"))
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31]
RANDOM_STATE = 42
SR = 22050

OUT_DIR = Path(__file__).resolve().parent / "report_figs"


def download_subset(root: Path):
    paths: list[Path] = []
    session = requests.Session()
    for genre in GENRES:
        gdir = root / genre
        gdir.mkdir(parents=True, exist_ok=True)
        n = 0
        i = 0
        while n < CLIPS_PER_GENRE:
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
            paths.append(dest)
            n += 1
            i += 1
            if i > 500:
                raise RuntimeError(f"could not find clips for {genre}")
    return paths


def features(seg: np.ndarray, sr: int) -> np.ndarray:
    if len(seg) == 0:
        return np.zeros(5)
    rms = float(np.sqrt(np.mean(seg**2)))
    zcr = float(np.mean((np.diff(np.sign(seg)) != 0).astype(float)))
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    if spec.sum() > 0:
        centroid = float(np.sum(freqs * spec) / np.sum(spec))
        flatness = float(np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12))
    else:
        centroid, flatness = 0.0, 0.0
    # Estimated SNR: ratio of low-band (signal-dominant) to high-band energy.
    low = spec[(freqs >= 200) & (freqs <= 4000)]
    high = spec[freqs > 6000]
    est_snr = float(10 * np.log10((np.mean(low**2) + 1e-12) / (np.mean(high**2) + 1e-12)))
    return np.array([rms, zcr, centroid, flatness, est_snr])


def babble_noise(all_clips, clip_idx, seg_len, rng) -> np.ndarray:
    """Background babble: a random segment from a *different* clip, normalised."""
    other = rng.integers(0, len(all_clips))
    while other == clip_idx:
        other = rng.integers(0, len(all_clips))
    src = all_clips[other]
    start = rng.integers(0, max(1, len(src) - seg_len))
    return src[start : start + seg_len].copy()


def main():
    tmp = Path(tempfile.mkdtemp(prefix="audelle-report-"))
    print(f"Downloading {CLIPS_PER_GENRE} clips x {len(GENRES)} genres ...")
    paths = download_subset(tmp)
    print(f"Downloaded {len(paths)} clips")

    X_clean, X_noisy = [], []
    seg_len = int(SEGMENT_SECONDS * SR)
    all_clips: list[np.ndarray] = []
    for p in paths:
        y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly

            y = resample_poly(y, SR, sr).astype(np.float32)
            sr = SR
        all_clips.append(y)

    for cidx, y in enumerate(all_clips):
        nseg = max(1, len(y) // seg_len)
        for i in range(nseg):
            seg = y[i * seg_len : (i + 1) * seg_len]
            X_clean.append(features(seg, sr))
            rng = np.random.default_rng(i * 1000 + cidx)
            noise = babble_noise(all_clips, cidx, seg_len, rng)
            seg_pow = np.mean(seg**2) + 1e-12
            noise_pow = np.mean(noise**2) + 1e-12
            target = seg_pow / (10 ** (SNR_DB / 10))
            noisy = seg + noise * np.sqrt(target / noise_pow)
            X_noisy.append(features(noisy, sr))

    X_clean = np.vstack(X_clean)
    X_noisy = np.vstack(X_noisy)
    print(f"Clean segments: {len(X_clean)}, Noisy segments: {len(X_noisy)}")

    X = np.vstack([X_clean, X_noisy])
    y = np.concatenate([np.zeros(len(X_clean), dtype=int), np.ones(len(X_noisy), dtype=int)])
    feature_names = ["RMS energy", "Zero-crossing rate", "Spectral centroid", "Spectral flatness", "SNR (dB)"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- VI.A: class separation scatter (RMS vs flatness) ---
    plt.figure(figsize=(5, 4))
    plt.scatter(X_clean[:, 0], X_clean[:, 3], s=6, alpha=0.5, label="Clean", color="#1f77b4")
    plt.scatter(X_noisy[:, 0], X_noisy[:, 3], s=6, alpha=0.5, label="Noisy", color="#d62728")
    plt.xlabel("RMS energy")
    plt.ylabel("Spectral flatness")
    plt.legend(loc="best")
    plt.title("Class separation: RMS energy vs spectral flatness")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_scatter.png", dpi=150)
    plt.close()

    # --- scaling + split ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

    # --- VI.B: accuracy vs k (sklearn) ---
    k_range = K_VALUES
    train_acc, test_acc = [], []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(Xtr, ytr)
        train_acc.append(accuracy_score(ytr, clf.predict(Xtr)))
        test_acc.append(accuracy_score(yte, clf.predict(Xte)))

    best_i = int(np.argmax(test_acc))
    best_k = k_range[best_i]
    best_test = test_acc[best_i]

    plt.figure(figsize=(5, 4))
    plt.plot(k_range, train_acc, "o-", label="Train accuracy", color="#1f77b4")
    plt.plot(k_range, test_acc, "s-", label="Test accuracy", color="#d62728")
    plt.axvline(best_k, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.title(f"Accuracy vs k (best k = {best_k})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_acc_vs_k.png", dpi=150)
    plt.close()

    # --- VI.C: custom kNN vs scikit-learn ---
    class CustomKNN:
        def __init__(self, k=5, weighted=False):
            self.k = k
            self.weighted = weighted

        def fit(self, X, y):
            self.X = X
            self.y = np.asarray(y)
            return self

        def _quicksort(self, pairs):
            if len(pairs) <= 1:
                return pairs
            pivot = pairs[len(pairs) // 2]
            lo = [p for p in pairs if p[0] < pivot[0]]
            mid = [p for p in pairs if p[0] == pivot[0]]
            hi = [p for p in pairs if p[0] > pivot[0]]
            return self._quicksort(lo) + mid + self._quicksort(hi)

        def _predict_one(self, x):
            dist = [(float(np.linalg.norm(self.X[i] - x)), int(self.y[i])) for i in range(len(self.X))]
            dist = self._quicksort(dist)[: self.k]
            if self.weighted:
                votes = {}
                for d, label in dist:
                    votes[label] = votes.get(label, 0.0) + (1.0 / (d + 1e-12))
                # tie-break: highest vote wins; first-seen label on exact tie
                return max(votes, key=lambda l: (votes[l], -next(i for i, x in enumerate(self.y) if x == l)))
            counts = {}
            for d, label in dist:
                counts[label] = counts.get(label, 0) + 1
            return max(counts, key=lambda l: (counts[l], -next(i for i, x in enumerate(self.y) if x == l)))

        def predict(self, X):
            return np.array([self._predict_one(x) for x in X])

    custom_test, sk_test = [], []
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
        sk_test.append(accuracy_score(yte, clf.predict(Xte)))
        custom_test.append(accuracy_score(yte, CustomKNN(k).fit(Xtr, ytr).predict(Xte)))

    plt.figure(figsize=(5, 4))
    plt.plot(k_range, sk_test, "o-", label="scikit-learn", color="#2ca02c")
    plt.plot(k_range, custom_test, "s--", label="Custom implementation", color="#9467bd")
    plt.xlabel("k")
    plt.ylabel("Test accuracy")
    plt.legend(loc="best")
    plt.title("Custom kNN vs scikit-learn")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_custom_vs_sklearn.png", dpi=150)
    plt.close()

    # --- VI.D: weighted vs unweighted ---
    unweighted_test, weighted_test = [], []
    for k in k_range:
        unweighted_test.append(accuracy_score(yte, CustomKNN(k, weighted=False).fit(Xtr, ytr).predict(Xte)))
        weighted_test.append(accuracy_score(yte, CustomKNN(k, weighted=True).fit(Xtr, ytr).predict(Xte)))

    plt.figure(figsize=(5, 4))
    plt.plot(k_range, unweighted_test, "o-", label="Unweighted", color="#1f77b4")
    plt.plot(k_range, weighted_test, "s-", label="Weighted (1/d)", color="#ff7f0e")
    plt.xlabel("k")
    plt.ylabel("Test accuracy")
    plt.legend(loc="best")
    plt.title("Weighted vs unweighted kNN")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_weighted.png", dpi=150)
    plt.close()

    # --- VI.E: metrics at best k ---
    clf = KNeighborsClassifier(n_neighbors=best_k).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = {
        "k": best_k,
        "train_acc": round(float(accuracy_score(ytr, clf.predict(Xtr))), 4),
        "test_acc": round(float(accuracy_score(yte, pred)), 4),
        "precision": round(float(precision_score(yte, pred)), 4),
        "recall": round(float(recall_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
        "confusion": confusion_matrix(yte, pred).tolist(),
    }
    per_class = {}
    for name, feats in [("Clean", X_clean), ("Noisy", X_noisy)]:
        per_class[name] = {
            "n": int(len(feats)),
            "mean_rms": round(float(feats[:, 0].mean()), 4),
            "mean_flatness": round(float(feats[:, 3].mean()), 4),
        }

    results = {
        "n_clips": len(paths),
        "n_clean": int(len(X_clean)),
        "n_noisy": int(len(X_noisy)),
        "snr_db": SNR_DB,
        "per_class": per_class,
        "k_values": k_range,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "best_k": best_k,
        "best_test_acc": round(float(best_test), 4),
        "custom_vs_sk": {"sk": sk_test, "custom": custom_test},
        "weighted": {"unweighted": unweighted_test, "weighted": weighted_test},
        "metrics": metrics,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Discarded raw dataset at {tmp}")


if __name__ == "__main__":
    main()