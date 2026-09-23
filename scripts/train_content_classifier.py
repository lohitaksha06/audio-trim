"""Train the audio-content + noise-condition classifier and SNR quality regressor.

Supervised, discriminative ML (NOT generative):
  - Classifier: RandomForestClassifier over 6 conditions
    {clean, hiss, hum, crowd, clipped, muffled}
  - Regressor: RandomForestRegressor predicting effective SNR_dB (quality)

Dataset (real + synthetic, temp-only, zero permanent storage):
  - Real base: GTZAN subset from Hugging Face
    `storylinez/gtzan-music-genre-dataset` (same mirror as
    scripts/train_genre_classifier.py), `--clips-per-genre` clips x 10 genres,
    first 10 s of each clip, downloaded to a temp dir and deleted after.
  - Synthetic degradations applied in RAM per clip (labeled, deterministic
    seeds; SNR sampled per clip so the regressor learns a continuum):
    hiss (white noise @8–18dB), hum (50+100+150Hz @5–15dB), crowd/babble
    (other clip @3–12dB), clipped (gain+hard-clip), muffled (FFT lowpass 1kHz).

Features (84-dim, fixed): genre 75-dim base (MFCC20 mean/std, chroma mean/std,
centroid/bandwidth/rolloff/ZCR/RMS mean/std, tempo) + 9 noise-specific
(flatness mean/std, hum-band ratio, hiss-band ratio, clip ratio, crest_dB,
hum-peak prominence, noise-floor ratio, spectral-contrast mean).

Usage:
    python scripts/train_content_classifier.py [--clips-per-genre 6] [--out ...]

Artifact (~1-2 MB) loads lazily in
server/ml/audio_understanding/content_classifier.py only when /api/ml/understand
runs; missing artifact -> content=None, everything else keeps working.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

DATASET_OWNER = "storylinez"
DATASET_NAME = "gtzan-music-genre-dataset"
GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
BASE = f"https://huggingface.co/datasets/{DATASET_OWNER}/{DATASET_NAME}/resolve/main"

SR = 22050
SEG_SECONDS = 10.0

CONDITIONS = ["clean", "hiss", "hum", "crowd", "clipped", "muffled"]
# Effective SNR targets for the regressor (dB). Clean capped at 40.
CONDITION_SNR = {"clean": 40.0, "hiss": 12.0, "hum": 8.0, "crowd": 6.0, "clipped": 12.0, "muffled": 18.0}

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "server" / "ml" / "models" / "content_classifier.joblib"


def download_subset(root: Path, clips_per_genre: int) -> list[Path]:
    paths: list[Path] = []
    session = requests.Session()
    for genre in GENRES:
        gdir = root / genre
        gdir.mkdir(parents=True, exist_ok=True)
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
            paths.append(dest)
            n += 1
            i += 1
            if i > 500:
                raise RuntimeError(f"could not find {clips_per_genre} clips for {genre}")
    return paths


def _at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    # Scale noise to reach target SNR, randomize polarity/phase slightly.
    sig_pow = float(np.mean(signal ** 2)) + 1e-12
    noise_pow = float(np.mean(noise ** 2)) + 1e-12
    target = sig_pow / (10.0 ** (snr_db / 10.0))
    mixed = signal + noise * np.sqrt(target / noise_pow)
    peak = float(np.max(np.abs(mixed))) + 1e-12
    if peak > 0.99:
        mixed = (mixed / peak * 0.89).astype(np.float32)
    return mixed.astype(np.float32)


def degrade(y: np.ndarray, condition: str, rng: np.random.Generator, donor: np.ndarray | None) -> tuple[np.ndarray, float]:
    """Returns (audio, true_effective_snr_db). SNR is sampled per clip so the
    regressor learns a continuous mapping, not 6 discrete values."""
    y = y.astype(np.float32)
    if condition == "clean":
        return y, 40.0
    if condition == "hiss":
        snr = float(rng.uniform(8.0, 18.0))
        noise = rng.standard_normal(len(y)).astype(np.float32)
        return _at_snr(y, noise, snr, rng), snr
    if condition == "hum":
        snr = float(rng.uniform(5.0, 15.0))
        t = np.arange(len(y)) / SR
        hum = (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
               + 0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        return _at_snr(y, hum, snr, rng), snr
    if condition == "crowd":
        snr = float(rng.uniform(3.0, 12.0))
        if donor is None or donor.size < len(y):
            noise = rng.standard_normal(len(y)).astype(np.float32)
        else:
            start = int(rng.integers(0, max(1, donor.size - len(y))))
            noise = donor[start:start + len(y)].copy()
        return _at_snr(y, noise, snr, rng), snr
    if condition == "clipped":
        hot = y * 3.0
        return np.clip(hot, -1.0, 1.0).astype(np.float32), float(rng.uniform(11.0, 13.0))
    if condition == "muffled":
        S = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), d=1.0 / SR)
        S[freqs > 1000.0] *= 0.05
        muff = np.fft.irfft(S, n=len(y)).astype(np.float32)
        peak = float(np.max(np.abs(muff))) + 1e-12
        return (muff / peak * 0.89 * max(float(np.max(np.abs(y))), 0.5)).astype(np.float32), float(rng.uniform(17.0, 19.0))
    raise ValueError(condition)


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """84-dim feature vector. MUST mirror content_classifier._extract_features."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(84, dtype=np.float32)
    feats: list[float] = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feats.extend(np.mean(mfcc, axis=1))
    feats.extend(np.std(mfcc, axis=1))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.extend(np.mean(chroma, axis=1))
    feats.extend(np.std(chroma, axis=1))
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats.extend([float(np.mean(cent)), float(np.std(cent))])
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats.extend([float(np.mean(bw)), float(np.std(bw))])
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats.extend([float(np.mean(roll)), float(np.std(roll))])
    zcr = librosa.feature.zero_crossing_rate(y)
    feats.extend([float(np.mean(zcr)), float(np.std(zcr))])
    rms = librosa.feature.rms(y=y)
    feats.extend([float(np.mean(rms)), float(np.std(rms))])
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.append(float(np.atleast_1d(tempo)[0]))
    except Exception:
        feats.append(120.0)
    # --- 6 noise-specific features ---
    flat = librosa.feature.spectral_flatness(y=y)
    feats.extend([float(np.mean(flat)), float(np.std(flat))])
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    total = float(np.sum(S ** 2)) + 1e-12
    hum_band = (freqs >= 40) & (freqs <= 130)
    hiss_band = freqs >= 6000
    feats.append(float(np.sum(S[hum_band] ** 2) / total))
    feats.append(float(np.sum(S[hiss_band] ** 2) / total))
    feats.append(float(np.mean(np.abs(y) >= 0.99)))
    peak = float(np.max(np.abs(y))) + 1e-12
    rms_v = float(np.sqrt(np.mean(y ** 2))) + 1e-12
    feats.append(float(20 * np.log10(peak / rms_v)))
    # --- 3 targeted discriminative features ---
    # hum-peak prominence: mains lines vs surrounding low band
    hum_lines = ((freqs >= 48) & (freqs <= 52)) | ((freqs >= 97) & (freqs <= 103)) | ((freqs >= 147) & (freqs <= 153))
    hum_sur = (freqs >= 40) & (freqs <= 160) & (~hum_lines)
    hum_peak = float(np.sum(S[hum_lines] ** 2) + 1e-12) / (float(np.sum(S[hum_sur] ** 2)) + 1e-12)
    feats.append(float(hum_peak))
    # noise floor: babble/crowd fills pauses -> high min/median frame energy
    frame_rms = librosa.feature.rms(y=y)[0] + 1e-12
    feats.append(float(np.percentile(frame_rms, 10) / (np.median(frame_rms) + 1e-12)))
    # spectral contrast mean: clean music has strong peak/valley structure
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats.append(float(np.mean(contrast)))
    except Exception:
        feats.append(0.0)
    assert len(feats) == 84, len(feats)
    return np.asarray(feats, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Audelle content/noise classifier + SNR regressor (temp-only dataset).")
    parser.add_argument("--clips-per-genre", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--keep-data", action="store_true")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="audelle-content-"))
    print(f"Downloading {args.clips_per_genre} clips x {len(GENRES)} genres into {tmp} ...")
    paths = download_subset(tmp, args.clips_per_genre)

    # Load base clips (first SEG_SECONDS each) into RAM.
    bases: list[np.ndarray] = []
    for p in paths:
        try:
            y, _ = librosa.load(str(p), sr=SR, mono=True, duration=SEG_SECONDS)
            if y.size > SR * 2:
                bases.append(y.astype(np.float32))
        except Exception as e:
            print(f"  ! {p.name}: {str(e)[:100]}")
    print(f"Loaded {len(bases)} base clips.")
    if len(bases) < 10:
        raise RuntimeError("too few base clips downloaded")

    rng = np.random.default_rng(42)
    X: list[np.ndarray] = []
    y_cond: list[str] = []
    y_snr: list[float] = []
    groups: list[int] = []
    for idx, base in enumerate(bases):
        donor = bases[(idx + 7) % len(bases)]
        for cond in CONDITIONS:
            try:
                aug, true_snr = degrade(base, cond, rng, donor)
                X.append(extract_features(aug, SR))
                y_cond.append(cond)
                y_snr.append(true_snr)
                groups.append(idx)
            except Exception as e:
                print(f"  ! aug {cond}: {str(e)[:100]}")
            except Exception as e:
                print(f"  ! aug {cond}: {str(e)[:100]}")
    X_arr = np.vstack(X)
    print(f"Feature matrix: {X_arr.shape} ({len(CONDITIONS)} conditions)")

    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y_cond))
    y_snr_arr = np.asarray(y_snr, dtype=np.float32)
    groups_arr = np.asarray(groups)

    # Split BY BASE CLIP (GroupShuffleSplit): all 6 degradations of one clip
    # stay on one side. Splitting augmented rows would leak musical content
    # across the split and inflate accuracy.
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X_arr, y_enc, groups_arr))
    Xtr, Xte, ytr, yte, str_, ste = X_arr[tr_idx], X_arr[te_idx], y_enc[tr_idx], y_enc[te_idx], y_snr_arr[tr_idx], y_snr_arr[te_idx]
    print(f"Split by clip: {len(set(groups_arr[tr_idx]))} train clips, {len(set(groups_arr[te_idx]))} test clips")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    print(f"Classifier test accuracy: {acc:.3f} ({len(yte)} held out)")
    print(classification_report(yte, clf.predict(Xte), target_names=le.classes_.tolist(), zero_division=0))
    print("Confusion (rows=true):")
    print(confusion_matrix(yte, clf.predict(Xte)))

    # Regressor: only conditions with a defined true SNR (additive noises +
    # clean anchor). Clipped/muffled have no physical SNR, so including them
    # would force the regressor to memorize condition means (R2 <= 0).
    le_names = le.classes_
    reg_mask_tr = np.array([le_names[c] in ("clean", "hiss", "hum", "crowd") for c in ytr])
    reg_mask_te = np.array([le_names[c] in ("clean", "hiss", "hum", "crowd") for c in yte])
    reg = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    reg.fit(Xtr[reg_mask_tr], str_[reg_mask_tr])
    pred_snr = reg.predict(Xte[reg_mask_te])
    mae = float(mean_absolute_error(ste[reg_mask_te], pred_snr))
    r2 = float(r2_score(ste[reg_mask_te], pred_snr))
    print(f"Regressor (clean/hiss/hum/crowd) MAE: {mae:.2f} dB | R2: {r2:.3f} ({int(reg_mask_te.sum())} held out)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "clf": clf, "reg": reg, "classes": le.classes_.tolist(),
        "feature_size": int(X_arr.shape[1]),
        "test_accuracy": round(acc, 4), "mae_db": round(mae, 3), "r2": round(r2, 4),
        "conditions": CONDITIONS,
    }, args.out)
    print(f"Saved {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")

    if not args.keep_data:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"Discarded raw dataset at {tmp}")


if __name__ == "__main__":
    main()
