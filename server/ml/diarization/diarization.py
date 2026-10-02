"""Speaker diarization ("who spoke when").

Lightweight implementation that runs fully on CPU with no model downloads:
  1. Voice-activity detection via energy + spectral flatness.
  2. Per-segment MFCC embeddings.
  3. Unsupervised clustering (KMeans, K chosen by silhouette score) to
     separate speakers.
  4. Adjacent same-speaker segments are merged.

The design allows swapping in pyannote.audio later by replacing ``diarize``
with the same return schema.
"""

from typing import Any

import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MIN_SPEECH_S = 0.5
FRAME_HOP = 512


def _vad_segments(
    y: np.ndarray, sr: int
) -> list[tuple[int, int]]:
    """Return (start_sample, end_sample) of speech-like regions.

    Thresholding is done in the log domain against an adaptive noise floor.
    The previous rule (``rms > percentile(rms, 40) * 1.5``) failed on almost any
    recording whose speech occupies more than ~60% of the file: the 40th
    percentile then lands mid-speech, so the multiplier excluded real speech and
    diarization returned zero segments.
    """
    rms = librosa.feature.rms(y=y, hop_length=FRAME_HOP)[0]
    if rms.size == 0:
        return []
    rms_db = 20.0 * np.log10(rms + 1e-10)

    noise_db = float(np.percentile(rms_db, 10))
    peak_db = float(np.percentile(rms_db, 90))
    if peak_db - noise_db < 6.0:  # flat noise or near-silence
        return []
    threshold = noise_db + max(5.0, (peak_db - noise_db) * 0.25)
    active = rms_db > threshold

    min_frames = max(1, int(MIN_SPEECH_S * sr / FRAME_HOP))
    hangover = max(1, int(0.25 * sr / FRAME_HOP))  # keep short pauses inside a turn

    raw: list[tuple[int, int]] = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        start = i
        last_on = i
        j = i
        while j < n:
            if active[j]:
                last_on = j
            elif j - last_on > hangover:
                break
            j += 1
        if last_on + 1 - start >= min_frames:
            raw.append((start, last_on + 1))
        i = last_on + 1

    return [(s * FRAME_HOP, min(e * FRAME_HOP, len(y))) for s, e in raw]


def _embedding(y: np.ndarray, sr: int, start: int, end: int) -> np.ndarray:
    """Mean + std MFCC over a segment (26-dim).

    Mean alone under-separated similar voices; adding the std captures delivery
    variation and measurably sharpens the clusters.
    """
    seg = y[start:end]
    if seg.size == 0:
        return np.zeros(26)
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
    if mfcc.shape[1] == 0:
        return np.zeros(26)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def diarize(
    audio_path: str, max_speakers: int = 4
) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    segments = _vad_segments(y, sr)
    if not segments:
        return {
            "segments": [],
            "speakers": [],
            "speech_duration_seconds": 0.0,
            "duration_seconds": round(duration, 2),
        }

    embeddings = np.stack([_embedding(y, sr, s, e) for s, e in segments])

    # k is limited by the segment count: silhouette needs at least k+1 samples.
    # The old `k + 2` guard meant a 3-segment clip could never produce 2
    # speakers, so short interviews silently collapsed to a single speaker.
    n = min(max_speakers, len(segments) - 1)
    best_k, best_score, best_labels = 1, -1.0, np.zeros(len(segments), dtype=int)
    for k in range(2, n + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(embeddings, labels)
        except ValueError:
            continue
        if score > best_score:
            best_score, best_labels, best_k = score, labels, k
    if best_k == 1:
        best_labels = np.zeros(len(segments), dtype=int)

    out: list[dict] = []
    last_speaker = None
    for (start_s, end_s), label in zip(segments, best_labels):
        if last_speaker == int(label) and out:
            out[-1]["end"] = round(end_s / sr, 2)
            out[-1]["duration"] = round(out[-1]["end"] - out[-1]["start"], 2)
            continue
        out.append(
            {
                "start": round(start_s / sr, 2),
                "end": round(end_s / sr, 2),
                "speaker": f"SPK_{int(label)}",
            }
        )
        out[-1]["duration"] = round(out[-1]["end"] - out[-1]["start"], 2)
        last_speaker = int(label)

    speakers = sorted({s["speaker"] for s in out})
    return {
        "segments": out,
        "speakers": speakers,
        "speech_duration_seconds": round(sum(s["duration"] for s in out), 2),
        "duration_seconds": round(duration, 2),
        "method": "VAD + MFCC clustering",
    }