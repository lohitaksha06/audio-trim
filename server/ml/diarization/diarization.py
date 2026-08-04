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
    """Return (start_sample, end_sample) of speech-like regions."""
    rms = librosa.feature.rms(y=y, hop_length=FRAME_HOP)[0]
    sf = librosa.feature.spectral_flatness(y=y, hop_length=FRAME_HOP)[0]

    floor = np.percentile(rms, 40)
    active = (rms > floor * 1.5) & (sf < 0.5)

    segments: list[tuple[int, int]] = []
    in_speech = False
    start = 0
    for i, on in enumerate(active):
        if on and not in_speech:
            in_speech = True
            start = i
        elif not on and in_speech:
            if i - start >= int(MIN_SPEECH_S * sr / FRAME_HOP):
                segments.append((start * FRAME_HOP, (i + 1) * FRAME_HOP))
            in_speech = False
    if in_speech and len(active) - start >= int(MIN_SPEECH_S * sr / FRAME_HOP):
        segments.append((start * FRAME_HOP, len(active) * FRAME_HOP))

    return segments


def _embedding(y: np.ndarray, sr: int, start: int, end: int) -> np.ndarray:
    seg = y[start:end]
    if seg.size == 0:
        return np.zeros(13)
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)


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

    n = min(max_speakers, len(segments))
    best_k, best_score, best_labels = 1, -1.0, np.zeros(len(segments), dtype=int)
    for k in range(2, n + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        if len(embeddings) < k + 2:
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