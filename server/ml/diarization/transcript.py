"""Speaker-attributed transcription and speaker-level editing.

``diarization.diarize`` already produced speaker segments but nothing in the
prompt pipeline could reach them, so the podcast page had nothing real to show.
This module joins speaker segments with the transcriber so you get a real,
speaker-labelled transcript, and provides the span maths for splitting,
keeping or removing a single speaker.

Speaker labels come from clustering, so they are anonymous and only stable
within one file. Nothing here claims identity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from server.ml.diarization.diarization import diarize


def _load_mono(audio_path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    import librosa

    y, sr_out = librosa.load(audio_path, sr=sr, mono=True)
    return y, sr_out


def _transcribe_segment(y: np.ndarray, sr: int, start: float, end: float) -> str:
    """Transcribe one time range. Returns '' on any failure (never raises)."""
    try:
        import librosa

        from server.ml.transcription.transcriber import Transcriber

        a = int(start * sr)
        b = min(int(end * sr), len(y))
        if b - a < int(0.15 * sr):
            return ""
        seg = y[a:b]
        if float(np.max(np.abs(seg))) < 1e-4:
            return ""
        # Whisper expects >= ~0.1s and benefits from a little padding.
        result = Transcriber().transcribe_array(seg, sr)
        return (result.get("text") or "").strip()
    except Exception:
        return ""


def transcribe_with_speakers(
    audio_path: str,
    max_speakers: int = 4,
    include_text: bool = True,
) -> dict[str, Any]:
    """Diarized speaker segments, each with its transcript."""
    dia = diarize(audio_path, max_speakers=max_speakers)
    segments = dia.get("segments") or []
    y, sr = _load_mono(audio_path)

    out: list[dict[str, Any]] = []
    for seg in segments:
        item = {
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg.get("duration", round(seg["end"] - seg["start"], 2)),
        }
        if include_text:
            item["text"] = _transcribe_segment(y, sr, seg["start"], seg["end"])
        out.append(item)

    speakers = sorted({s["speaker"] for s in out})
    # Real talk time per speaker, not just a segment count.
    talk_time: dict[str, float] = {s: 0.0 for s in speakers}
    for s in out:
        talk_time[s["speaker"]] = round(talk_time[s["speaker"]] + s["duration"], 2)

    return {
        "segments": out,
        "speakers": speakers,
        "talk_time_seconds": talk_time,
        "speech_duration_seconds": dia.get("speech_duration_seconds", 0.0),
        "duration_seconds": dia.get("duration_seconds", 0.0),
        "method": dia.get("method", ""),
    }


def speaker_spans(
    transcript: dict[str, Any], keep: set[str] | None = None
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split segments into (keep_spans, drop_spans) for a set of speakers."""
    keep_spans: list[tuple[float, float]] = []
    drop_spans: list[tuple[float, float]] = []
    for seg in transcript.get("segments", []):
        span = (float(seg["start"]), float(seg["end"]))
        if keep is None or seg["speaker"] in keep:
            keep_spans.append(span)
        else:
            drop_spans.append(span)
    return keep_spans, drop_spans


def complement_spans(
    spans: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """Inverse of `spans` over [0, duration] — the parts to cut."""
    if not spans:
        return []
    ordered = sorted(spans)
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in ordered:
        if a > cursor:
            out.append((cursor, min(a, duration)))
        cursor = max(cursor, b)
    if cursor < duration:
        out.append((cursor, duration))
    return [(a, b) for a, b in out if b - a > 0.02]


def chapters_from_transcript(
    transcript: dict[str, Any],
    min_gap: float = 3.0,
    max_chapters: int = 12,
) -> list[dict[str, Any]]:
    """Chapter markers from long pauses between speech segments.

    Uses real pause positions rather than inventing evenly spaced marks.
    """
    segs = sorted(transcript.get("segments", []), key=lambda s: s["start"])
    if not segs:
        return []
    marks = [{"time": 0.0, "label": "Start"}]
    for prev, cur in zip(segs, segs[1:]):
        gap = float(cur["start"]) - float(prev["end"])
        if gap >= min_gap:
            first_word = (cur.get("text") or "").split()
            label = " ".join(first_word[:6]) if first_word else "New section"
            marks.append({"time": float(cur["start"]), "label": label})
    marks = marks[:max_chapters]
    total = float(transcript.get("duration_seconds") or 0.0)
    for i, m in enumerate(marks):
        m["index"] = i
        m["percent"] = round(100 * m["time"] / total, 1) if total else 0.0
    return marks