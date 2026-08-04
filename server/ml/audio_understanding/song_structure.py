"""Song structure detection (intro / verse / chorus / bridge / outro).

Uses beat-synced chroma+MFCC features and recurrence/subsegment analysis to
find section boundaries, then labels sections by repetition: the most repeated
section is the chorus, lesser-repeated ones are verses, unique middle sections
are bridges, and the first/last sections are intro/outro.
"""

from typing import Any

import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _label_sections(sections: list[dict], repeats: dict[int, int]) -> list[dict]:
    if len(sections) <= 1:
        sections[0]["label"] = "single segment"
        return sections

    idx_by_repeat = sorted(range(len(sections)), key=lambda i: repeats.get(i, 0), reverse=True)
    chorus_idx = idx_by_repeat[0]
    verse_idx = next((i for i in idx_by_repeat if i != chorus_idx), None)

    for i, sec in enumerate(sections):
        if i == 0:
            label = "intro"
        elif i == len(sections) - 1:
            label = "outro"
        elif verse_idx is None:
            label = "chorus"
        elif repeats.get(i, 0) >= repeats.get(verse_idx, 1):
            label = "verse"
        else:
            label = "bridge"
        sections[i]["label"] = label

    return sections


def detect_structure(audio_path: str, max_sections: int = 8) -> dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 4.0:
        return {
            "sections": [{"start": 0.0, "end": round(duration, 2), "label": "single segment"}],
            "structure": "section detection (heuristic)",
            "duration_seconds": round(duration, 2),
        }

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    feat = np.vstack([chroma, mfcc, delta])

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beats = np.atleast_1d(beats).astype(int)

    if len(beats) > 2 and not np.isnan(tempo):
        sync_feat = librosa.util.sync(feat, beats, aggregate=np.mean)
        frame_map = beats  # index of first original frame per beat-frame
        n_frames = sync_feat.shape[1]
    else:
        sync_feat = feat
        frame_map = np.arange(feat.shape[1])
        n_frames = feat.shape[1]

    n_segments = min(max_sections, max(2, int(duration // 12)))
    n_segments = min(n_segments, n_frames - 1)

    try:
        bounds = librosa.segment.subsegment(
            sync_feat, frames=np.arange(n_frames), n_segments=n_segments
        )
        bounds = np.unique(np.concatenate([[0], bounds, [n_frames - 1]]).astype(int))
    except Exception:
        cuts = np.linspace(0, n_frames, n_segments + 1).astype(int)
        bounds = np.unique(np.concatenate([[0], cuts, [n_frames - 1]]).astype(int))

    sections: list[dict] = []
    seg_feats: list[np.ndarray] = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        a_orig = int(frame_map[min(a, len(frame_map) - 1)])
        b_orig = int(frame_map[min(b, len(frame_map) - 1)])
        start = round(a_orig * hop / sr, 2)
        end = round(min(b_orig + 1, feat.shape[1] - 1) * hop / sr, 2)
        if end - start < 0.1:
            continue
        seg_feats.append(sync_feat[:, a:b].mean(axis=1))
        sections.append({"start": start, "end": end, "label": "segment"})

    sim = cosine_similarity(np.asarray(seg_feats))
    repeats: dict[int, int] = {}
    for i in range(len(sections)):
        others = [sim[i][j] for j in range(len(sections)) if j != i]
        repeats[i] = int(sum(1 for s in others if s > 0.8))

    sections = _label_sections(sections, repeats)

    return {
        "sections": sections,
        "structure": "section detection (heuristic)",
        "duration_seconds": round(duration, 2),
    }