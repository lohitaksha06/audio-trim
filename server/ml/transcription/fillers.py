"""Disfluency (filler / hesitation) detection and removal.

Why acoustic rather than ASR-lexical: matching the literal word "um" needs
word-level transcription, which is not reliably available here (Whisper's
`return_timestamps="word"` does not yield word offsets in this transformers
build). Hesitation detection is also what commercial cleanup tools actually do —
it catches filled pauses ("uhh… so") without depending on the ASR lexicon, and it
transfers across accents and speakers.

Pipeline: frame energy -> adaptive speech floor -> speech regions -> interior
pauses and low-energy voiced blobs inside them -> candidate spans -> splice with
a short crossfade so joins do not click.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_MS = 20.0


@dataclass(frozen=True)
class DisfluencySpan:
    start: float
    end: float
    kind: str  # "pause" | "blab"

    @property
    def duration(self) -> float:
        return self.end - self.start


def _frame_db(y: np.ndarray, sr: int, frame_ms: float) -> tuple[np.ndarray, float]:
    """Per-frame RMS in dBFS plus the hop in seconds."""
    hop = max(1, int(sr * frame_ms / 1000.0))
    n = max(1, y.shape[-1] // hop)
    trimmed = y[..., : n * hop]
    frames = trimmed.reshape(n, hop, -1).mean(axis=-1) if trimmed.ndim > 2 else trimmed.reshape(n, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=-1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    return db, hop / sr


def _regions(mask: np.ndarray, hop_s: float, max_gap_s: float) -> list[tuple[float, float]]:
    """Contiguous True runs, bridged across gaps up to ``max_gap_s``."""
    out: list[tuple[float, float]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        j = i
        last_on = i
        while j < n:
            if mask[j]:
                last_on = j
            elif (j - last_on) * hop_s > max_gap_s:
                break
            j += 1
        out.append((start * hop_s, (last_on + 1) * hop_s))
        i = j + 1
    return out


def detect_disfluencies(
    y: np.ndarray,
    sr: int,
    min_pause: float = 0.35,
    max_pause: float = 0.9,
    pad: float = 0.02,
    dynamic_range_db: float = 22.0,
) -> list[DisfluencySpan]:
    """Find hesitation spans inside speech.

    ``min_pause`` is the shortest *removable* gap. It defaults to 0.35s because
    conversational inter-word and inter-phrase pauses routinely run 0.1-0.3s;
    a lower floor removes normal speech rhythm along with the fillers (measured:
    a 0.12s floor cut 45% of a real 10s utterance). ``max_pause`` keeps
    genuinely long intentional beats unless the caller asks otherwise.
    """
    mono = y[0] if np.ndim(y) > 1 else y
    if mono.size < sr // 10:
        return []

    db, hop_s = _frame_db(np.asarray(mono, dtype=np.float32), sr, FRAME_MS)

    noise_floor = float(np.percentile(db, 10))
    speech_level = float(np.percentile(db, 90))
    threshold = max(noise_floor + dynamic_range_db, speech_level - 30.0)
    if speech_level - noise_floor < 8.0:
        return []  # near-silent or uniform material: nothing reliable to cut

    # Bridge gaps up to ``max_pause`` when grouping speech, so anything in the
    # removable range stays *inside* a region and can be found as an interior
    # gap. Larger gaps split the region and are therefore preserved.
    speech = _regions(db > threshold, hop_s, max_gap_s=max_pause)
    spans: list[DisfluencySpan] = []

    for start, end in speech:
        a = int(round(start / hop_s))
        b = int(round(end / hop_s))
        if b - a < 3:
            continue
        seg = db[a:b]
        inner_speech = seg > threshold

        # Interior gaps long enough to be hesitations, short enough to be filler.
        for g0_s, g1_s in _regions(~inner_speech, hop_s, max_gap_s=0.0):
            dur = g1_s - g0_s
            if min_pause <= dur <= max_pause:
                spans.append(DisfluencySpan(start=g0_s, end=g1_s, kind="pause"))
                continue
            # A gap shorter than the pause floor may still be a filled
            # hesitation ("uhh") rather than clean silence. Only treat it as one
            # when it is long enough to be a word-ish sound AND clearly below
            # speech level — otherwise it is ordinary inter-word spacing and
            # removing it eats the speaker's rhythm.
            g0 = max(0, int(round(g0_s / hop_s)))
            g1 = max(0, int(round(g1_s / hop_s)))
            blob = seg[g0:g1]
            if (
                blob.size
                and dur >= max(0.18, min_pause * 0.6)
                and float(blob.max()) < threshold - 10.0
            ):
                spans.append(DisfluencySpan(start=g0_s, end=g1_s, kind="blab"))

    # Merge spans that overlap or sit within one frame of each other, then pad.
    spans.sort(key=lambda s: s.start)
    merged: list[DisfluencySpan] = []
    for s in spans:
        if merged and s.start <= merged[-1].end + hop_s:
            prev = merged[-1]
            merged[-1] = DisfluencySpan(prev.start, max(prev.end, s.end), prev.kind)
        else:
            merged.append(s)

    total = len(np.asarray(mono)) / sr
    out: list[DisfluencySpan] = []
    for s in merged:
        st = max(0.0, s.start - pad)
        en = min(total, s.end + pad)
        if en - st >= 0.05:
            out.append(DisfluencySpan(st, en, s.kind))
    return out


def remove_spans(
    y: np.ndarray,
    sr: int,
    spans: list[DisfluencySpan],
    xfade_ms: float = 8.0,
) -> np.ndarray:
    """Splice out ``spans`` with a short equal-gain crossfade at each join."""
    if not spans:
        return y
    xf = max(1, int(sr * xfade_ms / 1000.0))
    ordered = sorted(spans, key=lambda s: s.start)

    pieces: list[np.ndarray] = []
    cursor = 0
    n_samples = y.shape[-1]
    for s in ordered:
        a = int(round(s.start * sr))
        b = int(round(s.end * sr))
        # Overlapping/adjacent spans: trim to what is still unconsumed. A span
        # starting exactly at the cursor must still be cut — the old `a <=
        # cursor` guard skipped it and left leading audio in place.
        if a < cursor:
            a = cursor
        b = min(max(b, a + 1), n_samples)
        if b <= cursor:
            continue
        if a > cursor:
            pieces.append(y[..., cursor:a])
        cursor = b

    tail = y[..., cursor:]
    if len(tail):
        pieces.append(tail)
    if not pieces:
        return y[..., :0]

    out = pieces[0]
    for piece in pieces[1:]:
        n = min(out.shape[-1], piece.shape[-1])
        if n < xf or out.shape[-1] < xf:
            out = np.concatenate([out, piece], axis=-1)
            continue
        fade_out = out[..., -xf:].copy()
        fade_in = piece[..., :xf].copy()
        ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
        # (1, xf) so it broadcasts across channels, not (xf, 1).
        ramp = ramp.reshape((1, -1)) if out.ndim > 1 else ramp
        blend = fade_out * (1 - ramp) + fade_in * ramp
        out = np.concatenate([out[..., :-xf], blend, piece[..., xf:]], axis=-1)

    return np.clip(out, -1.0, 1.0).astype(np.float32)