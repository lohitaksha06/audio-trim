"""Named chord parsing and a strummed chord renderer.

The prompt engine can already add an "in-key triad", but real users ask for
specific chords: "add guitar chords: Am F C G", "play Cmaj7 then Am7", "use a
C/G". This module turns those names into MIDI notes and a strummed part.

It is deliberately small and deterministic. No chord-name model is involved;
the grammar below covers the common popular-music vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NOTE_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
PC_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Canonical quality -> the suffix people type / expect to see.
DISPLAY = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug",
    "sus2": "sus2", "sus4": "sus4",
    "maj7": "maj7", "min7": "m7", "dom7": "7",
    "m7b5": "m7b5", "dim7": "dim7",
    "maj9": "maj9", "min9": "m9", "dom9": "9",
    "maj11": "maj11", "min11": "m11", "dom11": "11",
    "dom13": "13", "min13": "m13",
    "maj6": "6", "min6": "m6",
    "add9": "add9", "madd9": "madd9",
    "7sus4": "7sus4",
}

# quality token -> canonical interval set
QUALITIES: dict[str, tuple[str, tuple[int, ...]]] = {
    "maj": ("maj", (0, 4, 7)),
    "major": ("maj", (0, 4, 7)),
    "m": ("min", (0, 3, 7)),
    "min": ("min", (0, 3, 7)),
    "minor": ("min", (0, 3, 7)),
    "dim": ("dim", (0, 3, 6)),
    "o": ("dim", (0, 3, 6)),
    "aug": ("aug", (0, 4, 8)),
    "+": ("aug", (0, 4, 8)),
    "sus2": ("sus2", (0, 2, 7)),
    "sus4": ("sus4", (0, 5, 7)),
    "sus": ("sus4", (0, 5, 7)),
    "maj7": ("maj7", (0, 4, 7, 11)),
    "M7": ("maj7", (0, 4, 7, 11)),
    "Δ7": ("maj7", (0, 4, 7, 11)),
    "Δ": ("maj7", (0, 4, 7, 11)),
    "m7": ("min7", (0, 3, 7, 10)),
    "min7": ("min7", (0, 3, 7, 10)),
    "7": ("dom7", (0, 4, 7, 10)),
    "m7b5": ("m7b5", (0, 3, 6, 10)),
    "ø": ("m7b5", (0, 3, 6, 10)),
    "dim7": ("dim7", (0, 3, 6, 9)),
    "o7": ("dim7", (0, 3, 6, 9)),
    "maj9": ("maj9", (0, 4, 7, 11, 14)),
    "M9": ("maj9", (0, 4, 7, 11, 14)),
    "m9": ("min9", (0, 3, 7, 10, 14)),
    "9": ("dom9", (0, 4, 7, 10, 14)),
    "maj11": ("maj11", (0, 4, 7, 11, 14, 17)),
    "m11": ("min11", (0, 3, 7, 10, 14, 17)),
    "11": ("dom11", (0, 4, 7, 10, 14, 17)),
    "13": ("dom13", (0, 4, 7, 10, 14, 21)),
    "m13": ("min13", (0, 3, 7, 10, 14, 21)),
    "6": ("maj6", (0, 4, 7, 9)),
    "m6": ("min6", (0, 3, 7, 9)),
    "add9": ("add9", (0, 4, 7, 14)),
    "madd9": ("madd9", (0, 3, 7, 14)),
    "7sus4": ("7sus4", (0, 5, 7, 10)),
}

_QUALITY_KEYS = sorted((k for k in QUALITIES if k), key=len, reverse=True)
_QUALITY_RE = "|".join(re.escape(k) for k in _QUALITY_KEYS)

# root + accidental + optional quality + optional /bass
_CHORD_RE = re.compile(
    rf"(?<![A-Za-z0-9])([A-Ga-g])([#b]?)({_QUALITY_RE})?(?:/([A-Ga-g])([#b]?))?(?![A-Za-z0-9])"
)
_CONTEXT_RE = re.compile(r"\b(chords?|progression|progressions|harmony|strum)\b", re.I)
_SEP_RE = re.compile(r"\s*(?:[-–—,|/]|\bthen\b|\bto\b|\binto\b)\s*", re.I)


@dataclass(frozen=True)
class Chord:
    root: int          # pitch class 0..11
    quality: str       # canonical, e.g. "min7"
    bass: int | None   # slash-chord bass pitch class, or None
    name: str          # normalized display name, e.g. "Am7", "C/G"

    @property
    def intervals(self) -> tuple[int, ...]:
        return QUALITIES[self.quality][1]


def _root_pc(letter: str, accidental: str = "") -> int | None:
    return NOTE_PC.get(letter.upper() + accidental)


def parse_chord_token(token: str) -> Chord | None:
    token = token.strip()
    if not token:
        return None
    m = _CHORD_RE.fullmatch(token)
    if not m:
        return None
    letter, accidental, quality, bass_letter, bass_acc = m.groups()
    root = _root_pc(letter, accidental)
    if root is None:
        return None
    q = quality or "maj"
    if q not in QUALITIES:
        return None
    canonical, _ = QUALITIES[q]
    bass = _root_pc(bass_letter, bass_acc) if bass_letter else None
    if bass_letter and bass is None:
        return None
    name = PC_NAME[root] + DISPLAY.get(canonical, canonical)
    if bass is not None:
        name += "/" + PC_NAME[bass]
    return Chord(root=root, quality=canonical, bass=bass, name=name)


def _tokens(text: str) -> list[tuple[str, bool]]:
    """Return (token, has_quality) for every chord-shaped token."""
    out = []
    for m in _CHORD_RE.finditer(text):
        letter, accidental, quality, bass_letter, bass_acc = m.groups()
        if _root_pc(letter, accidental) is None:
            continue
        if letter.islower() and not quality:
            # a stray lowercase "a" is almost never a chord
            continue
        out.append((m.group(0), bool(quality)))
    return out


def parse_progression(text: str) -> list[Chord] | None:
    """Find a chord progression in a prompt, or None.

    Accepts:
      "add guitar chords: Am F C G"
      "play Am-F-C-G on guitar"
      "strum Cmaj7 then Am7 then Fmaj7 then G7"
      "use a C/G"
    """
    if not text:
        return None
    context = bool(_CONTEXT_RE.search(text))
    tokens = _tokens(text)
    if not tokens:
        return None

    has_separator = bool(_SEP_RE.search(text))
    has_quality = any(q for _, q in tokens)
    if not context and not has_quality and not has_separator:
        return None
    if not context and len(tokens) < 2:
        return None

    chords: list[Chord] = []
    for token, _ in tokens:
        chord = parse_chord_token(token)
        if chord is not None:
            chords.append(chord)
    # drop duplicates that come from overlapping matches
    deduped: list[Chord] = []
    for chord in chords:
        if not deduped or deduped[-1] != chord:
            deduped.append(chord)
    return deduped or None


def chord_midi(chord: Chord, octave: int = 3) -> list[int]:
    """MIDI notes for a chord. C3 = 48; guitar range."""
    root_midi = 12 * (octave + 1) + chord.root
    notes = [root_midi + i for i in chord.intervals]
    if chord.bass is not None:
        notes = [12 * octave + chord.bass] + notes
    return [max(21, min(108, n)) for n in notes]


def _pluck(midi: int, sr: int, duration: float) -> "object":
    import numpy as np

    f = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    n = max(32, int(duration * sr))
    t = np.arange(n) / sr
    env = np.exp(-t * 4.2)
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, 7):
        sig += (0.6 ** (k - 1)) * np.sin(2 * np.pi * f * k * t)
    return (0.16 * sig * env).astype(np.float32)


def render_chord_progression(
    chords: list[Chord],
    tempo: float,
    beats: list[float],
    dur: float,
    sr: int,
    instrument: str = "guitar",
) -> tuple["object", int]:
    """One chord per bar, strummed. Returns (mono layer, chord count)."""
    import numpy as np

    n = int(dur * sr)
    out = np.zeros(n, dtype=np.float32)
    if not chords:
        return out, 0

    beat_sec = 60.0 / max(float(tempo), 40.0)
    bar_sec = beat_sec * 4.0
    start = float(beats[0]) if beats else 0.0
    family = "guitar"
    try:
        from server.ml.synthesis import timbre

        if timbre.available():
            family = timbre._nearest_family(instrument) or instrument
    except Exception:
        pass

    idx = 0
    t0 = start
    while t0 < dur:
        chord = chords[idx % len(chords)]
        notes = chord_midi(chord, octave=3)
        for j, midi in enumerate(notes):
            when = t0 + 0.012 * j
            if when >= dur:
                break
            note = None
            try:
                from server.ml.synthesis import timbre

                if timbre.available():
                    note, _ = timbre.render_note(
                        family, midi,
                        duration=min(max(bar_sec * 0.95, 0.4), 2.2),
                        velocity=96, sr=sr,
                        seed=(hash((midi, idx)) & 0x7FFFFFFF),
                    )
            except Exception:
                note = None
            if note is None:
                note = _pluck(midi, sr, min(max(bar_sec * 0.95, 0.4), 2.2))
            i = int(when * sr)
            seg = min(len(note), n - i)
            if seg > 0:
                out[i:i + seg] += note[:seg]
        idx += 1
        t0 += bar_sec

    return out, idx
