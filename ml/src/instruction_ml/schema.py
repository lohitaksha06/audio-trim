"""Canonical label schema for the instruction-understanding task.

Keep this file dependency-free. Everything else (data, rules, models, tests)
imports the vocabulary from here, so there is exactly one definition of what an
operation, target, or amount unit is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

OPERATIONS: dict[str, str] = {
    "remove_stem": "Mute or delete a stem / instrument",
    "isolate_stem": "Keep only one stem / instrument",
    "gain": "Change level of the mix or a stem",
    "fade": "Fade a region in or out",
    "cut_segment": "Delete a time region",
    "trim_segment": "Keep only a time region",
    "remove_speaker": "Drop a speaker (or keep only another one)",
    "denoise": "Remove noise, hiss, hum, room tone",
    "normalize": "Even out loudness",
    "speed": "Change tempo / playback speed",
    "add_effect": "Add reverb/delay/distortion or a mood treatment",
    "remove_silence": "Cut silence, pauses, or filler gaps",
}

OPERATION_KEYS: tuple[str, ...] = tuple(OPERATIONS)

# Op -> allowed amount units. An empty tuple means the operation has no amount.
UNITS_BY_OPERATION: dict[str, tuple[str, ...]] = {
    "remove_stem": (),
    "isolate_stem": (),
    "gain": ("dB",),
    "fade": ("seconds",),
    "cut_segment": ("seconds",),
    "trim_segment": ("seconds",),
    "remove_speaker": (),
    "denoise": (),
    "normalize": (),
    "speed": ("factor", "bpm"),
    "add_effect": (),
    "remove_silence": ("seconds",),
}

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

TARGETS: dict[str, str] = {
    "full_mix": "The whole track / no specific stem",
    "vocals": "Lead or background vocals / speech",
    "drums": "Drum kit, kick, snare, hats, cymbals",
    "bass": "Bass guitar or sub bass",
    "guitar": "Guitar (acoustic or electric)",
    "keys": "Piano, keys, organ, synths",
    "other": "FX, pads, strings, anything else",
    "speaker_1": "First speaker / host",
    "speaker_2": "Second speaker / guest",
}

TARGET_KEYS: tuple[str, ...] = tuple(TARGETS)

UNITS: tuple[str, ...] = ("dB", "seconds", "factor", "bpm")

# Operations that default to full_mix when the prompt names no stem.
DEFAULT_FULL_MIX_OPS = {
    "cut_segment",
    "trim_segment",
    "normalize",
    "speed",
    "denoise",
    "remove_silence",
    "add_effect",
}

# Operations where an amount is meaningful enough to regress.
AMOUNT_OPS = {"gain", "fade", "cut_segment", "trim_segment", "speed", "remove_silence"}


# ---------------------------------------------------------------------------
# Row schema
# ---------------------------------------------------------------------------


@dataclass
class Label:
    """One labeled instruction."""

    id: str
    text: str
    ops: list[str]
    targets: list[str] = field(default_factory=list)
    amount: float | None = None
    unit: str | None = None
    span: list[float | None] | None = None
    notes: str = ""
    source: str = "seed"
    group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "ops": list(self.ops),
            "targets": list(self.targets),
            "amount": self.amount,
            "unit": self.unit,
            "span": self.span,
            "notes": self.notes,
            "source": self.source,
        }
        if self.group is not None:
            row["group"] = self.group
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Label":
        return cls(
            id=str(row["id"]),
            text=str(row["text"]),
            ops=[str(x) for x in row.get("ops", [])],
            targets=[str(x) for x in row.get("targets", [])],
            amount=None if row.get("amount") is None else float(row["amount"]),
            unit=row.get("unit"),
            span=row.get("span"),
            notes=str(row.get("notes", "")),
            source=str(row.get("source", "seed")),
            group=row.get("group"),
        )


class SchemaError(ValueError):
    """Raised when a labeled row violates the schema."""


def validate_label(row: Label) -> list[str]:
    """Return a list of problems. Empty list means valid."""
    problems: list[str] = []
    text = " ".join(row.text.split())
    if not text:
        problems.append("empty text")
    if not row.id:
        problems.append("missing id")

    if not row.ops:
        problems.append("no operations")
    for op in row.ops:
        if op not in OPERATIONS:
            problems.append(f"unknown operation: {op!r}")
    if len(set(row.ops)) != len(row.ops):
        problems.append("duplicate operation")
    if not row.targets:
        problems.append("no targets")
    for target in row.targets:
        if target not in TARGETS:
            problems.append(f"unknown target: {target!r}")
    if len(set(row.targets)) != len(row.targets):
        problems.append("duplicate target")

    if row.amount is not None:
        if row.unit not in UNITS:
            problems.append(f"amount without a valid unit (got {row.unit!r})")
        allowed: set[str] = set()
        for op in row.ops:
            allowed.update(UNITS_BY_OPERATION.get(op, ()))
        if row.unit not in allowed:
            problems.append(
                f"unit {row.unit!r} not allowed for ops {row.ops!r} "
                f"(allowed: {sorted(allowed) or 'none'})"
            )
        if row.unit == "factor" and row.amount <= 0:
            problems.append("speed factor must be positive")
        if row.unit == "bpm" and row.amount <= 0:
            problems.append("bpm must be positive")
        if row.unit == "dB" and abs(row.amount) > 24:
            problems.append("gain beyond +/-24 dB is out of range")
    elif row.unit is not None:
        problems.append("unit set without amount")

    if row.span is not None:
        if not isinstance(row.span, (list, tuple)) or len(row.span) != 2:
            problems.append("span must be [start, end] or null")
        else:
            start, end = row.span
            if start is not None and start < 0:
                problems.append("span start must be >= 0")
            if end is not None and end < 0:
                problems.append("span end must be >= 0")
            if start is not None and end is not None and end <= start:
                problems.append("span end must be greater than start")

    # Amount regression is only well-defined for a single operation.
    if row.amount is not None and len(row.ops) > 1 and "amount on multi-op" not in row.notes:
        problems.append("amount is only allowed on single-operation rows")

    return problems


def primary_op(row: Label) -> str:
    """First operation; used for stratified splitting and the confusion matrix."""
    return row.ops[0] if row.ops else "unknown"


def op_set(rows: Iterable[Label]) -> set[str]:
    return {op for row in rows for op in row.ops}
