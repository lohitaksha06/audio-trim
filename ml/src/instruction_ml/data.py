"""Dataset loading, validation, grouped splitting, and statistics."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .schema import Label, primary_op, validate_label

ML_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ML_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"


def load_jsonl(path: str | Path) -> list[Label]:
    path = Path(path)
    rows: list[Label] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(Label.from_dict(payload))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Label]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def validate_rows(rows: Iterable[Label]) -> list[str]:
    """Return a list of human-readable problems across all rows."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_text: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], float | None]] = {}
    for row in rows:
        for problem in validate_label(row):
            problems.append(f"{row.id}: {problem}")
        if row.id in seen_ids:
            problems.append(f"{row.id}: duplicate id")
        seen_ids.add(row.id)
        norm = " ".join(row.text.lower().split())
        signature = (tuple(sorted(row.ops)), tuple(sorted(row.targets)), row.amount)
        if norm in seen_text:
            other_id, other_ops, other_targets, other_amount = seen_text[norm]
            # The same wording is fine when the labels agree; conflicting labels
            # for identical text are a labeling bug.
            if signature != (other_ops, other_targets, other_amount):
                problems.append(
                    f"{row.id}: duplicate text with conflicting labels (also {other_id})"
                )
        else:
            seen_text[norm] = (row.id, signature[0], signature[1], signature[2])
    return problems


def grouped_split(
    rows: list[Label],
    seed: int = 13,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
) -> dict[str, list[Label]]:
    """Split by `group` so paraphrases never cross splits.

    Rows without a group form their own group (their id).
    Stratified by primary operation with at least one group in val/test when
    the operation has enough groups.
    """
    if abs((train + val + test) - 1.0) > 1e-9:
        raise ValueError("train + val + test must equal 1.0")

    by_group: dict[str, list[Label]] = defaultdict(list)
    for row in rows:
        by_group[row.group or row.id].append(row)

    group_op: dict[str, str] = {}
    for group, members in by_group.items():
        group_op[group] = primary_op(members[0])

    by_op: dict[str, list[str]] = defaultdict(list)
    for group, op in group_op.items():
        by_op[op].append(group)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for op, groups in sorted(by_op.items()):
        rng.shuffle(groups)
        n = len(groups)
        if n == 1:
            assignment[groups[0]] = "train"
            continue
        if n == 2:
            assignment[groups[0]] = "train"
            assignment[groups[1]] = "test"
            continue
        n_test = max(1, round(n * test))
        n_val = max(1, round(n * val)) if n >= 5 else 0
        # never take so much that train is empty
        while n_test + n_val >= n:
            if n_val > 0:
                n_val -= 1
            else:
                n_test -= 1
        for i, group in enumerate(groups):
            if i < n - n_test - n_val:
                assignment[group] = "train"
            elif i < n - n_test:
                assignment[group] = "val"
            else:
                assignment[group] = "test"

    splits: dict[str, list[Label]] = {"train": [], "val": [], "test": []}
    for group, members in by_group.items():
        splits[assignment[group]].extend(members)
    for name in splits:
        splits[name].sort(key=lambda r: r.id)
    return splits


def dataset_stats(rows: list[Label]) -> dict:
    ops = Counter(op for row in rows for op in row.ops)
    targets = Counter(target for row in rows for target in row.targets)
    units = Counter(row.unit for row in rows if row.unit)
    sources = Counter(row.source for row in rows)
    return {
        "rows": len(rows),
        "groups": len({row.group or row.id for row in rows}),
        "multi_operation_rows": sum(1 for row in rows if len(row.ops) > 1),
        "rows_with_amount": sum(1 for row in rows if row.amount is not None),
        "rows_with_span": sum(1 for row in rows if row.span is not None),
        "operations": dict(sorted(ops.items())),
        "targets": dict(sorted(targets.items())),
        "units": dict(sorted(units.items())),
        "sources": dict(sorted(sources.items())),
    }


def split_stats(splits: dict[str, list[Label]]) -> dict:
    stats: dict[str, dict] = {}
    for name, rows in splits.items():
        stats[name] = dataset_stats(rows)
        stats[name]["handwritten"] = sum(1 for row in rows if row.source == "seed")
    return stats


def assert_no_group_leakage(splits: dict[str, list[Label]]) -> None:
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for row in rows:
            group = row.group or row.id
            if group in seen and seen[group] != name:
                raise AssertionError(
                    f"group {group!r} appears in both {seen[group]} and {name}"
                )
            seen[group] = name
