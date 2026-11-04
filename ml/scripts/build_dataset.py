"""Build the dataset: validate seeds, expand paraphrases, split by group.

Usage:
    python scripts/build_dataset.py [--variants 4] [--seed 13]
"""

from __future__ import annotations

import argparse
import json

import _common
from instruction_ml.augment import expand
from instruction_ml.data import (
    assert_no_group_leakage,
    dataset_stats,
    grouped_split,
    load_jsonl,
    split_stats,
    validate_rows,
    write_jsonl,
)

SEED_PATH = _common.DATA / "seed_prompts.jsonl"
DATASET_PATH = _common.DATA / "dataset.jsonl"
STATS_PATH = _common.DATA / "dataset_stats.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=int, default=4, help="paraphrases per seed row")
    parser.add_argument("--seed", type=int, default=13, help="split RNG seed")
    args = parser.parse_args()

    seeds = load_jsonl(SEED_PATH)
    problems = validate_rows(seeds)
    if problems:
        print("Seed dataset is invalid:")
        for problem in problems:
            print("  -", problem)
        return 1
    print(f"Seed rows: {len(seeds)}")

    rows = expand(seeds, variants=args.variants)
    problems = validate_rows(rows)
    if problems:
        print("Expanded dataset is invalid:")
        for problem in problems[:40]:
            print("  -", problem)
        print(f"... {len(problems)} problem(s) total")
        return 1

    hand_path = _common.DATA / "heldout_prompts.jsonl"
    heldout = load_jsonl(hand_path)
    heldout_problems = validate_rows(heldout)
    if heldout_problems:
        print("Held-out set is invalid:")
        for problem in heldout_problems:
            print("  -", problem)
        return 1
    overlap = {
        " ".join(row.text.lower().split()) for row in rows
    } & {" ".join(row.text.lower().split()) for row in heldout}
    if overlap:
        print(f"Held-out set overlaps the training data: {sorted(overlap)[:5]}")
        return 1
    print(f"Held-out rows: {len(heldout)} (never used for training)")

    write_jsonl(DATASET_PATH, rows)
    print(f"Dataset rows: {len(rows)} (seed + paraphrases)")

    splits = grouped_split(rows, seed=args.seed)
    assert_no_group_leakage(splits)
    for name, split_rows in splits.items():
        write_jsonl(_common.SPLITS / f"{name}.jsonl", split_rows)

    stats = {
        "dataset": dataset_stats(rows),
        "splits": split_stats(splits),
        "heldout": dataset_stats(heldout),
        "seed": args.seed,
        "variants_per_seed": args.variants,
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print("\nSplit sizes (rows / handwritten):")
    for name, split_rows in splits.items():
        handwritten = sum(1 for row in split_rows if row.source == "seed")
        print(f"  {name:5s} {len(split_rows):4d} / {handwritten:3d}")

    print("\nPrimary operation coverage by split:")
    ops = sorted(
        {op for split_rows in splits.values() for row in split_rows for op in row.ops}
    )
    for op in ops:
        counts = {
            name: sum(1 for row in split_rows if op in row.ops)
            for name, split_rows in splits.items()
        }
        print(
            f"  {op:16s} train={counts['train']:3d} "
            f"val={counts['val']:3d} test={counts['test']:3d}"
        )

    print(f"\nWrote {DATASET_PATH.name}, data/splits/*.jsonl, {STATS_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
