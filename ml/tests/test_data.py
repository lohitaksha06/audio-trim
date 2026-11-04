from pathlib import Path

from instruction_ml.augment import expand
from instruction_ml.data import (
    assert_no_group_leakage,
    grouped_split,
    load_jsonl,
    validate_rows,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "seed_prompts.jsonl"


def test_expansion_preserves_labels():
    seeds = load_jsonl(SEEDS)[:40]
    rows = expand(seeds, variants=3)
    assert validate_rows(rows) == []
    by_id = {row.id: row for row in rows if row.source == "seed"}
    for seed in seeds:
        original = by_id[seed.id]
        assert original.ops == seed.ops
        assert original.targets == seed.targets
        assert original.amount == seed.amount
    for row in rows:
        if row.source == "template":
            parent = next(seed for seed in seeds if row.id.startswith(seed.id + "-v"))
            assert row.ops == parent.ops
            assert row.targets == parent.targets
            assert row.amount == parent.amount


def test_split_has_no_group_leakage():
    seeds = load_jsonl(SEEDS)
    rows = expand(seeds, variants=2)
    splits = grouped_split(rows, seed=7)
    assert_no_group_leakage(splits)
    assert splits["train"] and splits["test"]


def test_split_is_reproducible():
    seeds = load_jsonl(SEEDS)
    rows = expand(seeds, variants=2)
    first = grouped_split(rows, seed=7)
    second = grouped_split(rows, seed=7)
    for name in first:
        assert [row.id for row in first[name]] == [row.id for row in second[name]]


def test_heldout_is_valid_and_disjoint():
    seeds = load_jsonl(SEEDS)
    rows = expand(seeds, variants=2)
    heldout = load_jsonl(ROOT / "data" / "heldout_prompts.jsonl")
    assert validate_rows(heldout) == []
    training_text = {" ".join(row.text.lower().split()) for row in rows}
    for row in heldout:
        assert " ".join(row.text.lower().split()) not in training_text, row.id
