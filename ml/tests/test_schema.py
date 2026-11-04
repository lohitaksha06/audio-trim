from pathlib import Path

import pytest

from instruction_ml.data import load_jsonl, validate_rows
from instruction_ml.schema import OPERATIONS, OPERATION_KEYS, TARGET_KEYS, validate_label

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "data" / "seed_prompts.jsonl"


@pytest.fixture(scope="module")
def seeds():
    return load_jsonl(SEEDS)


def test_seed_file_is_valid(seeds):
    assert validate_rows(seeds) == []


def test_seed_size_and_coverage(seeds):
    assert len(seeds) >= 100
    covered = {op for row in seeds for op in row.ops}
    assert covered == set(OPERATION_KEYS), f"missing operations: {set(OPERATION_KEYS) - covered}"


def test_every_seed_has_a_target(seeds):
    for row in seeds:
        assert row.targets, row.id
        assert set(row.targets) <= set(TARGET_KEYS), row.id


def test_multi_operation_rows_have_no_amount(seeds):
    for row in seeds:
        if len(row.ops) > 1:
            assert row.amount is None, f"{row.id} attaches one amount to several ops"


def test_cut_and_trim_are_both_present(seeds):
    ops = {op for row in seeds for op in row.ops}
    assert "cut_segment" in ops
    assert "trim_segment" in ops


def test_validator_rejects_bad_operation():
    from instruction_ml.schema import Label

    row = Label(
        id="bad-1",
        text="do something",
        ops=["explode"],
        targets=["vocals"],
    )
    assert validate_label(row)


def test_validator_rejects_unit_mismatch():
    from instruction_ml.schema import Label

    row = Label(
        id="bad-2",
        text="remove the vocals by 3 db",
        ops=["remove_stem"],
        targets=["vocals"],
        amount=3.0,
        unit="dB",
    )
    assert validate_label(row)


def test_operations_documented():
    assert len(OPERATIONS) == 12
