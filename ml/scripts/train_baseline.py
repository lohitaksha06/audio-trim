"""Train the TF-IDF + logistic-regression baseline.

Usage:
    python scripts/train_baseline.py [--artifact models/baseline.joblib]
"""

from __future__ import annotations

import argparse
import json

import _common
from instruction_ml.data import load_jsonl
from instruction_ml.metrics import comparison_table, error_examples, evaluate_by_source
from instruction_ml.model import DEFAULT_ARTIFACT, BaselineModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--train", default=str(_common.SPLITS / "train.jsonl"))
    parser.add_argument("--val", default=str(_common.SPLITS / "val.jsonl"))
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    print(f"Training on {len(train_rows)} rows, validating on {len(val_rows)} rows")

    model = BaselineModel().fit(train_rows)
    model.tune_thresholds(val_rows)
    model.save(args.artifact)
    print(f"Saved artifact: {args.artifact}")
    print(f"Amount regressors: {sorted(model.amount_models)}")
    print(f"Tuned operation thresholds: {model.ops_thresholds}")

    val_preds = [model.predict(row.text) for row in val_rows]
    results = evaluate_by_source(val_rows, val_preds, "tfidf-logreg")
    handwritten = results.get("handwritten", results["all"])
    print("\nValidation:")
    print(comparison_table({"validation": results["all"]}))
    print(
        f"Handwritten-only op micro F1: {handwritten['operations']['micro_f1']:.3f}, "
        f"target micro F1: {handwritten['targets']['micro_f1']:.3f}"
    )

    _common.REPORTS.mkdir(parents=True, exist_ok=True)
    (_common.REPORTS / "baseline_val.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    errors = error_examples(val_rows, val_preds, limit=20)
    (_common.REPORTS / "baseline_val_errors.json").write_text(
        json.dumps(errors, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote reports/baseline_val.json and reports/baseline_val_errors.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
