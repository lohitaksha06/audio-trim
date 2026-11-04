"""Evaluate the rule baseline, TF-IDF model, and optionally the transformer.

Usage:
    python scripts/evaluate.py                 # test split
    python scripts/evaluate.py --split val
    python scripts/evaluate.py --transformer-preds models/transformer/test_predictions.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _common
from instruction_ml.data import load_jsonl
from instruction_ml.metrics import (
    comparison_table,
    error_examples,
    evaluate_by_source,
)
from instruction_ml.model import DEFAULT_ARTIFACT, BaselineModel
from instruction_ml.rules import rule_label


def _rule_preds(rows) -> list[dict]:
    preds = []
    for row in rows:
        label = rule_label(row.text)
        preds.append(
            {
                "ops": label.ops,
                "targets": label.targets,
                "amount": label.amount,
                "unit": label.unit,
                "amounts": (
                    {label.ops[0]: {"amount": label.amount, "unit": label.unit}}
                    if label.amount is not None and len(label.ops) == 1
                    else {}
                ),
                "span": label.span,
            }
        )
    return preds


def _load_transformer_preds(path: Path, rows) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in payload}
    missing = [row.id for row in rows if row.id not in by_id]
    if missing:
        raise SystemExit(
            f"{path} is missing {len(missing)} row(s), e.g. {missing[:3]}. "
            "Re-run the transformer evaluation on this split."
        )
    return [by_id[row.id] for row in rows]


def _per_operation_table(result: dict) -> str:
    per_class = result["operations"]["per_class"]
    lines = ["| Operation | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for name, scores in per_class.items():
        lines.append(
            f"| {name} | {scores['precision']:.3f} | {scores['recall']:.3f} | "
            f"{scores['f1']:.3f} | {scores['support']} |"
        )
    return "\n".join(lines)


def _per_target_table(result: dict) -> str:
    per_class = result["targets"]["per_class"]
    lines = ["| Target | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for name, scores in per_class.items():
        lines.append(
            f"| {name} | {scores['precision']:.3f} | {scores['recall']:.3f} | "
            f"{scores['f1']:.3f} | {scores['support']} |"
        )
    return "\n".join(lines)


def _confusion_table(result: dict) -> str:
    confusion = result["confusion"]
    labels = confusion["labels"]
    if not labels:
        return "_No single-operation rows._"
    header = "| actual \\ predicted | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join("---:" for _ in labels) + "|"
    lines = [header, sep]
    for actual in labels:
        row = confusion["matrix"].get(actual, {})
        cells = [str(row.get(label, 0)) for label in labels]
        lines.append(f"| {actual} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_errors(name: str, rows, preds) -> Path:
    path = _common.REPORTS / f"errors_{name}.csv"
    errors = error_examples(rows, preds, limit=200)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "text",
                "source",
                "gold_ops",
                "pred_ops",
                "gold_targets",
                "pred_targets",
                "gold_amount",
                "pred_amount",
            ],
        )
        writer.writeheader()
        for item in errors:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in item.items()
                    if key != "top_scores"
                }
            )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "heldout"],
        default="test",
        help="heldout = independent hand-written prompts, never used for training",
    )
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--transformer-preds", default=None)
    args = parser.parse_args()

    if args.split == "heldout":
        rows = load_jsonl(_common.DATA / "heldout_prompts.jsonl")
    else:
        rows = load_jsonl(_common.SPLITS / f"{args.split}.jsonl")
    if not rows:
        raise SystemExit(f"No rows in {args.split} split")

    model = BaselineModel.load(args.artifact)

    predictions: dict[str, list[dict]] = {
        "rules": _rule_preds(rows),
        "tfidf-logreg": [model.predict(row.text) for row in rows],
    }
    if args.transformer_preds:
        predictions["transformer"] = _load_transformer_preds(
            Path(args.transformer_preds), rows
        )

    results = {
        name: evaluate_by_source(rows, preds, name) for name, preds in predictions.items()
    }
    all_results = {name: result["all"] for name, result in results.items()}

    print(f"\nEvaluation split: {args.split} ({len(rows)} rows)")
    print(comparison_table(all_results))
    if "handwritten" in results["rules"]:
        print("\nHandwritten rows only:")
        print(
            comparison_table(
                {name: result["handwritten"] for name, result in results.items()}
            )
        )

    _common.REPORTS.mkdir(parents=True, exist_ok=True)
    (_common.REPORTS / f"evaluation_{args.split}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    error_paths = []
    for name, preds in predictions.items():
        error_paths.append(_write_errors(f"{name}_{args.split}", rows, preds))

    best_model = "tfidf-logreg"
    best = all_results[best_model]
    handwritten = results[best_model].get("handwritten", best)
    report = [
        f"# Evaluation — `{args.split}` split",
        "",
        f"Rows: **{len(rows)}**",
        "",
        "## Comparison (all rows)",
        "",
        comparison_table(all_results),
        "",
        "## Handwritten rows only (honest headline)",
        "",
        comparison_table(
            {name: result["handwritten"] for name, result in results.items()}
            if "handwritten" in results["rules"]
            else all_results
        ),
        "",
        f"## Per-operation scores — {best_model}",
        "",
        _per_operation_table(best),
        "",
        f"## Per-target scores — {best_model}",
        "",
        _per_target_table(best),
        "",
        f"## Confusion matrix — {best_model} (single-operation rows)",
        "",
        _confusion_table(best),
        "",
        "## Amount errors (dB / seconds / factor / bpm)",
        "",
        "```json",
        json.dumps(best["amount"], indent=2),
        "```",
        "",
        "## Top confusions",
        "",
        "```json",
        json.dumps(best["confusion"]["top_confusions"], indent=2),
        "```",
        "",
        f"_Handwritten subset: op micro F1 {handwritten['operations']['micro_f1']:.3f}, "
        f"target micro F1 {handwritten['targets']['micro_f1']:.3f}._",
        "",
        "## Worst errors",
        "",
    ]
    for item in error_examples(rows, predictions[best_model], limit=15):
        report.append(
            f"- `{item['text']}` — gold {item['gold_ops']} / {item['gold_targets']} "
            f"→ predicted {item['pred_ops']} / {item['pred_targets']} "
            f"(scores {item['top_scores']})"
        )
    report.append("")
    report.append("Error CSVs: " + ", ".join(f"`{p.name}`" for p in error_paths))
    report_path = _common.REPORTS / f"report_{args.split}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote {report_path} and {len(error_paths)} error CSV(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
