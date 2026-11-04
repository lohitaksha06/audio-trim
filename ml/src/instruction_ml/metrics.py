"""Evaluation metrics shared by the rule baseline, TF-IDF model, and transformer."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .schema import OPERATION_KEYS, TARGET_KEYS, Label, primary_op

OPS = list(OPERATION_KEYS)
TARGETS = list(TARGET_KEYS)


def _binary_matrix(rows: list[Label], key: str, classes: list[str]) -> np.ndarray:
    return np.array([[1 if name in getattr(row, key) else 0 for name in classes] for row in rows])


def _pred_matrix(preds: list[dict], key: str, classes: list[str]) -> np.ndarray:
    return np.array([[1 if name in pred.get(key, []) else 0 for name in classes] for pred in preds])


def multilabel_scores(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    subset = float(np.mean(np.all(y_true == y_pred, axis=1))) if len(y_true) else 0.0
    hamming = float(np.mean(y_true != y_pred)) if y_true.size else 0.0
    precisions, recalls, f1s, supports = per_class
    return {
        "subset_accuracy": round(subset, 4),
        "micro_precision": round(float(micro[0]), 4),
        "micro_recall": round(float(micro[1]), 4),
        "micro_f1": round(float(micro[2]), 4),
        "macro_f1": round(float(macro[2]), 4),
        "hamming_loss": round(hamming, 4),
        "per_class": {
            name: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for name, p, r, f, s in zip(classes, precisions, recalls, f1s, supports)
        },
    }


def amount_errors(gold: list[Label], preds: list[dict]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    overall: list[float] = []
    for row, pred in zip(gold, preds):
        if row.amount is None or len(row.ops) != 1:
            continue
        op = row.ops[0]
        if op not in pred.get("ops", []):
            continue
        # Prefer the per-operation amount; fall back to the flat field.
        predicted = pred.get("amounts", {}).get(op, {})
        value = predicted.get("amount")
        unit = predicted.get("unit", row.unit)
        if value is None and pred.get("unit") == row.unit:
            value = pred.get("amount")
        if value is None or unit != row.unit:
            continue
        error = abs(float(value) - float(row.amount))
        groups[f"{op}|{row.unit}"].append(error)
        overall.append(error)

    def summarize(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "mae": None, "median_ae": None}
        return {
            "n": len(values),
            "mae": round(float(np.mean(values)), 3),
            "median_ae": round(float(np.median(values)), 3),
        }

    return {
        "overall": summarize(overall),
        "by_operation_unit": {key: summarize(values) for key, values in sorted(groups.items())},
    }


def confusion(gold: list[Label], preds: list[dict]) -> dict:
    """Confusion matrix over single-operation rows (predicted primary op)."""
    matrix: dict[str, Counter] = defaultdict(Counter)
    labels: list[str] = []
    for row, pred in zip(gold, preds):
        if len(row.ops) != 1:
            continue
        actual = row.ops[0]
        predicted = pred.get("ops", [])
        guess = predicted[0] if predicted else "none"
        if actual not in labels:
            labels.append(actual)
        if guess not in labels:
            labels.append(guess)
        matrix[actual][guess] += 1
    return {
        "labels": labels,
        "matrix": {actual: dict(counter) for actual, counter in matrix.items()},
        "top_confusions": [
            {"actual": actual, "predicted": predicted, "count": count}
            for actual, counter in matrix.items()
            for predicted, count in counter.items()
            if actual != predicted
        ][:12],
    }


def evaluate(gold: list[Label], preds: list[dict], name: str = "model") -> dict:
    if len(gold) != len(preds):
        raise ValueError(f"{name}: gold and predictions have different lengths")
    op_true = _binary_matrix(gold, "ops", OPS)
    op_pred = _pred_matrix(preds, "ops", OPS)
    target_true = _binary_matrix(gold, "targets", TARGETS)
    target_pred = _pred_matrix(preds, "targets", TARGETS)

    exact = sum(
        1
        for row, pred in zip(gold, preds)
        if set(row.ops) == set(pred.get("ops", []))
        and set(row.targets) == set(pred.get("targets", []))
    )
    return {
        "name": name,
        "n": len(gold),
        "operations": multilabel_scores(op_true, op_pred, OPS),
        "targets": multilabel_scores(target_true, target_pred, TARGETS),
        "action_exact_match": round(exact / len(gold), 4) if gold else 0.0,
        "amount": amount_errors(gold, preds),
        "confusion": confusion(gold, preds),
    }


def evaluate_by_source(gold: list[Label], preds: list[dict], name: str = "model") -> dict:
    """Overall plus the handwritten subset (the honest headline number)."""
    result = {"all": evaluate(gold, preds, name)}
    seed_indices = [i for i, row in enumerate(gold) if row.source == "seed"]
    if seed_indices:
        result["handwritten"] = evaluate(
            [gold[i] for i in seed_indices],
            [preds[i] for i in seed_indices],
            f"{name}-handwritten",
        )
    return result


def comparison_table(results: dict[str, dict]) -> str:
    header = (
        "| Model | n | Op subset acc | Op micro F1 | Op macro F1 | "
        "Target micro F1 | Target macro F1 | Action exact | Amount MAE |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for name, result in results.items():
        ops = result["operations"]
        targets = result["targets"]
        amount = result["amount"]["overall"]["mae"]
        lines.append(
            f"| {name} | {result['n']} | {ops['subset_accuracy']:.3f} | "
            f"{ops['micro_f1']:.3f} | {ops['macro_f1']:.3f} | "
            f"{targets['micro_f1']:.3f} | {targets['macro_f1']:.3f} | "
            f"{result['action_exact_match']:.3f} | "
            f"{amount if amount is not None else '-'} |"
        )
    return header + "\n".join(lines)


def error_examples(gold: list[Label], preds: list[dict], limit: int = 25) -> list[dict]:
    rows: list[dict] = []
    for row, pred in zip(gold, preds):
        if set(row.ops) == set(pred.get("ops", [])) and set(row.targets) == set(
            pred.get("targets", [])
        ):
            continue
        rows.append(
            {
                "id": row.id,
                "text": row.text,
                "source": row.source,
                "gold_ops": row.ops,
                "pred_ops": pred.get("ops", []),
                "gold_targets": row.targets,
                "pred_targets": pred.get("targets", []),
                "gold_amount": row.amount,
                "pred_amount": pred.get("amount"),
                "top_scores": {
                    key: round(value, 3)
                    for key, value in sorted(
                        pred.get("op_scores", {}).items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:3]
                },
            }
        )
    return rows[:limit]


def primary_op_counts(rows: list[Label]) -> dict[str, int]:
    return dict(sorted(Counter(primary_op(row) for row in rows).items()))
