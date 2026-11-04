"""Interactive inference demo.

Usage:
    python scripts/predict.py "make the drums louder by 6 db"
    python scripts/predict.py --rules --json "remove speaker 2"
    python scripts/predict.py --interactive
"""

from __future__ import annotations

import argparse
import json

import _common
from instruction_ml.model import DEFAULT_ARTIFACT, BaselineModel, top_k
from instruction_ml.rules import rule_label
from instruction_ml.schema import OPERATIONS, TARGETS


def describe(pred: dict) -> str:
    if not pred.get("ops"):
        return "No action recognised."
    lines: list[str] = []
    for op in pred["ops"]:
        targets = ", ".join(pred.get("targets", [])) or "full_mix"
        description = OPERATIONS.get(op, op)
        amount = pred.get("amounts", {}).get(op)
        if amount:
            lines.append(
                f"- {op:14s} -> {targets}  ({amount['amount']} {amount['unit']})  {description}"
            )
        else:
            lines.append(f"- {op:14s} -> {targets}  {description}")
    if pred.get("span"):
        start, end = pred["span"]
        lines.append(f"- span          -> {start}s .. {end}s")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="*", help="instruction text")
    parser.add_argument("--rules", action="store_true", help="use the rule baseline")
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    parser.add_argument("--interactive", action="store_true", help="read prompts from stdin")
    args = parser.parse_args()

    if args.rules:
        def predict(text: str) -> dict:
            label = rule_label(text)
            return {
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
                "op_scores": {},
            }
        source = "rule baseline"
    else:
        model = BaselineModel.load(args.artifact)
        predict = model.predict
        source = f"tfidf-logreg ({model.train_size} training rows)"

    def show(text: str) -> None:
        pred = predict(text)
        if args.json:
            print(json.dumps({"text": text, "source": source, **pred}, indent=2))
            return
        print(f"\n> {text}")
        print(describe(pred))
        if pred.get("op_scores"):
            print(
                "  top ops: "
                + ", ".join(f"{name} {score:.2f}" for name, score in top_k(pred["op_scores"]))
            )
        print(
            "  top targets: "
            + ", ".join(
                f"{name} {score:.2f}" for name, score in top_k(pred.get("target_scores", {}))
            )
        )

    if args.interactive:
        print(f"Model: {source}. Type a prompt, or 'quit'.")
        while True:
            try:
                text = input("\ninstruction> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in {"quit", "exit", "q"}:
                break
            if text:
                show(text)
        return 0

    text = " ".join(args.prompt).strip()
    if not text:
        parser.error("provide a prompt or use --interactive")
    show(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
