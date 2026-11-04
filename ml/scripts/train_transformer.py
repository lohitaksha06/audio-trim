"""Train a small multi-task transformer: DistilBERT/BERT-tiny backbone with

  * a multi-label operation head
  * a multi-label target head
  * a unit classification head (dB / seconds / factor / bpm / none)
  * a scalar amount regression head

Usage:
    python scripts/train_transformer.py --model-name prajjwal1/bert-tiny --epochs 8
    python scripts/train_transformer.py --model-name distilbert-base-uncased --epochs 4

Requires `torch` and `transformers` (not in the base requirements).
Writes checkpoint + test predictions to models/transformer/.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import _common
from instruction_ml.data import load_jsonl
from instruction_ml.metrics import evaluate_by_source
from instruction_ml.schema import OPERATION_KEYS, TARGET_KEYS, UNITS

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise SystemExit(
        "torch/transformers are not installed. Install them first:\n"
        "  pip install torch transformers\n"
        f"({exc})"
    ) from exc

UNIT_LABELS = ["none", *UNITS]
UNIT_TO_INDEX = {name: index for index, name in enumerate(UNIT_LABELS)}


class PromptDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        encoded = self.tokenizer(
            row.text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "ops": torch.tensor(
                [1.0 if op in row.ops else 0.0 for op in OPERATION_KEYS],
                dtype=torch.float,
            ),
            "targets": torch.tensor(
                [1.0 if target in row.targets else 0.0 for target in TARGET_KEYS],
                dtype=torch.float,
            ),
            "amount": torch.tensor(
                row.amount if row.amount is not None else 0.0, dtype=torch.float
            ),
            "has_amount": torch.tensor(
                1.0 if row.amount is not None else 0.0, dtype=torch.float
            ),
            "unit": torch.tensor(
                UNIT_TO_INDEX.get(row.unit or "none", 0), dtype=torch.long
            ),
            "index": torch.tensor(index, dtype=torch.long),
        }


class MultiTaskModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.op_head = nn.Linear(hidden, len(OPERATION_KEYS))
        self.target_head = nn.Linear(hidden, len(TARGET_KEYS))
        self.unit_head = nn.Linear(hidden, len(UNIT_LABELS))
        self.amount_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask) -> dict:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS pooling works for BERT/DistilBERT; fall back to mean pooling.
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            pooled = output.pooler_output
        else:
            pooled = output.last_hidden_state.mean(dim=1)
        pooled = self.dropout(pooled)
        return {
            "ops": self.op_head(pooled),
            "targets": self.target_head(pooled),
            "unit": self.unit_head(pooled),
            "amount": self.amount_head(pooled).squeeze(-1),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(rows, tokenizer, max_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        PromptDataset(rows, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


@torch.no_grad()
def collect_outputs(model, rows, tokenizer, max_len: int, batch_size: int, device) -> dict:
    model.eval()
    loader = make_loader(rows, tokenizer, max_len, batch_size, shuffle=False)
    op_probs, target_probs, unit_probs, amounts = [], [], [], []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        op_probs.append(torch.sigmoid(logits["ops"]).cpu().numpy())
        target_probs.append(torch.sigmoid(logits["targets"]).cpu().numpy())
        unit_probs.append(torch.softmax(logits["unit"], dim=-1).cpu().numpy())
        amounts.append(logits["amount"].cpu().numpy())
    return {
        "ops": np.concatenate(op_probs),
        "targets": np.concatenate(target_probs),
        "units": np.concatenate(unit_probs),
        "amounts": np.concatenate(amounts),
    }


def tune_thresholds(rows, outputs: dict) -> tuple[dict[str, float], dict[str, float]]:
    from sklearn.metrics import f1_score

    def tune(probs, gold, classes):
        thresholds: dict[str, float] = {}
        for index, name in enumerate(classes):
            truth = gold[:, index]
            if truth.min() == truth.max():
                continue
            best_f1, best = -1.0, 0.5
            for threshold in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
                score = f1_score(
                    truth, (probs[:, index] >= threshold).astype(int), zero_division=0
                )
                if score > best_f1:
                    best_f1, best = score, threshold
            thresholds[name] = best
        return thresholds

    op_gold = np.array(
        [[1 if op in row.ops else 0 for op in OPERATION_KEYS] for row in rows]
    )
    target_gold = np.array(
        [[1 if target in row.targets else 0 for target in TARGET_KEYS] for row in rows]
    )
    return (
        tune(outputs["ops"], op_gold, list(OPERATION_KEYS)),
        tune(outputs["targets"], target_gold, list(TARGET_KEYS)),
    )


def predict_rows(
    model,
    rows,
    tokenizer,
    max_len: int,
    batch_size: int,
    device,
    op_thresholds: dict[str, float] | None = None,
    target_thresholds: dict[str, float] | None = None,
) -> list[dict]:
    outputs = collect_outputs(model, rows, tokenizer, max_len, batch_size, device)
    op_thresholds = op_thresholds or {}
    target_thresholds = target_thresholds or {}
    preds: list[dict] = []
    for row_index in range(outputs["ops"].shape[0]):
        op_probs = outputs["ops"][row_index]
        target_probs = outputs["targets"][row_index]
        unit_probs = outputs["units"][row_index]
        amounts = outputs["amounts"]
        ops = [
            op
            for op, prob in zip(OPERATION_KEYS, op_probs)
            if prob >= op_thresholds.get(op, 0.5)
        ]
        targets = [
            target
            for target, prob in zip(TARGET_KEYS, target_probs)
            if prob >= target_thresholds.get(target, 0.5)
        ]
        unit_index = int(unit_probs.argmax())
        unit = UNIT_LABELS[unit_index]
        amount = float(amounts[row_index])
        if unit == "factor":
            amount = max(0.25, min(4.0, amount))
        elif unit == "dB":
            amount = max(-24.0, min(24.0, amount))
        elif unit == "bpm":
            amount = max(30.0, min(300.0, amount))
        if not ops:
            op_index = int(op_probs.argmax())
            if op_probs[op_index] > 0.2:
                ops = [OPERATION_KEYS[op_index]]
        preds.append(
            {
                "id": rows[row_index].id,
                "ops": ops,
                "targets": targets or (["full_mix"] if ops else []),
                "amount": round(amount, 2) if unit != "none" and ops else None,
                "unit": None if unit == "none" or not ops else unit,
                "amounts": (
                    {ops[0]: {"amount": round(amount, 2), "unit": unit}}
                    if unit != "none" and len(ops) == 1
                    else {}
                ),
                "span": None,
                "op_scores": {
                    op: round(float(prob), 4)
                    for op, prob in zip(OPERATION_KEYS, op_probs)
                },
                "target_scores": {
                    target: round(float(prob), 4)
                    for target, prob in zip(TARGET_KEYS, target_probs)
                },
                "unit_probs": {
                    name: round(float(prob), 4)
                    for name, prob in zip(UNIT_LABELS, unit_probs)
                },
            }
        )
    return preds


def combined_f1(rows, preds) -> float:
    from sklearn.metrics import f1_score

    def matrix(items, key, classes):
        return np.array(
            [[1 if name in getattr(row, key) else 0 for name in classes] for row in items]
        ), np.array(
            [[1 if name in pred.get(key, []) else 0 for name in classes] for pred in preds]
        )

    op_true, op_pred = matrix(rows, "ops", list(OPERATION_KEYS))
    target_true, target_pred = matrix(rows, "targets", list(TARGET_KEYS))
    return float(
        0.5 * f1_score(op_true, op_pred, average="micro", zero_division=0)
        + 0.5 * f1_score(target_true, target_pred, average="micro", zero_division=0)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="prajjwal1/bert-tiny")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default=str(_common.MODELS / "transformer"))
    parser.add_argument("--max-rows", type=int, default=0, help="smoke-test subset size")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_rows = load_jsonl(_common.SPLITS / "train.jsonl")
    val_rows = load_jsonl(_common.SPLITS / "val.jsonl")
    test_rows = load_jsonl(_common.SPLITS / "test.jsonl")
    if args.max_rows:
        train_rows = train_rows[: args.max_rows]
    print(
        f"Rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultiTaskModel(args.model_name).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    l1 = nn.SmoothL1Loss(reduction="none")

    train_loader = make_loader(
        train_rows, tokenizer, args.max_len, args.batch_size, shuffle=True
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            ops_loss = bce(logits["ops"], batch["ops"].to(device))
            targets_loss = bce(logits["targets"], batch["targets"].to(device))
            unit_loss = ce(logits["unit"], batch["unit"].to(device))
            amount_loss = l1(
                logits["amount"], batch["amount"].to(device)
            ) * batch["has_amount"].to(device)
            amount_loss = amount_loss.sum() / batch["has_amount"].sum().clamp(min=1.0)
            loss = ops_loss + targets_loss + 0.5 * unit_loss + 0.5 * amount_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")

        val_preds = predict_rows(
            model, val_rows, tokenizer, args.max_len, args.batch_size, device
        )
        val_f1 = combined_f1(val_rows, val_preds)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running / max(len(train_loader), 1), 4),
                "val_combined_f1": round(val_f1, 4),
            }
        )
        print(f"Epoch {epoch}: train loss {history[-1]['train_loss']:.4f}, val F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_outputs = collect_outputs(
        model, val_rows, tokenizer, args.max_len, args.batch_size, device
    )
    op_thresholds, target_thresholds = tune_thresholds(val_rows, val_outputs)
    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "max_len": args.max_len,
                "operations": list(OPERATION_KEYS),
                "targets": list(TARGET_KEYS),
                "units": UNIT_LABELS,
                "op_thresholds": op_thresholds,
                "target_thresholds": target_thresholds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    test_preds = predict_rows(
        model,
        test_rows,
        tokenizer,
        args.max_len,
        args.batch_size,
        device,
        op_thresholds,
        target_thresholds,
    )
    (output_dir / "test_predictions.json").write_text(
        json.dumps(test_preds, indent=2) + "\n", encoding="utf-8"
    )

    heldout_path = _common.DATA / "heldout_prompts.jsonl"
    if heldout_path.exists():
        heldout_rows = load_jsonl(heldout_path)
        heldout_preds = predict_rows(
            model,
            heldout_rows,
            tokenizer,
            args.max_len,
            args.batch_size,
            device,
            op_thresholds,
            target_thresholds,
        )
        (output_dir / "heldout_predictions.json").write_text(
            json.dumps(heldout_preds, indent=2) + "\n", encoding="utf-8"
        )

    results = evaluate_by_source(test_rows, test_preds, "transformer")
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "epochs": args.epochs,
                "max_len": args.max_len,
                "best_val_combined_f1": round(best_f1, 4),
                "op_thresholds": op_thresholds,
                "target_thresholds": target_thresholds,
                "history": history,
                "test": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nTest (all rows):")
    print(json.dumps(results["all"]["operations"], indent=2))
    print(json.dumps(results["all"]["targets"], indent=2))
    print(f"Artifacts: {output_dir}")
    print(
        "Evaluate against the other models with:\n"
        f"  python scripts/evaluate.py --transformer-preds {output_dir / 'test_predictions.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
