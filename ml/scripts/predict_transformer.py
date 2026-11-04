"""Run a trained transformer checkpoint on a JSONL file.

Usage:
    python scripts/predict_transformer.py --input data/heldout_prompts.jsonl \
        --output models/transformer/heldout_predictions.json

This is also the hook for a future inference demo; it does not touch Audelle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _common
from instruction_ml.data import load_jsonl

try:
    import torch
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install torch + transformers first") from exc

from train_transformer import MultiTaskModel, predict_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(_common.MODELS / "transformer"))
    parser.add_argument("--input", default=str(_common.DATA / "heldout_prompts.jsonl"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    rows = load_jsonl(args.input)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = MultiTaskModel(config["model_name"])
    state = torch.load(checkpoint / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    preds = predict_rows(
        model,
        rows,
        tokenizer,
        int(config["max_len"]),
        16,
        device,
        config.get("op_thresholds"),
        config.get("target_thresholds"),
    )
    output = Path(args.output) if args.output else checkpoint / "predictions.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(preds, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(preds)} predictions to {output}")
    for item in preds[:5]:
        print(f"  {item['id']}: {item['ops']} / {item['targets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
