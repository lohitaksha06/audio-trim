"""Optional LLM paraphrase expansion — proposals only, never auto-accepted.

Reads the hand-labeled seeds, asks an OpenAI-compatible chat endpoint for
paraphrases, and writes `data/proposed_paraphrases.jsonl` for human review.
Nothing is added to the dataset until a person checks the labels and copies the
rows into `data/seed_prompts.jsonl` (with new seed ids).

Usage:
    set OPENAI_API_KEY=...
    python scripts/expand_with_llm.py --limit 10
    python scripts/expand_with_llm.py --dry-run

Environment:
    OPENAI_API_KEY   required unless --dry-run
    OPENAI_BASE_URL  default https://api.openai.com/v1
    OPENAI_MODEL     default gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

import _common
from instruction_ml.data import load_jsonl
from instruction_ml.schema import OPERATIONS, TARGETS

SEED_PATH = _common.DATA / "seed_prompts.jsonl"
OUT_PATH = _common.DATA / "proposed_paraphrases.jsonl"

SYSTEM = """You rewrite audio-editing instructions.
Keep the meaning identical. Do not add or remove operations, targets, numbers,
units, or timestamps. Return strict JSON only."""


def build_prompt(row) -> str:
    return f"""Instruction: {row.text!r}
Labels: ops={row.ops}, targets={row.targets}, amount={row.amount}, unit={row.unit}, span={row.span}

Write 5 natural paraphrases a real user might type. Keep all labels unchanged.
Allowed operations: {", ".join(OPERATIONS)}
Allowed targets: {", ".join(TARGETS)}
Return JSON: {{"paraphrases": ["...", "..."]}}"""


def call_llm(prompt: str, model: str, base_url: str, api_key: str) -> list[str]:
    payload = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return [str(item).strip() for item in parsed.get("paraphrases", []) if str(item).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="seed rows to expand")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = load_jsonl(SEED_PATH)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("Set OPENAI_API_KEY or pass --dry-run")

    proposals = []
    for row in seeds[: args.limit]:
        if args.dry_run:
            print(build_prompt(row))
            print("-" * 60)
            continue
        try:
            paraphrases = call_llm(build_prompt(row), args.model, args.base_url, api_key)
        except Exception as exc:  # network / quota / schema errors
            print(f"{row.id}: request failed: {exc}")
            continue
        for text in paraphrases:
            proposals.append(
                {
                    "seed_id": row.id,
                    "text": text,
                    "suggested_ops": row.ops,
                    "suggested_targets": row.targets,
                    "amount": row.amount,
                    "unit": row.unit,
                    "span": row.span,
                    "verified": False,
                    "note": "LLM paraphrase - check meaning before accepting",
                }
            )
        print(f"{row.id}: {len(paraphrases)} proposals")

    if args.dry_run:
        return 0

    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for proposal in proposals:
            fh.write(json.dumps(proposal, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(proposals)} proposals to {OUT_PATH}")
    print("Review each row, set verified=true, then copy accepted rows into seed_prompts.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
