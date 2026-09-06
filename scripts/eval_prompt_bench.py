"""CLI: run prompt bench and print report for CI / paper."""

import json
from server.ml.eval.prompt_bench import evaluate_prompt_bench

if __name__ == "__main__":
    res = evaluate_prompt_bench()
    print(json.dumps(res, indent=2))
    # exit code 1 if accuracy < 0.95 so CI fails on NLU regressions
    raise SystemExit(0 if res["accuracy"] >= 0.95 else 1)
