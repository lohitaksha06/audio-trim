"""One command that must pass before a change ships.

    python -m scripts.release_check

Runs the server tests, the frontend tests, ESLint, the production build, and
prints the two NLU benchmark scores. Exits non-zero on the first failure.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"


def run(cmd: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"release check failed: {' '.join(cmd)}")


def main() -> None:
    run([sys.executable, "-m", "pytest", "server/tests", "-q"], ROOT)

    bun = shutil.which("bun")
    if not bun:
        raise SystemExit("bun is not on PATH; the frontend tests cannot run")
    npx = shutil.which("npx") or "npx"

    run([bun, "test", "src"], WEBSITE)
    run([npx, "eslint", "src"], WEBSITE)
    run([npx, "next", "build"], WEBSITE)

    sys.path.insert(0, str(ROOT))
    from server.ml.eval.prompt_bench import evaluate_prompt_bench, evaluate_real_bench

    curated = evaluate_prompt_bench()
    real = evaluate_real_bench()
    print("\nNLU benchmark")
    print(f"  curated: {curated['correct']}/{curated['total']} ({curated['accuracy'] * 100:.1f}%)")
    print(f"  real:    {real['correct']}/{real['total']} ({real['accuracy'] * 100:.1f}%)")
    if curated["accuracy"] < 1.0:
        raise SystemExit("curated prompt bench regressed")
    if real["accuracy"] < 0.8:
        raise SystemExit("real-prompt bench fell below 80%")


if __name__ == "__main__":
    main()
