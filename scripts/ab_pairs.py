"""Build a blind A/B/X listening set from two renders.

This does not judge the audio; a human does. It loudness-matches B to A (RMS),
writes A.wav, B.wav and X.wav (X is a random copy of one), and seals the answer
in answer.txt so the listener can score without knowing which is which.

Usage:
    python -m scripts.ab_pairs --a clean_a.wav --b clean_b.wav --out validation/run1
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    a, sr_a = sf.read(args.a, dtype="float32")
    b, sr_b = sf.read(args.b, dtype="float32")
    if sr_a != sr_b:
        raise SystemExit("A and B must share a sample rate")
    if len(a) != len(b):
        raise SystemExit("A and B must have the same length for a fair A/B")

    # Match B's RMS to A so loudness does not decide the preference.
    b = b * (rms(a) / rms(b))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "A.wav", a, sr_a)
    sf.write(out / "B.wav", b, sr_a)

    x_is_a = random.random() < 0.5
    shutil.copyfile(out / ("A.wav" if x_is_a else "B.wav"), out / "X.wav")
    (out / "answer.txt").write_text(
        "X is A\n" if x_is_a else "X is B\n", encoding="utf-8"
    )
    (out / "README.txt").write_text(
        "Blind A/B/X set\n"
        "1. Match playback level and listen to the same 10-20 s excerpt, looped.\n"
        "2. Compare A vs B, then decide whether X matches A or B.\n"
        "3. Do not open answer.txt until you have written your score.\n"
        "4. Run at least 10 trials across a corpus before claiming a win.\n",
        encoding="utf-8",
    )
    print(f"wrote {out} (X is {'A' if x_is_a else 'B'})")


if __name__ == "__main__":
    main()
