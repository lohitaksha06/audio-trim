"""One-shot repair of the Audelle paper's citation scheme.

The first draft mixed two incompatible conventions: named `\\bibitem{key}`
entries with hand-typed `[n]` markers in the text. In LaTeX a hand-typed `[n]` is
just text -- it is not a citation -- and IEEEtran numbers entries by their order
in `thebibliography`, not by order of appearance, so the markers would have been
numbered wrongly even if they had been real citations.

This script converts the draft to a consistent scheme:

  * in-text `[n]` becomes `\\cite{key}`;
  * the bibliography is reordered so that entry *n* is the one whose key was
    remapped to *n*;
  * `\\bibitem` keys stay stable, so `paper/REFERENCES.md` keeps matching.

Run once, then `python paper/check_paper.py` to confirm.
"""

from __future__ import annotations

import re
from pathlib import Path

TEX = Path(__file__).resolve().parent / "audelle_ieee.tex"

# Which key the draft's hand-typed `[n]` markers referred to.
OLD_TO_KEY = {
    1: "rouard2023hybrid",
    2: "hennequin2020spleeter",
    3: "stoter2019openunmix",
    4: "luo2019convtasnet",
    5: "hayes2024ddsp",
    6: "leroux2019sdr",
    7: "ephraim1992statistical",
    8: "tzanetakis2002genre",
    9: "mcfee2015librosa",
    10: "krumhansl1982perceived",
    11: "davis1990comparison",
    12: "rafii2013repet",
    13: "rousseeuw1987silhouettes",
    14: "breiman2001random",
    15: "radford2022whisper",
    16: "bazin2020spectrogram",
}


def split_bibliography(src: str) -> tuple[str, list[str], str]:
    head = r"\begin{thebibliography}{99}"
    tail = r"\end{thebibliography}"
    i = src.index(head)
    j = src.index(tail)
    body = src[i + len(head): j]
    entries = re.split(r"(?=\\bibitem\{)", body)
    return src[:i + len(head)], [e for e in entries if e.strip()], src[j:]


def main() -> None:
    src = TEX.read_text(encoding="utf-8")
    head, entries, tail = split_bibliography(src)

    by_key = {}
    for entry in entries:
        key = re.match(r"\\bibitem\{([^}]+)\}", entry).group(1)
        by_key[key] = entry

    body = head  # everything before the bibliography
    unknown: set[int] = set()

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if n not in OLD_TO_KEY:
            unknown.add(n)
            return m.group(0)
        return rf"\cite{{{OLD_TO_KEY[n]}}}"

    body, count = re.subn(r"\[(\d{1,2})\]", repl, body)
    if unknown:
        raise SystemExit(f"unmapped citation markers: {sorted(unknown)}")
    print(f"converted {count} in-text markers to \\cite{{}}")

    # Derive the bibliography order from the text instead of hardcoding it.
    # IEEEtran numbers entries in the order they appear in `thebibliography`,
    # so that order must equal the order of first citation. Hardcoding this was
    # the previous bug: the guessed order was wrong and produced a mismatch.
    order: list[str] = []
    for group in re.findall(r"\\cite\{([^}]+)\}", body):
        for k in (x.strip() for x in group.split(",")):
            if k and k not in order:
                order.append(k)

    missing = [k for k in order if k not in by_key]
    if missing:
        raise SystemExit(f"cited but no bibitem: {missing}")
    uncited = [k for k in by_key if k not in order]
    if uncited:
        raise SystemExit(f"bibitem never cited: {uncited}")

    out = body.rstrip() + "\n\n" + "".join(by_key[k] for k in order) + tail
    TEX.write_text(out, encoding="utf-8")
    print(f"bibliography reordered to match first citation, {len(order)} entries:")
    for i, k in enumerate(order, start=1):
        print(f"  [{i:>2}] {k}")


if __name__ == "__main__":
    main()