r"""Sanity-check the paper's LaTeX before it reaches a PDF toolchain.

No LaTeX toolchain is installed on this machine, so the structural checks that
normally come free from `\documentclass` are done here instead:

  - braces balance, and every environment is opened and closed exactly once;
  - every `\cite{key}` resolves to a `\bibitem{key}`, and every `\bibitem` is
    cited at least once;
  - citations appear in ascending first-appearance order, because IEEEtran
    numbers `thebibliography` entries in the order they are written -- a
    bibliography listed out of order renders numbers that contradict the text;
  - no stray hand-typed `[n]` citation remains (it would compile as plain text);
  - every `\ref` resolves, and every `\label` is referenced.

This catches the class of mistake that a renumbering pass introduces, which is
exactly how the first draft of this paper ended up with eight broken
cross-references and a bibliography whose numbering did not match its text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TEX = Path(__file__).resolve().parent / "audelle_ieee.tex"


def main() -> int:
    src = TEX.read_text(encoding="utf-8")
    problems: list[str] = []

    # --- structure -------------------------------------------------------
    depth = 0
    for i, ch in enumerate(src):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                problems.append(f"closing brace with no opener at offset {i}")
                break
    if depth != 0:
        problems.append(f"braces unbalanced by {depth}")

    for env in ("document", "abstract", "thebibliography", "IEEEkeywords"):
        b, e = src.count(rf"\begin{{{env}}}"), src.count(rf"\end{{{env}}}")
        if b != e or b != 1:
            problems.append(f"environment {env}: {b} begin, {e} end (expected 1/1)")

    # --- citations -------------------------------------------------------
    bibitems = re.findall(r"\\bibitem\{([^}]+)\}", src)
    bib_keys = list(dict.fromkeys(bibitems))  # bibliography order, deduplicated

    # Only the body may cite; the bibliography itself mentions no \cite.
    body_end = src.index(r"\begin{thebibliography}")
    body = src[:body_end]
    cited: list[str] = []
    for group in re.findall(r"\\cite\{([^}]+)\}", body):
        cited.extend(k.strip() for k in group.split(","))

    dangling = sorted(set(cited) - set(bib_keys))
    if dangling:
        problems.append(f"cited but no such bibitem: {dangling}")
    uncited = [k for k in bib_keys if k not in cited]
    if uncited:
        problems.append(f"bibitem never cited: {uncited}")

    # IEEEtran numbers in bibliography order, so first appearance must be 1,2,3...
    first_seen: list[str] = []
    for k in cited:
        if k not in first_seen:
            first_seen.append(k)
    expected = [k for k in bib_keys if k in first_seen]
    if first_seen != expected:
        for i, (a, b) in enumerate(zip(first_seen, expected), start=1):
            if a != b:
                problems.append(
                    f"bibliography order mismatch at position {i}: "
                    f"text first cites '{a}' but entry {i} is '{b}'"
                )
                break

    # A hand-typed [n] is plain text in LaTeX, not a citation.
    stray = [
        m.group(0)
        for m in re.finditer(r"(?<!\\)\[\d{1,2}\]", body)
    ]
    if stray:
        problems.append(f"hand-typed numeric citations left in text: {stray}")

    # --- cross references ------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
    bad_refs = sorted(refs - labels)
    if bad_refs:
        problems.append(f"\\ref to unknown label: {bad_refs}")
    unused_labels = sorted(labels - refs)
    if unused_labels:
        problems.append(f"label defined but never \\ref'd: {unused_labels}")

    # --- section cross references (Section~IV-A style) --------------------
    sections = re.findall(r"\\section\{([^}]+)\}", src)
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    available = set(roman[: len(sections)])
    for num in re.findall(r"Section~([IVX]+-[A-H])", src):
        sec = num.split("-")[0]
        if sec not in available:
            problems.append(
                f"cross-reference Section~{num} but only {len(sections)} sections exist"
            )

    # --- report ----------------------------------------------------------
    print(f"sections:  {len(sections)} -> " + ", ".join(
        f"{roman[i]}={s}" for i, s in enumerate(sections)))
    print(f"references: {len(bib_keys)} in thebibliography, {len(set(cited))} cited")

    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nstructure OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())