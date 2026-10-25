# Audelle paper

`audelle_ieee.tex` — IEEE conference format (IEEEtran), 9 sections, 16
references.

## Building

The PDF in this directory was built with
[Tectonic](https://tectonic-typesetting.github.io/) 0.17.0, which fetches the
TeX packages it needs on first run:

```powershell
tectonic audelle_ieee.tex
```

Tectonic reruns TeX and `xdvipdfmx` itself, so one command is enough. The
result is 6 pages, with the references sharing the final page. There are two
cosmetic `Underfull \hbox` warnings (the abstract and the failure-modes intro),
and no errors.

With TeX Live instead:

```powershell
# IEEEtran ships with most TeX Live installs
pdflatex -interaction=nonstopmode audelle_ieee.tex
bibtex audelle_ieee          # not used: thebibliography is hand-written
pdflatex -interaction=nonstopmode audelle_ieee.tex
pdflatex -interaction=nonstopmode audelle_ieee.tex
```

Two passes minimum, three recommended, so that `\ref` resolves. Rebuild
`audelle_ieee.pdf` after any edit to the `.tex` file; do not edit the PDF.

## Checking it without LaTeX

```powershell
python paper\check_paper.py
```

This verifies, without a PDF toolchain:

- braces balance and every environment is opened/closed exactly once;
- every `\cite{key}` resolves to a `\bibitem{key}`, and every `\bibitem` is cited;
- **the bibliography is in first-citation order**, which is what IEEEtran numbers
  by — an out-of-order bibliography renders numbers that contradict the text;
- no stray hand-typed `[n]` remains, because in LaTeX that is plain text and not
  a citation;
- every `\ref` resolves and every `\label` is referenced;
- `Section~<roman>` cross-references point at sections that exist.

It currently reports `structure OK`.

## Files

| File | Purpose |
|---|---|
| `audelle_ieee.tex` | The paper |
| `audelle_ieee.pdf` | Built PDF (Tectonic 0.17.0), 6 pages |
| `REFERENCES.md` | Every reference, what verified it, and what was rejected |
| `check_paper.py` | Structural validation, run in CI-style |
| `fix_citations.py` | One-shot conversion of hand-typed `[n]` markers to `\cite{}` |

## Reproducing the numbers in the paper

```powershell
python scripts\evaluate_genre_classifier.py --clips-per-genre 10
python -m pytest server\tests -q
python server\ml\eval\prompt_bench.py
```

The genre evaluation writes `server/ml/models/genre_eval.json`. It previously
printed to stdout and threw the audio away, so nothing in the repository recorded
what the shipped artefact actually scores.

## What the paper deliberately does not claim

- **Inpainting quality.** Duration preservation and absence of silence are
  measured. Recovery of the original *notes* is not claimed, because on
  through-composed material that audio no longer exists in the file.
- **Seam smoothness as an improvement.** A 50 ms crossfade can post a smaller
  single-sample step than the reconstructed splice. The tests assert a bound, not
  superiority.
- **Genre accuracy against the literature.** 100 clips is not a benchmark, so no
  comparison is made.
- **Any perceptual quality claim.** No listening test was conducted.