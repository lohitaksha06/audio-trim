# Reference verification

Every reference in `paper/audelle_ieee.tex` was checked against a live
bibliographic record before being written down. Nothing was cited from memory.
This file records what was checked, so the bibliography can be audited.

Verified on 2026-10-06. Sources used: **Crossref REST API**
(`api.crossref.org`) for anything with a DOI, **arXiv** for preprints, and
**Zenodo** for the AIMC paper.

One near-miss is recorded deliberately: arXiv `1712.06884` was assumed to be the
NSynth paper and is in fact *"Experimental entanglement of temporal order"*, a
quantum physics paper. It was caught before being cited and **not** used. The
NSynth dataset is therefore cited nowhere in this paper, because the training
data is referred to by its Hugging Face mirror rather than by paper.

## Verified references

Numbering below is the rendered numbering. IEEEtran numbers `thebibliography`
entries in the order they are written, so the bibliography is ordered by first
citation in the text; `check_paper.py` enforces that the two agree.

| # | Reference | Verified against | DOI / ID |
|---|---|---|---|
| 1 | Luo, Mesgarani, *Conv-TasNet*, IEEE/ACM TASLP | Crossref | `10.1109/TASLP.2019.2915167` |
| 2 | Hennequin, Khlif, Voituret, Moussallam, *Spleeter*, J. Open Source Softw. | Crossref | `10.21105/joss.02154` |
| 3 | Stöter, Uhlich, Liutkus, Mitsufuji, *Open-Unmix*, J. Open Source Softw. | Crossref | `10.21105/joss.01667` |
| 4 | Rouard, Massa, Défossez, *Hybrid Transformers for Music Source Separation*, ICASSP | Semantic Scholar + arXiv abs 2211.08553 | `10.1109/ICASSP49357.2023.10096956` |
| 5 | Ephraim, *Statistical-model-based speech enhancement systems*, Proc. IEEE | Crossref | `10.1109/5.168664` |
| 6 | Radford, Kim, Xu, Brockman, McLeavey, Sutskever, *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper) | arXiv abs 2212.04356, submitted 6 Dec 2022 | `arXiv:2212.04356` |
| 7 | Davis, Mermelstein, *Comparison of parametric representations for monosyllabic word recognition* | Crossref | `10.1016/B978-0-08-051584-7.50010-3` |
| 8 | Rousseeuw, *Silhouettes*, J. Comput. Appl. Math. | Crossref | `10.1016/0377-0427(87)90125-7` |
| 9 | Hayes, Shier, Fazekas, McPherson, Saitis, *A review of differentiable DSP for music and speech synthesis*, Frontiers in Signal Processing | Crossref | `10.3389/frsip.2023.1284100` |
| 10 | Bazin, Hadjeres, Esling, Malt, *Spectrogram Inpainting for Interactive Generation of Instrument Sounds*, AIMC 2020 | Zenodo record 4285406 | `10.5281/zenodo.4285406` |
| 11 | Rafii, Pardo, *REPET*, IEEE/ACM TASLP | Crossref | `10.1109/TASL.2012.2213249` |
| 12 | Tzanetakis, Cook, *Musical genre classification of audio signals*, IEEE Trans. Speech Audio Process. | Crossref | `10.1109/TSA.2002.800560` |
| 13 | Breiman, *Random Forests*, Machine Learning | Crossref | `10.1023/A:1010933404324` |
| 14 | Krumhansl, Bharucha, Kessler, *Perceived harmonic structure of chords in three related musical keys* | Crossref | `10.1037/0096-1523.8.1.24` |
| 15 | McFee et al., *librosa: Audio and music signal analysis in Python*, SciPy | Crossref | `10.25080/Majora-7b98e3ed-003` |
| 16 | Le Roux, Wisdom, Erdogan, Hershey, *SDR — Half-baked or well done?*, ICASSP | Crossref | `10.1109/ICASSP.2019.8683855` |

## Rejected during verification

Searched for, not found in a bibliographic record, therefore **not cited**:

- Groove MIDI dataset (Hawthorne et al.) — Crossref returned unrelated hits.
- Fitzgerald, *Harmonic/Percussive Separation using Median Filtering* (DAFx-10) —
  Crossref only returned Fitzgerald's 2012 IET paper, a different work.
- Böck / Ellis beat and downbeat tracking — ISMIR and older venues are not
  well indexed by Crossref.
- Powerset clustering (Wang et al.) — Crossref returned three different papers;
  none was the one intended.
- Krumhansl 1982 *Internal representations for music perception* — found, but
  the 1982 probe-tone profile paper [10] is the one actually used by the code.

## Volume and page numbers

Where Crossref supplied a volume, issue or page range it is included. Where it
did not, the field is omitted rather than filled in from memory. Every reference
carries a DOI or an arXiv identifier, which is the authoritative locator.

## Reproducing the numbers

```powershell
python scripts\evaluate_genre_classifier.py --clips-per-genre 10
python -m pytest server/tests -q
```

The genre evaluation writes `server/ml/models/genre_eval.json`. It previously
printed to stdout and discarded the audio, so nothing in the repository recorded
what the shipped artefact scored.