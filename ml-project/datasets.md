# Datasets

## VoiceBank-DEMAND

- **URL:** <https://datashare.ed.ac.uk/handle/10283/2791>
- **License:** CC BY 4.0
- **Size:** several GB at 48 kHz; the 16 kHz derivatives are much smaller.
  Check the archive listing before downloading — the plan asks for the clean
  set and a small test subset, not the whole corpus.
- **Contents:** 11,572 noisy/clean utterance pairs from 28 speakers for training;
  824 utterances from 2 unseen speakers for test. Training noise is 8 real
  DEMAND recordings plus 2 artificial types. The test noises never appear in
  training.
- **What the models learn:** mapping a noisy utterance to its clean reference.
  The clean half is also the speech source for synthesising our own mixtures and
  clipped files, where the labels are exact.

## DEMAND noise

- **URL:** <https://zenodo.org/record/1227121>
- **License:** see the record; commonly CC BY-SA for the noise recordings.
- **Size:** a few GB for the full set. We need a handful of environments.
- **What the models learn:** the noise distribution. We mix at continuous SNRs
  rather than the benchmark's four fixed levels so the diagnoser is not trained
  on a handful of values.

## Splits (ours, on top of the benchmark's)

- **Train / validation:** speaker-disjoint within the 28 training speakers.
- **Test:** the official 2 unseen speakers, untouched.
- **Secondary test:** unseen noise types, to check generalisation.

Never split by utterance. Splitting by utterance leaks the same speaker into
both sides and inflates every score.

## Not used

- Room impulse responses / OpenSLR RIRs — only relevant if the dereverberator
  stretch goal is attempted.
- MUSDB18-HQ, Slakh, AudioSet, NSynth — these belonged to the earlier
  separation / classification plan, which is dropped.

## Download sources

Verified 2026-11-28. Sizes are exact bytes read from the provider's metadata
API (DSpace REST `sizeBytes`, Zenodo `files[].size`, Hugging Face tree API),
not the rounded strings on the HTML pages. Nothing was downloaded.

### VoiceBank-DEMAND, original (University of Edinburgh DataShare)

- **Record:** <https://datashare.ed.ac.uk/handle/10283/2791>
- **DOI:** <https://doi.org/10.7488/ds/2117>
- **License:** CC BY 4.0 (the record page links
  <https://creativecommons.org/licenses/by/4.0/>).
- **Sample rate: 48 kHz only.** The record's own description says the database
  "was designed to train and test speech enhancement methods that operate at
  48kHz". There is no 16 kHz bitstream on this record — the 16 kHz version
  everyone cites is a downstream resample, so a 16 kHz pipeline either
  downloads 48 kHz and resamples, or uses a mirror.

| File | Bytes | ~Size |
| --- | ---: | ---: |
| `clean_testset_wav.zip` | 154,332,064 | 147 MB |
| `noisy_testset_wav.zip` | 170,578,854 | 163 MB |
| `testset_txt.zip` | 440,277 | 430 KB |
| `clean_trainset_28spk_wav.zip` | 2,486,057,279 | 2.32 GB |
| `noisy_trainset_28spk_wav.zip` | 2,830,205,201 | 2.64 GB |
| `trainset_28spk_txt.zip` | 6,489,312 | 6.2 MB |
| `clean_trainset_56spk_wav.zip` | 4,770,619,279 | 4.44 GB |
| `noisy_trainset_56spk_wav.zip` | 5,627,055,496 | 5.24 GB |

Download links are `/bitstreams/<uuid>/download` under the same host (the uuids
are in `scripts/download_data.py`). The record publishes **no sha256** — the
DSpace `checksum` field is `null` for every bitstream, so integrity has to be
checked by size only.

### VoiceBank-DEMAND, 16 kHz mirror (Hugging Face)

- **Repo:** <https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k>
  (most-downloaded 16 kHz mirror: ~3.8k downloads, 17 likes; several identical
  forks exist and are interchangeable).
- **License:** `cc-by-4.0`, stated both in the repo card and in the repo's
  `license.txt`.
- **Format:** Parquet, two audio columns per row (`clean`, `noisy`), both at
  **16,000 Hz** (declared in the dataset card's feature schema).
- **Total download:** 2,283,261,552 bytes (2.28 GB) for all 12,396 pairs.

| File | Bytes | Rows |
| --- | ---: | ---: |
| `data/test-00000-of-00001.parquet` | 132,200,447 | 824 |
| `data/train-00000-of-00005.parquet` | 444,302,320 | ~2,314 |
| `data/train-00001-of-00005.parquet` | 416,366,498 | ~2,314 |
| `data/train-00002-of-00005.parquet` | 430,855,381 | ~2,314 |
| `data/train-00003-of-00005.parquet` | 444,005,802 | ~2,314 |
| `data/train-00004-of-00005.parquet` | 415,510,617 | ~2,314 |

**Smallest usable 16 kHz subset (recommended): the test shard,
132,200,447 bytes (132.2 MB), 824 utterances from the 2 unseen speakers.** It is
self-contained (one file), already 16 kHz, and CC BY 4.0. For training speech you
need at least one train shard: the smallest is
`train-00004-of-00005.parquet` at 415,510,617 bytes (~2,314 pairs). One test
shard + one train shard = 547,711,064 bytes (548 MB), which is the practical
minimum for a real run.

A smaller 16 kHz option exists but is **not** usable without a license check:
`DynamicSuperb/SuperbSE_VoiceBankDEMAND-Test` is 63,982,214 bytes (64.0 MB),
200 rows, 16 kHz, columns `audio` (noisy) + `label` (clean) — but the repo
declares **no license tag at all**. Use only if that is resolved.

### DEMAND noise (Zenodo)

- **Record:** <https://zenodo.org/records/1227121>
- **DOI:** <https://doi.org/10.5281/zenodo.1227121>
- **License: CC BY 4.0** (`license.id = "cc-by-4.0"` in the Zenodo API). The
  earlier note in this file said "commonly CC BY-SA" — that was wrong.
- **Total: 7,363,293,564 bytes (7.36 GB)** across 38 files: 18 environments at
  48 kHz, 17 at 16 kHz (`SCAFE` has no 16 kHz build), plus `scripts.zip` and
  `DEMAND.pdf`.
- Zenodo publishes **md5** per file, not sha256.

16 kHz files, smallest first:

| File | Bytes |
| --- | ---: |
| `OHALLWAY_16k.zip` | 77,941,854 |
| `DLIVING_16k.zip` | 80,170,627 |
| `OMEETING_16k.zip` | 82,703,355 |
| `NPARK_16k.zip` | 86,029,332 |
| `NFIELD_16k.zip` | 86,536,782 |
| `OOFFICE_16k.zip` | 88,995,191 |
| `NRIVER_16k.zip` | 98,689,565 |
| `DWASHING_16k.zip` | 102,343,250 |
| `PCAFETER_16k.zip` | 107,431,494 |
| `SCAFE_16k.zip` | *(not published)* |
| `DKITCHEN_16k.zip` | 110,501,049 |
| `SPSQUARE_16k.zip` | 110,928,528 |
| `PRESTO_16k.zip` | 111,292,815 |
| `STRAFFIC_16k.zip` | 118,572,691 |
| `PSTATION_16k.zip` | 119,441,094 |
| `TMETRO_16k.zip` | 126,589,090 |
| `TBUS_16k.zip` | 128,916,709 |
| `TCAR_16k.zip` | 130,045,975 |

**Smallest usable noise subset: `OHALLWAY_16k.zip`, 77,941,854 bytes
(77.9 MB).** One environment is enough to build mixtures; `DLIVING`, `OOFFICE`
and `NPARK` are the usual second choices for noise-type-disjoint tests.

### Could not be confirmed

- **sha256 for the DataShare bitstreams** — not published (only sizes).
- **sha256 for the Zenodo files** — Zenodo exposes md5 only.
- **Exact row counts per 16 kHz train shard** — the card gives the split total
  (11,572) and file sizes, not a per-shard count; the ~2,314 figures are the
  even split, not a published number.
- **The DataShare HTML page's GB values are rounded** ("2.32 GB"); the byte
  counts above come from the REST API and are exact.
