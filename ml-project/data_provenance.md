# Data provenance

What was downloaded for the "Fix my recording" project, where from, and its
integrity hash. Everything under `data/` is gitignored; this file is tracked so
the provenance survives a clone.

## VoiceBank-DEMAND — clean speech (48 kHz, resampled to 16 kHz locally)

- Source: Edinburgh DataShare, <https://datashare.ed.ac.uk/handle/10283/2791>, DOI 10.7488/ds/2117
- License: CC BY 4.0
- Files (downloaded, resumable):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `clean_trainset_28spk_wav.zip` | 2,486,057,279 | `DBB13B9FFE752C58764D75DE272706A2B467744F33455CEC6DEB2B12CA01DF66` |
| `clean_testset_wav.zip` | 154,332,064 | `B60BFC2B9DF38466D8EB45057C087C6E5DEE196C67A0ED7BCA628AE2A8611C4E` |

## DEMAND — noise (16 kHz builds)

- Source: Zenodo record 1227121, <https://zenodo.org/records/1227121>
- License: CC BY 4.0
- Files (downloaded):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `OHALLWAY_16k.zip` | 77,941,854 | `4D5EF858A05954F03340048131F60A2004B7B28AFC65F7BD2301636A7002D6CB` |
| `DLIVING_16k.zip` | 80,170,627 | `2B1726FE06E41551CE2397F2AAF3E4FB692C912D81914B04708DF0BBB5252338` |
| `OMEETING_16k.zip` | 82,703,355 | `512016CAD3A42511A0FB7E3C74D6924E6EFDE4921A6101361F1306BF90BF2E5A` |

Each DEMAND environment unpacks to 16 mono WAV files (~146.5 MB).

## What we intentionally did NOT download

- `noisy_trainset_28spk_wav.zip` (2.83 GB) and `noisy_testset_wav.zip` (171 MB):
  we build our own noise mixtures at continuous SNRs, so the bundled noisy
  halves are redundant.
- `clean_trainset_56spk_wav.zip` (4.77 GB): the 28-speaker set is enough.
- The other 15 DEMAND environments (~7 GB total): three environments are enough
  for a first run; more can be added for the unseen-noise generalisation test.
