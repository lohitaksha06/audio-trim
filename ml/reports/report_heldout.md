# Evaluation — `heldout` split

Rows: **80**

## Comparison (all rows)

| Model | n | Op subset acc | Op micro F1 | Op macro F1 | Target micro F1 | Target macro F1 | Action exact | Amount MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | 80 | 0.562 | 0.709 | 0.664 | 0.951 | 0.810 | 0.512 | 0.533 |
| tfidf-logreg | 80 | 0.863 | 0.894 | 0.892 | 0.927 | 0.750 | 0.787 | 4.387 |

## Handwritten rows only (honest headline)

| Model | n | Op subset acc | Op micro F1 | Op macro F1 | Target micro F1 | Target macro F1 | Action exact | Amount MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | 80 | 0.562 | 0.709 | 0.664 | 0.951 | 0.810 | 0.512 | 0.533 |
| tfidf-logreg | 80 | 0.863 | 0.894 | 0.892 | 0.927 | 0.750 | 0.787 | 4.387 |

## Per-operation scores — tfidf-logreg

| Operation | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| remove_stem | 0.700 | 0.875 | 0.778 | 8 |
| isolate_stem | 0.800 | 0.667 | 0.727 | 6 |
| gain | 1.000 | 1.000 | 1.000 | 10 |
| fade | 1.000 | 1.000 | 1.000 | 7 |
| cut_segment | 0.833 | 0.625 | 0.714 | 8 |
| trim_segment | 0.857 | 0.857 | 0.857 | 7 |
| remove_speaker | 1.000 | 1.000 | 1.000 | 7 |
| denoise | 1.000 | 0.857 | 0.923 | 7 |
| normalize | 1.000 | 1.000 | 1.000 | 6 |
| speed | 0.857 | 0.857 | 0.857 | 7 |
| add_effect | 0.875 | 1.000 | 0.933 | 7 |
| remove_silence | 0.833 | 1.000 | 0.909 | 5 |

## Per-target scores — tfidf-logreg

| Target | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| full_mix | 0.976 | 0.954 | 0.965 | 43 |
| vocals | 1.000 | 0.909 | 0.952 | 11 |
| drums | 1.000 | 1.000 | 1.000 | 8 |
| bass | 1.000 | 1.000 | 1.000 | 5 |
| guitar | 1.000 | 1.000 | 1.000 | 5 |
| keys | 1.000 | 1.000 | 1.000 | 4 |
| other | 0.000 | 0.000 | 0.000 | 0 |
| speaker_1 | 0.200 | 1.000 | 0.333 | 1 |
| speaker_2 | 1.000 | 0.333 | 0.500 | 6 |

## Confusion matrix — tfidf-logreg (single-operation rows)

| actual \ predicted | gain | fade | remove_stem | add_effect | isolate_stem | cut_segment | trim_segment | none | remove_speaker | denoise | remove_silence | normalize | speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gain | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fade | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| remove_stem | 0 | 0 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| add_effect | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| isolate_stem | 0 | 0 | 2 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cut_segment | 0 | 0 | 1 | 0 | 0 | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| trim_segment | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| none | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| remove_speaker | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| denoise | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 0 | 0 |
| remove_silence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 |
| normalize | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 |
| speed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 5 |

## Amount errors (dB / seconds / factor / bpm)

```json
{
  "overall": {
    "n": 15,
    "mae": 4.387,
    "median_ae": 1.01
  },
  "by_operation_unit": {
    "fade|seconds": {
      "n": 4,
      "mae": 0.875,
      "median_ae": 0.815
    },
    "gain|dB": {
      "n": 8,
      "mae": 1.458,
      "median_ae": 0.9
    },
    "remove_silence|seconds": {
      "n": 1,
      "mae": 0.26,
      "median_ae": 0.26
    },
    "speed|bpm": {
      "n": 2,
      "mae": 25.19,
      "median_ae": 25.19
    }
  }
}
```

## Top confusions

```json
[
  {
    "actual": "remove_stem",
    "predicted": "add_effect",
    "count": 1
  },
  {
    "actual": "isolate_stem",
    "predicted": "remove_stem",
    "count": 2
  },
  {
    "actual": "cut_segment",
    "predicted": "trim_segment",
    "count": 1
  },
  {
    "actual": "cut_segment",
    "predicted": "remove_stem",
    "count": 1
  },
  {
    "actual": "cut_segment",
    "predicted": "none",
    "count": 1
  },
  {
    "actual": "trim_segment",
    "predicted": "cut_segment",
    "count": 1
  },
  {
    "actual": "denoise",
    "predicted": "remove_silence",
    "count": 1
  },
  {
    "actual": "speed",
    "predicted": "none",
    "count": 1
  }
]
```

_Handwritten subset: op micro F1 0.894, target micro F1 0.927._

## Worst errors

- `no more vocals` — gold ['remove_stem'] / ['vocals'] → predicted ['add_effect'] / ['vocals'] (scores {'add_effect': 0.646, 'remove_stem': 0.156, 'isolate_stem': 0.107})
- `i only want the singer` — gold ['isolate_stem'] / ['vocals'] → predicted ['isolate_stem'] / ['full_mix'] (scores {'isolate_stem': 0.308, 'trim_segment': 0.127, 'remove_speaker': 0.105})
- `strip everything except the bass` — gold ['isolate_stem'] / ['bass'] → predicted ['remove_stem'] / ['bass'] (scores {'remove_stem': 0.689, 'normalize': 0.074, 'isolate_stem': 0.065})
- `bring the vocals forward and drop the rest` — gold ['isolate_stem'] / ['vocals'] → predicted ['remove_stem'] / ['vocals'] (scores {'remove_stem': 0.616, 'gain': 0.246, 'add_effect': 0.09})
- `rip out the second chorus` — gold ['cut_segment'] / ['full_mix'] → predicted ['trim_segment'] / ['full_mix'] (scores {'trim_segment': 0.64, 'cut_segment': 0.201, 'fade': 0.105})
- `take the spoken intro off` — gold ['cut_segment'] / ['full_mix'] → predicted ['remove_stem'] / ['full_mix'] (scores {'remove_stem': 0.861, 'cut_segment': 0.304, 'remove_speaker': 0.145})
- `chop out the breakdown` — gold ['cut_segment'] / ['full_mix'] → predicted [] / ['full_mix'] (scores {'cut_segment': 0.117, 'remove_stem': 0.087, 'normalize': 0.067})
- `keep 0:10 through 0:50 and drop the rest` — gold ['trim_segment'] / ['full_mix'] → predicted ['cut_segment'] / ['full_mix'] (scores {'cut_segment': 0.311, 'remove_speaker': 0.277, 'trim_segment': 0.156})
- `make it start at 5 seconds and end at 25` — gold ['trim_segment'] / ['full_mix'] → predicted ['trim_segment', 'speed'] / ['full_mix'] (scores {'trim_segment': 0.646, 'speed': 0.419, 'remove_silence': 0.117})
- `drop the guest` — gold ['remove_speaker'] / ['speaker_2'] → predicted ['remove_speaker'] / ['speaker_1'] (scores {'remove_speaker': 0.981, 'remove_stem': 0.093, 'gain': 0.068})
- `i only need the host` — gold ['remove_speaker'] / ['speaker_2'] → predicted ['remove_speaker'] / ['speaker_1'] (scores {'remove_speaker': 0.733, 'isolate_stem': 0.256, 'trim_segment': 0.19})
- `keep speaker one, delete the other` — gold ['remove_speaker'] / ['speaker_2'] → predicted ['remove_speaker'] / ['speaker_1'] (scores {'remove_speaker': 0.985, 'cut_segment': 0.046, 'isolate_stem': 0.02})
- `remove the air conditioner drone` — gold ['denoise'] / ['full_mix'] → predicted ['remove_silence'] / ['full_mix'] (scores {'remove_silence': 0.45, 'remove_stem': 0.222, 'remove_speaker': 0.095})
- `double time` — gold ['speed'] / ['full_mix'] → predicted [] / ['full_mix'] (scores {'speed': 0.078, 'normalize': 0.076, 'cut_segment': 0.071})
- `pull the vocals down 3 db and add reverb to the drums` — gold ['gain', 'add_effect'] / ['vocals', 'drums'] → predicted ['isolate_stem', 'gain', 'add_effect'] / ['vocals', 'drums'] (scores {'add_effect': 0.95, 'gain': 0.807, 'isolate_stem': 0.76})

Error CSVs: `errors_rules_heldout.csv`, `errors_tfidf-logreg_heldout.csv`