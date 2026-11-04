# Evaluation — `test` split

Rows: **225**

## Comparison (all rows)

| Model | n | Op subset acc | Op micro F1 | Op macro F1 | Target micro F1 | Target macro F1 | Action exact | Amount MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | 225 | 0.831 | 0.881 | 0.885 | 0.913 | 0.718 | 0.742 | 0.811 |
| tfidf-logreg | 225 | 0.782 | 0.798 | 0.786 | 0.888 | 0.724 | 0.698 | 4.575 |

## Handwritten rows only (honest headline)

| Model | n | Op subset acc | Op micro F1 | Op macro F1 | Target micro F1 | Target macro F1 | Action exact | Amount MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rules | 45 | 0.844 | 0.889 | 0.891 | 0.913 | 0.718 | 0.756 | 0.75 |
| tfidf-logreg | 45 | 0.822 | 0.830 | 0.816 | 0.879 | 0.685 | 0.711 | 4.52 |

## Per-operation scores — tfidf-logreg

| Operation | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| remove_stem | 0.615 | 0.800 | 0.696 | 20 |
| isolate_stem | 0.800 | 1.000 | 0.889 | 20 |
| gain | 1.000 | 1.000 | 1.000 | 25 |
| fade | 1.000 | 1.000 | 1.000 | 20 |
| cut_segment | 0.364 | 0.600 | 0.453 | 20 |
| trim_segment | 0.167 | 0.050 | 0.077 | 20 |
| remove_speaker | 1.000 | 0.600 | 0.750 | 20 |
| denoise | 0.826 | 0.950 | 0.884 | 20 |
| normalize | 1.000 | 0.850 | 0.919 | 20 |
| speed | 1.000 | 1.000 | 1.000 | 20 |
| add_effect | 1.000 | 0.933 | 0.966 | 15 |
| remove_silence | 1.000 | 0.667 | 0.800 | 15 |

## Per-target scores — tfidf-logreg

| Target | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| full_mix | 0.858 | 0.958 | 0.905 | 120 |
| vocals | 1.000 | 0.800 | 0.889 | 25 |
| drums | 1.000 | 0.750 | 0.857 | 20 |
| bass | 1.000 | 1.000 | 1.000 | 15 |
| guitar | 1.000 | 1.000 | 1.000 | 10 |
| keys | 1.000 | 1.000 | 1.000 | 20 |
| other | 0.000 | 0.000 | 0.000 | 0 |
| speaker_1 | 0.667 | 0.200 | 0.308 | 10 |
| speaker_2 | 0.625 | 0.500 | 0.556 | 10 |

## Confusion matrix — tfidf-logreg (single-operation rows)

| actual \ predicted | gain | fade | remove_stem | denoise | isolate_stem | cut_segment | none | trim_segment | remove_speaker | normalize | speed | add_effect | remove_silence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gain | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fade | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| remove_stem | 0 | 0 | 16 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| denoise | 0 | 0 | 0 | 19 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| isolate_stem | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cut_segment | 0 | 0 | 1 | 0 | 0 | 9 | 1 | 4 | 0 | 0 | 0 | 0 | 0 |
| none | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| trim_segment | 0 | 0 | 0 | 0 | 5 | 14 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| remove_speaker | 0 | 0 | 7 | 0 | 0 | 1 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| normalize | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 0 | 0 | 0 |
| speed | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 |
| add_effect | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 14 | 0 |
| remove_silence | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |

## Amount errors (dB / seconds / factor / bpm)

```json
{
  "overall": {
    "n": 33,
    "mae": 4.575,
    "median_ae": 4.69
  },
  "by_operation_unit": {
    "fade|seconds": {
      "n": 5,
      "mae": 4.978,
      "median_ae": 5.02
    },
    "gain|dB": {
      "n": 23,
      "mae": 5.456,
      "median_ae": 4.69
    },
    "remove_silence|seconds": {
      "n": 5,
      "mae": 0.122,
      "median_ae": 0.12
    }
  }
}
```

## Top confusions

```json
[
  {
    "actual": "remove_stem",
    "predicted": "denoise",
    "count": 4
  },
  {
    "actual": "cut_segment",
    "predicted": "none",
    "count": 1
  },
  {
    "actual": "cut_segment",
    "predicted": "trim_segment",
    "count": 4
  },
  {
    "actual": "cut_segment",
    "predicted": "remove_stem",
    "count": 1
  },
  {
    "actual": "trim_segment",
    "predicted": "cut_segment",
    "count": 14
  },
  {
    "actual": "trim_segment",
    "predicted": "isolate_stem",
    "count": 5
  },
  {
    "actual": "remove_speaker",
    "predicted": "remove_stem",
    "count": 7
  },
  {
    "actual": "remove_speaker",
    "predicted": "cut_segment",
    "count": 1
  },
  {
    "actual": "denoise",
    "predicted": "cut_segment",
    "count": 1
  },
  {
    "actual": "normalize",
    "predicted": "remove_stem",
    "count": 2
  },
  {
    "actual": "add_effect",
    "predicted": "trim_segment",
    "count": 1
  },
  {
    "actual": "remove_silence",
    "predicted": "cut_segment",
    "count": 5
  }
]
```

_Handwritten subset: op micro F1 0.830, target micro F1 0.879._

## Worst errors

- `remove the background vocals` — gold ['remove_stem'] / ['vocals'] → predicted ['denoise'] / ['vocals'] (scores {'denoise': 0.534, 'remove_stem': 0.429, 'gain': 0.083})
- `take out the background vocals` — gold ['remove_stem'] / ['vocals'] → predicted ['denoise'] / ['vocals'] (scores {'denoise': 0.59, 'remove_stem': 0.287, 'gain': 0.055})
- `i need you to remove the background vocals thanks` — gold ['remove_stem'] / ['vocals'] → predicted ['denoise'] / ['vocals'] (scores {'remove_stem': 0.472, 'denoise': 0.284, 'gain': 0.093})
- `go ahead and strip the background vocals` — gold ['remove_stem'] / ['vocals'] → predicted ['denoise'] / ['vocals'] (scores {'remove_stem': 0.492, 'denoise': 0.468, 'add_effect': 0.073})
- `go ahead and delete everything between 1:00 and 1:10` — gold ['cut_segment'] / ['full_mix'] → predicted [] / ['full_mix'] (scores {'remove_speaker': 0.207, 'cut_segment': 0.141, 'trim_segment': 0.078})
- `remove the last 10 seconds` — gold ['cut_segment'] / ['full_mix'] → predicted ['trim_segment'] / ['full_mix'] (scores {'trim_segment': 0.33, 'remove_stem': 0.09, 'fade': 0.082})
- `get rid of the last 10 seconds` — gold ['cut_segment'] / ['full_mix'] → predicted ['trim_segment'] / ['full_mix'] (scores {'trim_segment': 0.313, 'fade': 0.078, 'denoise': 0.071})
- `delete the last 10 seconds` — gold ['cut_segment'] / ['full_mix'] → predicted ['trim_segment'] / ['full_mix'] (scores {'trim_segment': 0.32, 'cut_segment': 0.182, 'fade': 0.08})
- `i'd like to get rid of the last 10 seconds` — gold ['cut_segment'] / ['full_mix'] → predicted ['trim_segment'] / ['full_mix'] (scores {'trim_segment': 0.396, 'fade': 0.074, 'cut_segment': 0.068})
- `go ahead and remove the last 10s` — gold ['cut_segment'] / ['full_mix'] → predicted ['remove_stem'] / ['full_mix'] (scores {'remove_stem': 0.46, 'remove_speaker': 0.092, 'denoise': 0.08})
- `keep only from 0:05 to 0:20` — gold ['trim_segment'] / ['full_mix'] → predicted ['cut_segment'] / ['full_mix'] (scores {'cut_segment': 0.899, 'isolate_stem': 0.238, 'trim_segment': 0.23})
- `just keep from 0:05 to 0:20` — gold ['trim_segment'] / ['full_mix'] → predicted ['cut_segment'] / ['full_mix'] (scores {'cut_segment': 0.898, 'trim_segment': 0.28, 'isolate_stem': 0.231})
- `trim to 0:05 to 0:20` — gold ['trim_segment'] / ['full_mix'] → predicted ['cut_segment'] / ['full_mix'] (scores {'cut_segment': 0.871, 'trim_segment': 0.672, 'normalize': 0.113})
- `keep only from 0:05 to 0:20 please` — gold ['trim_segment'] / ['full_mix'] → predicted ['cut_segment'] / ['full_mix'] (scores {'cut_segment': 0.876, 'isolate_stem': 0.273, 'trim_segment': 0.189})
- `just keep from 0:05 to 0:20 when you can` — gold ['trim_segment'] / ['full_mix'] → predicted ['cut_segment'] / ['full_mix'] (scores {'cut_segment': 0.859, 'isolate_stem': 0.3, 'trim_segment': 0.196})

Error CSVs: `errors_rules_test.csv`, `errors_tfidf-logreg_test.csv`