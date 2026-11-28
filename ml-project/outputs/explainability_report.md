# Explainability - diagnoser

- model: outputs\models\diagnoser.pt (154978 params)
- inputs: 4 clips, log-mel 64 x 175
- method: |d SNR_head / d input log-mel|, averaged over clips

## Predictions on the explained clips
| clip | pred SNR (dB) | true SNR (dB) | pred clipped | true clipped |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5.50 | 5.17 | 0.0045 | 0.0024 |
| 1 | 23.75 | 27.06 | 0.0033 | 0.0114 |
| 2 | 4.32 | 5.72 | 0.0033 | 0.0050 |
| 3 | 19.72 | 19.15 | 0.0472 | 0.0650 |

## Most influential mel bands
- mel  1 (   28-   86 Hz): 3.5% of saliency
- mel  4 (  117-  183 Hz): 2.7% of saliency
- mel  5 (  150-  218 Hz): 2.7% of saliency
- mel  3 (   86-  150 Hz): 2.5% of saliency
- mel  0 (    0-   56 Hz): 2.4% of saliency

## Most influential time regions
- frame 132 (~1.32s): 0.9% of saliency
- frame 71 (~0.71s): 0.8% of saliency
- frame 74 (~0.74s): 0.8% of saliency
- frame 42 (~0.42s): 0.8% of saliency
- frame 81 (~0.81s): 0.8% of saliency

## Concentration and band groups
- top 10% of time-frequency cells hold 23.2% of total saliency
- low (<500 Hz): 29.7%; mid (500-2000 Hz): 32.1%; high (>=2000 Hz): 38.3%

## LIME
- tabular LIME on an 8x8 pooled log-mel grid (regression surrogate for the SNR head)
- mel 0 (0-56 Hz), time ~1.53s: weight +0.975
- mel 0 (0-56 Hz), time ~0.22s: weight -0.770
- mel 0 (0-56 Hz), time ~1.09s: weight +0.711
- mel 1 (27-86 Hz), time ~0.44s: weight -0.683
- mel 0 (0-56 Hz), time ~0.44s: weight -0.680
- mel 1 (27-86 Hz), time ~0.22s: weight -0.663
- 0.99 < mel0_t0 <= 1.29: weight -0.660
- mel 1 (27-86 Hz), time ~1.31s: weight -0.546
- mel 1 (27-86 Hz), time ~0.66s: weight -0.530
- mel 2 (56-117 Hz), time ~1.31s: weight -0.426

## SHAP
- SHAP not installed - optional: pip install shap

## Figure
- outputs\figures\explain_saliency.png

