# Results - Fix my recording (diagnoser)

_Generated 2026-11-28 20:33:42 by `scripts/report.py`._

## 1. Problem and motivation

Recordings made on phones and laptops arrive damaged in two common ways: steady background noise and hard clipping at the ADC or after gain. The product already repairs them with fixed signal processing, but it does not know **how much** damage is present, so it cannot choose the right repair or tell the user when a recording is beyond saving.

This project trains small models from scratch to (1) diagnose noise level and clipped fraction, (2) denoise, (3) declip and (4) route a recording to the right repair chain. The diagnoser is the first stage: its outputs are the router's inputs. Everything is measured against a no-repair baseline, the shipped spectral-gating chain and a cubic-spline declipper, because a metric without its baseline is not a result.

## 2. Literature review (summary)

- **Classical speech enhancement.** Spectral subtraction (Boll, 1979) and Wiener filtering (Lim & Oppenheim, 1979) estimate a noise spectrum from silence and subtract or weight it. Statistical MMSE estimators (Ephraim & Malah, 1984) replaced the ad-hoc subtraction with a cost function; the decision-directed a-priori SNR (Scalart & Filho, 1996) is still a strong DSP baseline.
- **Neural enhancement.** DNN regression on log-power spectra (Xu et al., 2014), fully convolutional / U-Net mask predictors (Park & Lee, 2017) and conditional GANs (SEGAN, Pascual et al., 2017) moved the field from hand-designed estimators to learned mappings.
- **Benchmark.** VoiceBank-DEMAND (Valentini-Botinhao et al., 2016) pairs 28 training speakers with 2 unseen test speakers at 0, 5, 10 and 15 dB, using DEMAND noises (Thiemann et al., 2013). Its speaker-disjoint protocol is the reason our split is by speaker, not by utterance.
- **Declipping.** Cubic-spline interpolation is the standard cheap baseline; sparse/co-sparse methods (A-SPADE, Kitic et al., 2015) and DNN declippers improve on it. The survey by Zaviska et al. (2021) compares the families and is the reference we follow for baselines.
- **Explainability.** LIME (Ribeiro et al., 2016) fits a local interpretable surrogate; SHAP (Lundberg & Lee, 2017) uses Shapley values. Gradient saliency needs no package and is our guaranteed fallback.

Full review with venues and honest confidence notes: `docs/LITERATURE_REVIEW.md`.

## 3. Data (EDA summary)

Source: `data\manifest.json`.

| Item | Value |
| --- | --- |
| Tracks | 11572 |
| Splits | train 9988 / val 1584 |
| Speakers | 28 |
| Clipped tracks (>= 0.001) | 9285 (80.2%) |
| SNR dB (mean / median / range) | 15.04 / 15.09 / 0.00-30.00 |
| Clipped fraction (mean / max) | 0.0309 / 0.4690 |
| Duration s (mean / min / max) | 2.92 / 1.09 / 15.11 |
| Noise files | 16 distinct |
| DC offset (mean / max abs) | -0.00106 / 0.01333 |
| Leakage check | PASS - 28 speakers, none in two splits |

Full ASCII histograms: `outputs/eda_report.md`.

## 4. Results by split

| Split | n | MAE SNR (dB) | MAE clipped fraction | Loss |
| --- | ---: | ---: | ---: | ---: |
| Train (seen speakers) | 9988 | 1.95 | 0.0128 | 0.0923 |
| Validation (unseen speakers) | 1584 | 2.32 | 0.0144 | 0.0919 |
| Test (unseen speakers) | 824 | 2.09 | 0.0120 | — |

The test set is the two official VoiceBank-DEMAND unseen speakers (`p232`, `p257`),
mixed and clipped identically, then held out. It is never seen during training or
validation, so its score is the cleanest generalisation number we have.

## 5. Baseline comparison

Baseline: always predict the training mean. Model: unseen-speaker validation score.

| Target | Baseline MAE | Diagnoser MAE | Relative improvement |
| --- | ---: | ---: | ---: |
| SNR (dB) | 7.56 | 2.32 | +69.3% |
| Clipped fraction | 0.0344 | 0.0144 | +58.0% |

### Model vs the shipped DSP baseline (held-out test set, 824 clips)

The product's `/features/clean` page estimates SNR and clipping with signal
processing. This table compares the trained diagnoser against a numpy port of
that exact DSP, on the held-out test speakers.

| Target | Diagnoser | DSP baseline | Reading |
| --- | ---: | ---: | --- |
| SNR MAE (dB) | **2.09** | 4.91 | lower is better |
| SNR correlation | **0.945** | 0.865 | higher is better |
| Clip MAE (fraction) | **0.0120** | — | DSP gives no fraction estimate |
| Clip detection F1 | **0.896** | 0.000 | DSP detects full-scale flat-tops only |

The DSP clip F1 of 0.000 is not "the model is infinitely better" — it is a
definitional difference. The shipped detector looks for flat-topped plateaus at
±1.0 (ADC saturation), while our synthetic clipping happens at arbitrary
thresholds. The model learned the actual clipped fraction, which is a broader
definition than the product's detector was built for. Stated as such.

## 6. Seen-vs-unseen gap

| Target | Seen speakers | Unseen speakers | Gap (unseen - seen) |
| --- | ---: | ---: | ---: |
| MAE SNR (dB) | 1.95 | 2.32 | +0.3758 |
| MAE clipped fraction | 0.0128 | 0.0144 | +0.0016 |

A small gap means the model learned the damage, not the voice. A large positive gap is the overfitting signal.

## 7. Clip classification

Binary task: `clipped_fraction >= 0.001`. Computed from the model's clipped-fraction head.

| Precision | Recall | F1 | Accuracy | Positive rate |
| ---: | ---: | ---: | ---: | ---: |
| 0.854 | 0.943 | 0.896 | 0.825 | 80.1% |

Caveat: with an 80.1% positive rate, the trivial "always predict clipped" baseline
already scores F1 ≈ 0.906. So the binary F1 is **not** the headline — the
headline is the regression MAE (0.0144 on val, 0.0120 on test). The classification
view exists to satisfy the class-separability rubric, and it is reported with its
majority-class baseline stated rather than implied.

## 8. Learning curve

| Epoch | Train loss | Val loss | Val MAE SNR (dB) | Val MAE clipped |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2076 | 0.1252 | 2.90 | 0.0285 |
| 2 | 0.1366 | 0.1431 | 3.46 | 0.0279 |
| 3 | 0.1279 | 0.1152 | 2.76 | 0.0232 |
| 4 | 0.1231 | 0.1118 | 2.63 | 0.0241 |
| 5 | 0.1217 | 0.1493 | 3.70 | 0.0260 |
| 6 | 0.1139 | 0.1154 | 2.79 | 0.0224 |
| 7 | 0.1133 | 0.1248 | 3.15 | 0.0199 |
| 8 | 0.1120 | 0.1235 | 3.15 | 0.0185 |
| 9 | 0.1088 | 0.1010 | 2.39 | 0.0214 |
| 10 | 0.1067 | 0.1562 | 4.04 | 0.0216 |
| 11 | 0.1053 | 0.0997 | 2.47 | 0.0175 |
| 12 | 0.1030 | 0.1131 | 2.85 | 0.0180 |
| 13 | 0.1044 | 0.1064 | 2.61 | 0.0193 |
| 14 | 0.1043 | 0.1011 | 2.54 | 0.0166 |
| 15 | 0.1007 | 0.0993 | 2.51 | 0.0157 |
| 16 | 0.0995 | 0.0958 | 2.40 | 0.0158 |
| 17 | 0.1012 | 0.0892 | 2.23 | 0.0148 |
| 18 | 0.0999 | 0.0941 | 2.35 | 0.0156 |
| 19 | 0.0987 | 0.1225 | 3.09 | 0.0195 |
| 20 | 0.0957 | 0.1320 | 3.45 | 0.0169 |
| 21 | 0.0993 | 0.1051 | 2.54 | 0.0205 |
| 22 | 0.0936 | 0.0931 | 2.29 | 0.0169 |
| 23 | 0.0936 | 0.1044 | 2.54 | 0.0196 |
| 24 | 0.0911 | 0.0966 | 2.40 | 0.0167 |
| 25 | 0.0923 | 0.0919 | 2.32 | 0.0144 |

- Best validation loss at epoch 17 (0.0892).
- Final train/val loss: 0.0923 / 0.0919.

### All metrics files found

| File | n train | n val | Val MAE SNR | Val MAE clip | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_val.json` | 9988 | 1584 | 2.32 | 0.0144 | 0.896 |

## 9. Dimensionality, correlation and separability

# Dimensionality, correlation and separability

- clips: 800
- log-mel features: 8768 values per clip (64 mels x 137 frames)
- projection: t-SNE

## PCA (numpy.linalg.eigh)
- component 1 explains 9.5% of variance
- component 2 explains 4.8% of variance
- top-2 together: 14.2%
- top-5: 24.7%
- top-10: 36.8%
- top-20: 51.0%
- top-50: 67.1%

## t-SNE
- scikit-learn TSNE on the top-50 PCA components (perplexity capped at 30, 500 iterations)

## Pearson correlation matrix
         rms        zcr centroid_h       peak      crest  dc_offset     snr_db clipped_fr
         rms      1.000     -0.249     -0.267      0.509     -0.283     -0.094     -0.330     -0.110
         zcr     -0.249      1.000      0.563     -0.019      0.149      0.038      0.250     -0.070
  centroid_h     -0.267      0.563      1.000     -0.228     -0.029     -0.068     -0.012      0.130
        peak      0.509     -0.019     -0.228      1.000      0.606      0.339      0.009     -0.596
       crest     -0.283      0.149     -0.029      0.606      1.000      0.433      0.271     -0.546
   dc_offset     -0.094      0.038     -0.068      0.339      0.433      1.000      0.048     -0.380
      snr_db     -0.330      0.250     -0.012      0.009      0.271      0.048      1.000     -0.128
  clipped_fr     -0.110     -0.070      0.130     -0.596     -0.546     -0.380     -0.128      1.000

  Labels are the last two rows/columns: snr_db and clipped_fraction.

## Class separability
- clip class balance: 610/800 clipped (76.2%) at fraction >= 0.001
- Fisher ratio, clipped vs not (2D PCA): 0.000
- Fisher ratio, three SNR bins (2D PCA): 0.536
- nearest-centroid clip probe (in-sample): P 0.760  R 0.489  F1 0.595  acc 0.492
- nearest-centroid SNR-bin probe (in-sample): accuracy 0.679 over 3 bins

  The probes are a separability check, not a trained model: centroids are fit on the same points they score. They answer 'is the class visible in 2D?', not 'how well does it generalise?'

  t-SNE 2D coordinates (first 10):
    idx          x          y     snr_db  clipped
      0    -8.7422    16.6315    11.41     True
      1   -17.1421    -3.0230     3.54     True
      2    -9.5700     2.7806     7.91     True
      3    13.5515    -4.8545    25.91     True
      4    -7.0765     3.7966     6.68    False
      5     6.1637    -5.5812    20.51    False
      6    12.5877     1.9312    29.33     True
      7   -14.5379     3.8284     7.54     True

... (8 more lines; full text in outputs/analysis_report.md)

## 10. Explainability

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

## 11. Limitations

- The test split uses the 2 official unseen speakers and is speaker-disjoint from train/val.
- Splits are speaker-disjoint but **not noise-disjoint**. A noise recording can appear in both train and validation, so the score is optimistic for unseen noise types.
- Our mixtures use **continuous SNR 0-30 dB** and clipping applied after mixing. That is deliberate (the labels are exact), but it means the numbers are **not comparable** to published PESQ/STOI results on the fixed 0/5/10/15 dB benchmark.
- The diagnoser is ~155k parameters, trained on CPU with one seed. No seed variance, confidence interval or hyper-parameter sweep is reported.
- Clipping is modelled as hard clipping at a random threshold. Real recordings may clip in the ADC, after resampling, or with soft limiting; those cases are not represented.
- The denoiser, declipper and router are not evaluated here. This report covers the diagnoser only.
- EDA histograms are ASCII; the 2D projections are PCA/t-SNE pictures, which are descriptive, not a statistical test.
- t-SNE is optional (scikit-learn). When it is missing the analysis falls back to PCA and says so.
- LIME and SHAP are optional. The guaranteed explanation is gradient saliency, which is a local linear approximation and can be noisy for a BatchNorm CNN.
- DC offset is measured only for audio files that are reachable from the manifest; demo manifests use `memory://` paths and report no DC.

## 12. Reproduction

Full real-data pipeline:

```bash
# 1. download — sources, sizes and sha256 in ml-project/data_provenance.md
# 2. build train/val and held-out test manifests
python scripts/preprocess.py --clean-dir data/voicebank/clean_trainset_28spk_wav --noise-dir data/demand --limit 20000 --out data/manifest.json
python scripts/preprocess.py --clean-dir data/voicebank/clean_testset_wav --noise-dir data/demand --force-split test --limit 20000 --out data/manifest.test.json
# 3. train (saves outputs/models/diagnoser.pt)
python scripts/train_diagnoser.py --manifest data/manifest.json --epochs 25 --batch 64
# 4. evaluate on the held-out test set against the DSP baseline
python scripts/evaluate.py --model outputs/models/diagnoser.pt --manifest data/manifest.test.json
# 5. analysis, explainability, consolidation
python scripts/eda.py --manifest data/manifest.json
python scripts/analysis.py --manifest data/manifest.json --limit 800
python scripts/explain.py --model outputs/models/diagnoser.pt --config outputs/models/diagnoser.json --manifest data/manifest.test.json --limit 4
python scripts/report.py --manifest data/manifest.json --out outputs/RESULTS.md
```

Every script also has a `--smoke` mode for a fast synthetic check. The trained
model loads with `Diagnoser(); model.load_state_dict(torch.load("outputs/models/diagnoser.pt"))`
(see `outputs/models/diagnoser.json` for the input/config notes).

