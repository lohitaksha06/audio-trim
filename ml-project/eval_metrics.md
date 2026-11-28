# Evaluation metrics

| Model | Task | Metric |
| --- | --- | --- |
| Diagnoser | Regression | SNR error (dB), clipped-% error, clipping F1 |
| Denoiser | Speech enhancement | SI-SDR improvement, STOI, PESQ |
| Declipper | Restoration | SI-SDR improvement vs cubic-spline baseline |
| Router | Pipeline choice | Agreement with the oracle |

- **SI-SDR (scale-invariant signal-to-distortion ratio):** how much of the
  output is the wanted signal versus the error, ignoring overall volume. Higher
  is better. Reported as the *improvement* over the unprocessed input.
- **STOI (short-time objective intelligibility):** 0–1 estimate of how much
  speech a listener would still understand. Higher is better.
- **PESQ:** perceptual quality estimate, roughly 1–4.5. Higher is better. Needs
  a C build, so it may have to run on Colab rather than Windows.
- **SNR error:** |predicted SNR − true SNR| in dB. Lower is better. True SNR is
  exact because we built the mixture.
- **Clipped-% error:** |predicted − true| percentage of clipped samples.
  Lower is better.
- **Clipping F1:** whether the model finds the clipped regions at all. Precision
  and recall are reported too, since a model that never predicts clipping can
  look accurate on a mostly-unclipped test set.
- **Oracle agreement:** fraction of clips where the router picks the pipeline
  that the exhaustive search found best.

## Baselines that must be reported

1. No repair.
2. Spectral gating (the shipped `/features/clean` chain).
3. Cubic-spline declipping.
4. The trained model.

A metric without its baseline is not a result.
