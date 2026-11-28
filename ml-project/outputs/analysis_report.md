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
      8     5.3716    -0.9093    27.66    False
      9     6.8596   -11.1325    28.20     True

## Figures
- outputs\figures\analysis_pca_clip.png
- outputs\figures\analysis_pca_snr.png
- outputs\figures\analysis_tsne_clip.png
- outputs\figures\analysis_tsne_snr.png

