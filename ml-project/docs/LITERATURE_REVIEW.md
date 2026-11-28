# Literature review

Scope: speech enhancement, the VoiceBank-DEMAND benchmark, DEMAND noise,
audio declipping, the metrics we report, and the explainability methods used in
`scripts/explain.py`. The review is deliberately short. It ends with the
objective of this project and what is and is not claimed.

No DOI is invented. Where a bibliographic detail (page range, volume) could not
be checked against the publisher record it is omitted or marked **[confirm]**.

## 1. Classical speech enhancement

Spectral subtraction (Boll, 1979) estimates the noise magnitude during
non-speech frames and subtracts it from the noisy magnitude spectrum. It is
cheap and still useful as a baseline, but it leaves "musical noise" and depends
on a voice-activity detector. Wiener filtering (Lim & Oppenheim, 1979) replaces
hard subtraction with a gain derived from the a-priori and a-posteriori SNRs,
which is smoother but needs an estimate of the noise spectrum.

Statistical estimators put the problem on a cost function. Ephraim and Malah
(1984) derived the minimum mean-square error short-time spectral amplitude
(MMSE-STSA) estimator under a Gaussian model; the log-MMSE variant followed in
1985. The decision-directed a-priori SNR estimator of Scalart and Filho (1996)
is the ingredient that made these methods practical, and it remains a strong
DSP reference point. All of these methods assume the noise is stationary enough
to estimate from pauses, which is exactly where learned models help.

## 2. Learned speech enhancement

Xu, Du, Dai and Lee (2014) framed enhancement as DNN regression from noisy to
clean log-power spectra and showed a large gain over classical estimators on
seen and unseen noise. Park and Lee (2017) used a fully convolutional network
to predict a time-frequency mask, which preserves phase and generalises across
input lengths. SEGAN (Pascual, Bonafonte & Serra, 2017) added an adversarial
and perceptual objective to a fully convolutional generator, improving the
perceived quality of the output rather than only the squared error.

These three papers define the design space we borrow from: regression versus
masking, and a plain loss versus an adversarial/perceptual loss. We do not
reproduce their architectures or their numbers; our models are intentionally
small and trained from scratch.

## 3. VoiceBank-DEMAND

The benchmark comes from Valentini-Botinhao, Wang, Takaki and Yamagishi
(2016), "Investigating RNN-based speech enhancement methods for noise-robust
Text-to-Speech", 9th ISCA Speech Synthesis Workshop (SSW9). The corpus pairs
clean VoiceBank speech with DEMAND noise at four SNRs: 0, 5, 10 and 15 dB. The
training set has 28 speakers (11,572 utterances); the test set has two unseen
speakers (824 utterances), and the test noises do not appear in training. The
original release is 48 kHz; the widely used 16 kHz version is a downstream
resample.

Two protocol details matter for us:

- The split is **speaker-disjoint**. Splitting by utterance leaks a speaker
  into both sides and inflates every score. Our manifest follows the same rule.
- The four fixed SNRs make published PESQ/STOI numbers comparable to each
  other, but not automatically to a system trained on continuous SNRs.

The dataset record is Valentini-Botinhao (2017), University of Edinburgh
DataShare, DOI 10.7488/ds/2117, CC BY 4.0. That DOI is confirmed in
`datasets.md`. Page numbers for the SSW9 paper are **[confirm]**.

## 4. DEMAND noise

Thiemann, Ito and Vincent (2013), "The Diverse Environments Multi-channel
Acoustic Noise Database (DEMAND): A database of multichannel environmental
noise recordings", Proceedings of Meetings on Acoustics (POMA), introduced 18
environments recorded with a 16-channel array at 48 kHz. Sixteen kHz builds are
distributed separately; the Zenodo record is 10.5281/zenodo.1227121 (confirmed
in `datasets.md`).

We use three environments (office hallway, domestic living room, office
meeting) and mix at a continuous SNR drawn from 0-30 dB rather than at four
levels. The reason is specific to the diagnoser: a regressor trained on four
SNR values can memorise the levels instead of learning a noise-to-speech ratio.
The cost is that our numbers are not directly comparable to the benchmark's
four-level protocol, and we say so in `outputs/RESULTS.md`.

## 5. Audio declipping

Hard clipping maps every sample above a threshold to the threshold, so the
waveform shape is lost exactly where the signal is loudest. The cheap baseline
is interpolation: fit a cubic spline through the unclipped samples around each
clipped run and replace the flat top. It is fast and needs no training, and it
is the baseline our DSP chain already ships.

Sparse and co-sparse formulations treat declipping as a constrained recovery
problem. A-SPADE and related consistent iterative hard-thresholding methods
(Kitic, Jacques, Madhu, Hopwood & Fuchs, 2015 **[confirm: exact author list and
pages]**) alternate between enforcing consistency with the clipped observations
and sparsity in a transform domain. They beat spline interpolation on synthetic
clipping but are iterative and slower than a feed-forward model.

DNN declippers learn the mapping from clipped to clean audio directly, usually
with a U-Net or a mask predictor and an L1/L2 waveform loss. The survey by
Zaviska, Rajmic, Ozerov and Rencker (2021), "A survey and an extensive
evaluation of popular audio declipping methods", IEEE Journal of Selected
Topics in Signal Processing **[confirm: volume and pages]**, compares the
families on common metrics and is the reference we follow when choosing
baselines. Our declipper is a small U-Net trained from scratch; the cubic
spline is the mandatory baseline, not a straw man.

## 6. Metrics

- **SI-SDR** (Le Roux, Wisdom, Erdogan & Hershey, 2019, ICASSP) measures the
  ratio of the projected target to the residual, invariant to scale. We report
  the improvement over the unprocessed input.
- **STOI** (Taal, Hendriks, Heusdens & Jensen, 2011, IEEE TASLP) predicts
  short-time intelligibility in [0, 1].
- **PESQ** (Rix, Beerends, Hollier & Hekstra, 2001, ICASSP) predicts perceptual
  quality, roughly 1-4.5. It needs a native build, so it may run on Colab rather
  than Windows.
- **Diagnoser metrics** are task-specific: MAE in dB for SNR, MAE for the
  clipped fraction, and precision/recall/F1 for "is this clip clipped?" at a
  0.001 fraction threshold. A model that never predicts clipping can look
  accurate on an imbalanced set, so F1 is reported next to accuracy.

## 7. Explainability and dimensionality

LIME (Ribeiro, Singh & Guestrin, 2016, KDD) fits a local interpretable
surrogate around one prediction by perturbing the input and weighting the
samples by proximity. SHAP (Lundberg & Lee, 2017, NeurIPS) assigns Shapley
values from a cooperative-game formulation. Both are optional dependencies in
this project. The guaranteed method is gradient saliency (Simonyan, Vedaldi &
Zisserman, 2014, ICLR workshop **[confirm: workshop track]**): backpropagate the
output of interest to the input and inspect the absolute gradient. It needs no
package and is honest about being a local linear approximation.

For the data analysis we use PCA (Jolliffe, 2002) and t-SNE (van der Maaten &
Hinton, 2008, JMLR). PCA is implemented with `numpy.linalg.eigh`; t-SNE is only
used if scikit-learn is installed, and the analysis falls back to the 2D PCA
projection when it is not. We do not hand-roll t-SNE.

## 8. Our objective

Train, from scratch and with small models, four components:

1. a **diagnoser** that predicts SNR (dB) and clipped fraction from log-mel,
2. a **denoiser** (conv-GRU mask) that removes stationary noise,
3. a **declipper** (small U-Net) that restores flat-topped peaks,
4. a **router** that chooses the repair chain from the diagnosis.

Evaluate every component against explicit baselines: no repair, the shipped
spectral-gating chain, and cubic-spline declipping. Use speaker-disjoint
splits, exact labels built by construction, and report the seen-vs-unseen gap
rather than a single aggregate score.

What this project does **not** claim: state-of-the-art enhancement, parity with
published VoiceBank-DEMAND numbers, or a perceptual study. The contribution is
a small, reproducible, honestly measured pipeline whose first stage (the
diagnoser) is analysed for dimensionality, correlation, class separability and
explainability. The limitations are listed in `outputs/RESULTS.md`.

## References

- Boll, S. F. (1979). Suppression of acoustic noise in speech using spectral
  subtraction. *IEEE Transactions on Acoustics, Speech, and Signal Processing*,
  27(2).
- Ephraim, Y., & Malah, D. (1984). Speech enhancement using a minimum
  mean-square error short-time spectral amplitude estimator. *IEEE Transactions
  on Acoustics, Speech, and Signal Processing*, 32(6).
- Ephraim, Y., & Malah, D. (1985). Speech enhancement using a minimum
  mean-square error log-spectral amplitude estimator. *IEEE Transactions on
  Acoustics, Speech, and Signal Processing*, 33(2).
- Kitic, S., Jacques, L., Madhu, N., Hopwood, P. W., & Fuchs, A. (2015).
  Consistent iterative hard thresholding for signal declipping. *ICASSP 2015*.
  **[confirm: exact author list and pages]**
- Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. (2019). SDR -
  half-baked or well done? *ICASSP 2019*.
- Lim, J. S., & Oppenheim, A. V. (1979). Enhancement and bandwidth compression
  of noisy speech. *Proceedings of the IEEE*, 67(12).
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. *NeurIPS 2017*.
- Park, S. R., & Lee, J. (2017). A fully convolutional neural network for
  speech enhancement. *Interspeech 2017*.
- Pascual, S., Bonafonte, A., & Serra, J. (2017). SEGAN: Speech Enhancement
  Generative Adversarial Network. *Interspeech 2017*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?"
  Explaining the predictions of any classifier. *KDD 2016*.
- Rix, A. W., Beerends, J. G., Hollier, M. P., & Hekstra, A. P. (2001).
  Perceptual evaluation of speech quality (PESQ). *ICASSP 2001*.
- Scalart, P., & Filho, J. V. (1996). Speech enhancement based on a priori
  signal to noise estimation. *ICASSP 1996*.
- Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Deep inside convolutional
  networks: Visualising image classification models and saliency maps.
  *ICLR 2014 workshop*. **[confirm: workshop track]**
- Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). An
  algorithm for intelligibility prediction of time-frequency weighted noisy
  speech. *IEEE Transactions on Audio, Speech, and Language Processing*, 19(7).
- Thiemann, J., Ito, N., & Vincent, E. (2013). The Diverse Environments
  Multi-channel Acoustic Noise Database (DEMAND). *Proceedings of Meetings on
  Acoustics*, 19.
- Valentini-Botinhao, C., Wang, X., Takaki, S., & Yamagishi, J. (2016).
  Investigating RNN-based speech enhancement methods for noise-robust
  Text-to-Speech. *9th ISCA Speech Synthesis Workshop (SSW9)*.
  **[confirm: page range]**
- Valentini-Botinhao, C. (2017). Noisy speech database for training speech
  enhancement algorithms and TTS models. University of Edinburgh DataShare.
  DOI 10.7488/ds/2117.
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE.
  *Journal of Machine Learning Research*, 9.
- Xu, Y., Du, J., Dai, L.-R., & Lee, C.-H. (2014). A regression approach to
  speech enhancement based on deep neural networks. *IEEE/ACM Transactions on
  Audio, Speech, and Language Processing*, 23(1). **[confirm: year/volume]**
- Zaviska, P., Rajmic, P., Ozerov, A., & Rencker, L. (2021). A survey and an
  extensive evaluation of popular audio declipping methods. *IEEE Journal of
  Selected Topics in Signal Processing*. **[confirm: volume and pages]**
