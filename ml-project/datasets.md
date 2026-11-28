# Datasets

## MUSDB18-HQ

- **URL:** <https://sigsep.github.io/datasets/musdb.html>
- **License:** Research use only (see the dataset page for the full terms)
- **Full size:** ~30 GB (44.1 kHz uncompressed stems)
- **Subset for this project:** the 7-second test clips, or 2–3 full test tracks
- **What the model learns:** separating a stereo mix into vocals, drums, bass
  and other. Demucs `htdemucs` is evaluated on these stems.

## DNS Challenge (Deep Noise Suppression)

- **URL:** <https://github.com/microsoft/DNS-Challenge>
- **License:** CC BY 4.0 for the synthetic data (check per release)
- **Full size:** hundreds of GB across releases
- **Subset for this project:** a handful of clean/noisy pairs from the
  `dev_testset` (tens of MB)
- **What the model learns:** removing background noise and reverb from speech.
  DeepFilterNet3 is evaluated on these pairs.

## NSynth / AudioSet

- **URL:** <https://magenta.tensorflow.org/datasets/nsynth> and
  <https://research.google.com/audioset/>
- **License:** CC BY 4.0 (NSynth), CC BY 4.0 (AudioSet labels; audio via YouTube)
- **Full size:** NSynth ~30 GB, AudioSet audio is very large (labels are small)
- **Subset for this project:** a few hundred NSynth test notes, or AudioSet
  label metadata only
- **What the model learns:** mapping audio to instrument / sound-event labels.
  PANNs CNN14 is evaluated on this labelled audio.

## Note

We only need enough audio to sanity-check the pretrained models. Do not download
the full datasets for the final project.
