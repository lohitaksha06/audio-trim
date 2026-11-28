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
