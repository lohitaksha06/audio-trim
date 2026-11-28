# Audelle — ML

This folder holds the machine-learning part of Audelle.

## Models

| Task | Model | Purpose |
| --- | --- | --- |
| Stem separation | Demucs `htdemucs` | Split a mix into vocals, drums, bass and other |
| Audio cleaning | DeepFilterNet3 | Remove noise and reverb from speech |
| Instrument classification | PANNs CNN14 | Tag which instruments are present |

## Datasets

- **MUSDB18-HQ** — <https://sigsep.github.io/datasets/musdb.html>
- **DNS Challenge** — <https://github.com/microsoft/DNS-Challenge>
- **NSynth / AudioSet** — instrument and sound-event data

See `datasets.md` for sizes, licenses and subsets.

## Scope

For the final project we use **pretrained models** and **small evaluation
subsets**. We do not train the models from scratch, and we do not download the
full datasets. This keeps the work reproducible on a normal laptop.
