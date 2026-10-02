# Audelle — "make it real" checklist

Rule for every item: **no fake data, no placebo buttons.** If a feature cannot do
what it claims, either make it work, or label it honestly in the UI.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done and measured · `[-]` dropped

---

## Done and measured

- [x] **Filler / hesitation removal** (`server/ml/transcription/fillers.py`)
      Acoustic detection + crossfade splice. Was calling `_remove_silence`.
      Measured 0 false positives on clean narration; cuts 0.45–0.5s hesitations.
- [x] **Voice clarity / background noise** (`server/ml/speech/voice.py`)
      Was routing speech through Demucs (music-trained) and measuring
      **−5 to −7 dB SNR**. Now Wiener denoise + noise-aware presence.
      Measured **+1.3 to +10.1 dB**, voice 1–3 dB *louder*, clean untouched.
- [x] **Prompt routing bugs**
      `phonk`→"honk" hijack, `duration` parsed out of "0.5 seconds",
      "cut pauses"→TRIM. Bench now 73/73, full suite green.
- [x] **Feature pages** — removed invented data (random stem bars, fake
      speakers, dead bitrate sliders). Now render real analysis.

## Queue

- [~] **#4 Speaker diarization → real podcast tools**
      `diarization.py` exists (VAD + MFCC + KMeans) but was never reachable
      from the prompt pipeline, which is why the podcast page was fake.
      - [~] Per-speaker transcript (run Whisper per diarized segment)
      - [ ] `SPLIT_SPEAKERS` intent → one file per speaker
      - [ ] `KEEP_SPEAKER` / `REMOVE_SPEAKER` intents
      - [ ] `CHAPTERS` intent → timestamped chapters from the transcript
      - [ ] Wire all of it into the podcast page with real output
- [ ] **#2 Trained instrument synthesis** — replace hand-written DSP presets
      - [ ] Acquire a labelled multi-instrument note corpus
      - [ ] Extract per-note timbre features (harmonic envelope, ADSR)
      - [ ] Train a timbre model (CPU-feasible; hybrid neural+render, as DDSP/NNSF do)
      - [ ] Render from the model; delete the sine/noise presets
- [ ] **Electronic / beat expansion** — user-facing priority
      - [ ] Real drum one-shots (recorded, not synthesized) + tempo sync
      - [ ] Train a style→hit selector instead of 24 hand-written patterns
      - [ ] Expand instrument coverage well past drums/bass

## Known fakes still in the product

- [ ] **Inpainting is a crossfade.** "Remove that cymbal crash and fill
      smoothly" reconstructs nothing. Needs a generative model.
- [ ] **`add drums` / `add bass` are hand-written sine+noise synthesis.**
      Being replaced under #2 above.
- [ ] **Style presets are DSP**, not generative — "convert to phonk style"
      applies EQ plus a drum pattern.
- [ ] **MusicGen never benchmarked.** No GPU here (torch `2.10.0+cpu`,
      Ryzen 7 7840HS, 15.3 GB). Must be a bounded benchmark before it ships.

## Environment constraints (measured, not assumed)

| | |
|---|---|
| torch | `2.10.0+cpu` — no CUDA |
| CPU / RAM | Ryzen 7 7840HS (mobile), 8C/16T / 15.3 GB |
| Installed | transformers 4.57.3, demucs 4.1.0, librosa 0.11, sklearn 1.8 |
| Missing | `diffusers`, `audiocraft`, `torchaudio` |
| Known broken | Whisper word timestamps (returns a plain tensor); HF ASR
  pipeline (Keras 3 / `tf_keras` conflict) |

Backend must be started with `--reload` during development — a plain
`uvicorn server.main:app` silently serves stale code and makes fixes look like
they did not work.