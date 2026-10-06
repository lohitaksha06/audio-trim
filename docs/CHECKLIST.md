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
      - [x] `server/ml/diarization/transcript.py` — speaker segments + per-segment
            Whisper text, real talk time per speaker
      - [x] `SPLIT_SPEAKERS` / `KEEP_SPEAKER` / `REMOVE_SPEAKER` / `CHAPTERS` intents
      - [x] Podcast page shows real detected speakers and chapter counts
      Bugs found by measuring, not by reading:
      - VAD thresholded `rms > percentile(rms, 40) * 1.5`, which excluded real
        speech whenever speech was most of the file → **zero** segments. Now an
        adaptive log-domain noise floor with hangover.
      - Cluster selection required `len(segments) >= k + 2`, so a 3-turn clip
        could never resolve 2 speakers.
      - `remove_spans` skipped any span starting at sample 0, so leading audio
        survived every "keep only speaker 1".
      - "remove background noise from her **voice**" matched the speaker pattern;
        keep/remove now require the literal word "speaker".
- [x] **#2 Trained instrument synthesis** — replaces hand-written presets
      - [x] Data: `confit/nsynth-parquet`, one `instrument/` shard (379 MB,
            3,856 notes). Fields: audio + instrument + pitch + filename-encoded
            velocity. 27 family/source combos (11 bases x acoustic/electronic/synthetic).
      - [x] Features per note: 32-band log spectral envelope over the sustain,
            plus attack, decay ratio, centroid, flatness.
      - [x] Model: `TimbreNet` — instrument embedding(24) + [pitch, velocity] ->
            128 -> 128 -> 36. **Val loss 0.946 -> 0.685**, CPU, ~30 s.
      - [x] Renderer: partial amplitudes read off the predicted envelope, with
            inharmonicity, a flatness-driven noise layer and predicted ADSR.
      - [x] `server/ml/synthesis/timbre.py` + `scripts/train_timbre_model.py`
      - [x] Wired into `_add_instrument`; notes follow the track's detected key.
      Honest scope: the **timbre is learned**, the renderer is deterministic DSP —
      the same hybrid design DDSP and NSF models use. Not a generative model.
      - [x] Unknown instruments return `None` instead of fuzzy-matching (that
            fallback mapped "acid" -> `vocal/acoustic`, same trap as the old
            `phonk`->`honk` bug) and fall back to the existing DSP synths.
- [x] **Electronic / beat expansion**
      - [x] `airasoul/drum-kit` -> **1,200 real recorded one-shots**, 10 drum types
            (kick, snare, clap, hat, cymbal, crash, ride, tom, conga, rim),
            trimmed and de-clicked. FLAC: 60 MB -> 34.7 MB committed.
      - [x] All **18 existing grooves** now play recorded hits instead of a
            pitch-swept sine for the kick and filtered noise for the snare.
      - [x] `server/ml/synthesis/drums.py` — 10-style 16th-note sequencer,
            tempo-locked to measured BPM. Measured: kick-band onset energy is
            **1.97x higher on-grid than off-grid**.
      - [x] UI catalogue (`instrumentCatalog.ts`): 11 trained instruments,
            4 honestly-labelled DSP sounds, 10 beat styles, 10 drum types.
      - [x] `test_catalog_sync.py` fails if the UI advertises anything the
            backend cannot render, or omits anything it can.
      Honest scope: hits are real recordings; the *arrangement* is a rule-based
      pattern library. No trained drum model is claimed.

## Known fakes still in the product

- [ ] **Style presets are DSP**, not generative - "convert to phonk style"
      applies EQ plus a drum pattern.
- [ ] **MusicGen never benchmarked.** No GPU here (torch `2.10.0+cpu`,
      Ryzen 7 7840HS, 15.3 GB). Must be a bounded benchmark before it ships.
- [ ] **Groove classifier trained on synthetic loops**, not real music, which
      degrades "add drums that match your song". Retrain with
      `train_groove_classifier.py --songs-dir <root>`.
- [ ] **Audio→video muxing unconfirmed.** `video_extractor.py` handles extract.

## Inpainting (done — was a crossfade)

`server/ml/inpainting/inpainter.py`. "remove that cymbal crash and fill smoothly"
used to splice the samples out and crossfade the ends. That is a delete, not a
fill: the file came back shorter and the audio was still discontinuous.

- [x] `mode="fill"` (default) keeps the timeline length and rebuilds the hole
- [x] Strategy 1: bar-aligned repeat, using the existing beat detector, since the
      right material for a ruined bar is the same musical position 4/8 bars back
- [x] Strategy 2: log-mel feature search for similar material
- [x] Fallback: spectral fill (log-magnitude interpolation + phase propagation)
- [x] `mode="close"` keeps the old splice-out when explicitly asked for
- [x] Provenance reported: strategy, source offset, match distance — a fallback is
      never presented as a successful repeat

Measured: duration preserved (8.00 s vs 7.50 s), filled region non-silent.

**Not claimed:** recovery of the original notes. On through-composed material
those notes no longer exist anywhere in the file.

**Not claimed:** seam smoothness as an improvement. A 50 ms equal-gain crossfade
can post a smaller single-sample step than the reconstructed splice, so
`test_inpaint.py` asserts a bound rather than superiority.

An intermediate design that minimised seam continuity as its refinement
objective was **discarded after measurement** — that objective is minimised at
any quiet point, so the matcher drifted away from the phase-correct repeat.

## Rebuilding the trained assets

    python -m scripts.train_timbre_model --epochs 80   # ~4 min, CPU
    python -m scripts.build_drum_bank --per-label 120   # ~1 min

Both read Hugging Face datasets (`confit/nsynth-parquet`, `airasoul/drum-kit`).
Their outputs — `server/ml/models/timbre_model.pt` and
`server/ml/models/drum_bank/` — are committed so the app works out of the box.

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