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
      **Correction (2026-10-10):** an earlier version of this file claimed
      "+1.3 to +10.1 dB". That figure came from `broadband_snr_db`, which
      compares the 90th-percentile frame to the 10th — a *dynamic-range* proxy,
      not SNR. SNR is not computable at inference because it needs a clean
      reference that does not exist. Checked against a known-clean signal on
      the repo's own fixture, the true SNR gain is **+0.28 dB (strength 1),
      +0.53 dB (2), +0.73 dB (3)**, while the proxy reported +0.60/+1.10/+1.52
      — roughly 2x optimistic. The metadata field was renamed
      `snr_gain_db` → `dynamic_range_gain_db` and the UI now reports the
      directly-measurable `noise_floor_drop_db` and `level_gain_db` instead of
      "X dB better". Measured: **voice +1.9 to +2.9 dB louder**, background
      floor down ~0.3–1.0 dB, clean input returned byte-identical.
      Honest scope: this is a gentle broadband denoise and presence lift for
      **stationary** noise (hiss, hum, fan). It is not a noise remover.
- [x] **Prompt routing bugs**
      `phonk`→"honk" hijack, `duration` parsed out of "0.5 seconds",
      "cut pauses"→TRIM. Bench now 99/99 with 57/57 parameter checks; full suite green.
- [x] **Pause threshold was ignored** — `REMOVE_SILENCE` never received
      `max_pause`, and `_remove_silence` concatenated every non-silent island.
      "Cut pauses over 0.5s" therefore removed all gaps. The prompt binding now
      covers `REMOVE_SILENCE`; the operation keeps gaps shorter than the
      threshold and an 80 ms margin at each cut. Regression test: a 2.0 s pause
      is removed at a 0.5 s threshold (**1.84 s cut**) and kept at 3 s.
- [x] **Feature pages** — removed invented data (random stem bars, fake
      speakers, dead bitrate sliders). Now render real analysis.
      Also removed the convert page's reference to a non-existent "High
      quality" sidebar control, corrected the film dialogue hint (it denoises,
      it does not isolate speech), and stated the real tempo behaviour. The Lab
      now releases its WebAudio graph when the file changes, so the knobs keep
      controlling the audio after a second upload.
- [x] **Tempo read real music an octave low** — autocorrelation alone reported
      **60 BPM** for both a 90 BPM and a 120 BPM Audionautix track. Added a
      comb-grid score and a broad tempo prior; measured **90.0 BPM** and
      **120.1 BPM**, and speech now returns no tempo instead of inventing one.
- [x] **Prompt-page info answers offered a download** — "What BPM is this?"
      rendered the unchanged audio as an edit. It now answers in the chat with
      no download and no edit counter, matching the feature pages.
- [x] **"Give me just the drums" was caught as an output follow-up** — the
      bare "give" in the follow-up matcher swallowed the separation request, so
      a Server button replied "No output yet" instead of explaining the backend.
      "give me just/only …" is now treated as an edit.
- [x] **Keep-last and empty edits** — "Keep the last 30 seconds" now trims the
      head rather than the tail; a file shorter than the window is kept whole;
      any edit that would remove every sample is rejected instead of writing a
      zero-length WAV.
- [x] **Browser-only honesty at the edges** — video files are rejected at the
      picker with a Server explanation, the unused output-format settings are
      hidden, and the home/guide/prompt copy states which operations run
      locally. A real Chromium pass clicked **88 buttons** across nine feature
      pages: 88 OK, zero console errors.
- [x] **Public labels** — removed “Browser edition”, “on-device”, “No account
      needed” and the per-page browser badges. Server-only actions are still
      marked **Server**; no capability was added to the UI by removing the
      labels.
- [x] **Backend deployment package** — added a CPU-only `Dockerfile`,
      `.dockerignore`, `server/.env.example`, `scripts/warm_models.py` and
      `docs/DEPLOY_BACKEND.md`. CORS now reads
      `AUDIDELLE_CORS_ORIGINS`. The backend is **not deployed yet**; the
      guide requires a host account and persistent volume.

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

- [ ] **Style presets are DSP**, not generative. Corrected 2026-10-10: this
      entry used to say "applies EQ plus a drum pattern", which *understated*
      the work. Each preset lays a beat-locked recorded-drum pattern, a
      key-aware synth voice and a sidechain pump under the ducked original, at
      the detected tempo and detected key root. That is a real stylistic
      treatment — but it is still a layered DSP treatment, not a conversion or
      a re-recording, so the metadata now carries `generative: false` and the UI
      says "treated as X" rather than implying a conversion.
      Making it genuinely generative needs a model this CPU box cannot run
      (see MusicGen below).
- [ ] **MusicGen never benchmarked.** No GPU here (torch `2.10.0+cpu`,
      Ryzen 7 7840HS, 15.3 GB). Must be a bounded benchmark before it ships.
- [ ] **Groove classifier trained on synthetic loops**, not real music, which
      degrades "add drums that match your song". Retrain with
      `train_groove_classifier.py --songs-dir <root>`.
      **Blocked, not forgotten:** the retrain needs a labelled folder of real
      songs per feel (`data/songs/<label>/`). No such corpus is in the repo and
      the script downloads nothing — it only reads what you point it at. Until
      real audio exists this cannot be closed honestly.
- [x] **Audio→video muxing.** Was "unconfirmed" — `video_extractor.py` only ever
      extracted. A video upload now carries its original through
      `source_video_path`, and any edit that produces audio is muxed back onto
      that video (`POST /api/process` → `video_download_key`, "Download result
      as video"). Video is **stream-copied** (`-c:v copy`), never re-encoded.
      Verified against a real generated clip: video stream preserved and the
      audio genuinely replaced (880 Hz edit over a 440 Hz source, measured on
      the demuxed result, not just a 200 response).
      *Trap recorded:* `-af apad` combined with `-c:v copy` makes ffmpeg **hang
      indefinitely**. Measured — no output after 45 s, versus 0.2 s with
      `-shortest` alone. Padding is unnecessary because a stream-copied video
      already fixes the duration. `test_mux_does_not_hang` guards this.
      Muxing failure never fails the whole edit; it is reported as
      `video_mux_error` and the audio download still works.

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

## Browser-only mode (no backend)

`website/` can deploy without the FastAPI server. When `NEXT_PUBLIC_API_URL` is
unset, the prompt page decodes the upload in an `OfflineAudioContext` and runs
`website/src/lib/audio/*` in a Web Worker. Nothing is uploaded.

The worker takes channel buffers in and returns the finished WAV plus the edited
float channels (all transferred, no copies), so chaining an edit never
re-quantises the audio. One worker per job, terminated when it finishes. If
`new Worker` throws (old browser, strict CSP) the same pure `applyPlan` runs on
the main thread, which still produces the right file.

| Operation | Method | Measured |
|---|---|---|
| trim, fade, gain, normalise, reverse | time domain | sample-exact length; peak hits −1 dBFS |
| speed | phase vocoder, pitch preserved | 20 s → **16.0 s** at 1.25x; dominant tone stayed **180.0 Hz** |
| exact BPM target | onset flux + comb grid + autocorrelation, target ratio | 8.0 s at 120 BPM → **7.5009 s** at 128 BPM (expected 7.5); real 90 BPM track → **25.129 s** at 128 BPM (expected 25.13) |
| tempo question | same estimator, no audio change | 8 s 120 BPM click → **120 BPM**; real 90/120 BPM tracks → **90.0 / 120.1 BPM**; real speech → **no tempo**, no download button |
| denoise | Wiener, 12th-percentile noise profile, temporal gain smoothing | **+0.67 / +1.20 / +1.66 dB** true SNR at strength 1 / 2 / 3 on a gated 440 Hz fixture at 4 dB input |
| boost vocals / drums / bass | 1–6 kHz / <200 Hz + 2–4 kHz / <250 Hz shelves | **+4.6 / +5.6 + 3.5 / +5.1 dB** relative to the rest of the mix |
| 5-band EQ, high/low-pass | RBJ biquads, same frequencies as the preview | +6 dB at 60 Hz raises the 60 Hz / 4 kHz energy ratio by **>1.5×** |
| pitch shift | resample + phase vocoder, duration preserved | +12 st moves a 440 Hz tone to **880 Hz ±20 Hz**, length **1.00 s** |
| pause removal | 20 ms RMS, 10 ms hop, 80 ms cut margin | 2.0 s pause at a 0.5 s threshold removes **1.84 s**; a 0.25 s gap is kept |
| remove section, loop | concatenation with 10 ms crossfades | 10 s minus 2 s → **7.99 s**; a 1 s loop ×3 on a 4 s file → **5.98 s** |
| pitch / transpose | same resample + phase-vocoder path, then peak-normalise | +2 st moves 440 Hz to **493.9 Hz ±10 Hz**, duration **1.00 s** |
| keep-last trim | keep the tail, reject an empty result | 40 s → last 30 s = **30.0 s**; a 20 s file is kept whole; cutting the whole file throws an error |
| Lab render | EQ → pitch → speed → gain → reverb, in a worker | preset export of 34 s: **33.933 s**, SHA-256 differs; +2 st + 1.25x + reverb → **27.146 s** (expected 27.146) |
| real pause removal | 20 ms RMS, 10 ms hop, 80 ms cut margin | LibriVox 36.467 s → **25.917 s**; 14 pauses ≥0.5 s, 10.6 s total |

Unsupported prompts (stem separation, generated instruments, transcription,
diarization, two-stem balance, non-WAV export) are marked **Server** on the
feature pages in browser builds. Clicking one still returns an explicit "that
one needs the server" message with the local alternative; they do not silently
no-op.

The Lab page is fully local in browser builds: it decodes the upload, measures
tempo and brightness, applies a deterministic preset, previews EQ/filters/gain/
speed with WebAudio, and renders the export in the worker. The server path is
unchanged when `NEXT_PUBLIC_API_URL` is set. The "AI" label was removed because
the preset is a rule set over measured features, not a model.

## Multitrack Studio

The Studio page (`/studio`) is a separate workspace: multiple tracks, clips, a
channel strip per track, a master bus, project bundles and DAW interchange. It
does not pretend to be a DAW.

| Operation | Method | Measured |
|---|---|---|
| timeline | clips, drag, split, duplicate, delete, fades, 50 ms crossfade | 2 tracks imported; split → 3 clips; undo → 2; redo → 3 |
| clip drag | pointer drag, 50 ms snap | 2.00 s start moved by 2.0 s |
| mixer | per-track gain/pan/EQ/compressor, master ceiling | rendered 20.0 s mix |
| loudness | K-weighted, gated 400 ms blocks; 4× true peak | **−14.5 LUFS integrated, −1.0 dBTP** on a 20 s mix |
| automation | volume points, linear dB interpolation, draggable lane | fade-in automation raises late RMS >10× over early RMS |
| sidechain | source-track envelope ducking | ducked mix RMS is lower than the same mix without sidechain |
| project bundle | ZIP with `project.json` + WAV assets | 2 tracks / 3 clips restored from the ZIP |
| stems | one WAV per track, ZIP | 2 stems downloaded |
| MIDI | Standard MIDI import/export, simple note synth | 120 BPM MIDI with C4/E4 notes → 1.0 s render, exported file starts `MThd` |
| lookahead limiter | 5 ms lookahead, 90 ms release, −1 dB ceiling | 20 s mix measured **−1.4 dBTP** |
| DAW interchange | EDL (Reaper/Premiere/Resolve), FCPXML (Final Cut), MIDI | EDL starts `TITLE: Untitled session`; FCPXML has `asset-clip` entries |
| video import | audio track decoded from MP4 | 5 s MP4 → 1 track / **4.99 s** audio |
| validation panel | LUFS, true peak, crest, DC, stereo correlation | 20 s mix: **−13.7 LUFS**, **−1.4 dBTP**, all four checks pass |

Known limits: no piano-roll MIDI editing, no VST/AU hosting, no ARA or spectral
editing, no perceptual listening validation. The channel compressor is a simple
feed-forward detector; stems ignore mute/solo by design and do not include
sidechain ducking. The validation checks are technical, not a claim of audible
quality. `docs/VALIDATION.md` defines the listening protocol.

## Combine takes

The Combine page (`/features/combine`) puts 2–5 audio files on one timeline. It
is local: decoding, alignment, mixing and WAV export all run in the browser.

| Operation | Method | Measured |
|---|---|---|
| multi-file import | 2–5 audio files, role guessed from filename | 2 files → 2 tracks; `piano.wav`/`vocals.wav` roles inferred correctly |
| auto-align | onset-envelope correlation (25 Hz coarse, 200 Hz fine), 25–120 s search | a vocal recorded **1.150 s late** was moved exactly **−1.150 s** |
| manual timing | offset in seconds, ±0.5/0.1/0.01 s nudges, drag a lane | −1.150 s dragged +0.5 s → **−0.650 s** |
| level match | RMS to a −18 dBFS target | both test files land at the target before faders |
| prompt balance | role + dB parser | “highlight the vocals over the piano by 3 dB” → vocals **+1.5 dB**, piano **−1.5 dB** |
| ordinal balance | “first…over…second”, “track 1 over track 2 by 4 dB” | first **+1.5 dB**, second **−1.5 dB** |
| polish chain | denoise → reverb → peak normalize, run in the DSP worker | “clean up… add a little reverb… reduce noise” → **−1.0 dBFS peak**, report lists all three steps |
| mix/export | stereo WAV, −1 dBFS safety ceiling | 2 files → **7.15 s WAV, 44.1 kHz, stereo**, clean console |

Known limits: alignment compares onset timing, not pitch or tempo, so files
recorded at different tempos are not stretched. A low-confidence match is
reported and the offset is left alone rather than guessed. Roles are inferred
from filenames and can be changed by hand; the page does not claim to identify a
voice from its sound. There is no source separation here — the files must already
be separate parts. **Polish is a fixed chain** (noise reduction, reverb, peak
normalization), not a mastering model; “make my music perfect” applies that chain
and reports each step rather than promising a perfect master.

Cost on mono 44.1 kHz, measured on this machine:

| Job | Bun/JSC | Chromium worker |
|---|---|---|
| denoise, 180 s | 8.1 s | 25.6 s on the 144 s file (≈32 s at 180 s) |
| speed 1.25x, 180 s | 7.0 s | **27.5 s** |

The earlier note that "V8 is comparable" was wrong: this Chromium build is
~3.5× slower than Bun/JSC on the same loops, so the browser timings — not the
Bun ones — are what a user sees.

Responsiveness is measured, not assumed. A 20 s denoise stalls the main thread
for **3.04 s** when run in-process and for at most **65 ms** in the worker
(50 ms heartbeat sampled across the whole job). Progress arrives in 2% steps and
renders as a labelled bar ("Denoising 42%"); a second click during a job is
ignored by a ref guard as well as the disabled button.

Verified on the deployed site (`https://audelle.vercel.app/prompt`, no backend
env): a 40 s WAV denoise took **6.9 s**, reported Starting 0% → Denoising 20/40/98%
→ Encoding WAV 100%, worst main-thread stall **64 ms**, and the downloaded file
was 44.1 kHz mono 16-bit PCM at exactly 40.000 s.

Known limit, same as the server: the denoiser assumes the noise floor is visible
in gaps. A continuous tone has no gaps, so its own magnitude becomes the 12th
percentile and the filter attenuates the tone.