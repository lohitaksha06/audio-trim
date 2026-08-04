# Audelle — AI-Powered Audio Editor

An intelligent audio editor where **everything is possible by prompting**. The AI understands audio at a deep level — instruments, structure, mood, and speaker intent — and executes edits from natural language. No drag handles, no waveform scrubbing (except manual mode fallback).

---

## Features

### 🧠 AI Audio Understanding (Core ML)
When audio is loaded, the model automatically analyzes and catalogs:
- **Source separation** — Identifies all stems: vocals, drums, bass, guitar, keys, pads, FX
- **Instrument classification** — "This is a Fender Strat, fingerpicked, with reverb"
- **Song structure** — Intro, verse, chorus, bridge, outro with timestamps
- **Mood/energy curve** — Tension, energy, loudness over time
- **Key, BPM, time signature**
- **Speaker diarization** — Who spoke when (for podcasts/interviews)
- **Transcription** — Full word-level transcript with timestamps

### 🎤 Natural Language Prompting

**Trimming & Arrangement**
- "Cut from the second chorus to the outro"
- "Remove the bridge entirely"
- "Make a 30-sec highlight reel from the best parts"

**Instrument / Stem Editing**
- "Remove the kick drum from 2:30 to 3:45"
- "Keep only the vocals"
- "Turn down the hi-hats by 50%"
- "Swap the bass with a synth pad in the bridge"
- "Give me the drums as a separate stem"

**Audio Inpainting & Fixing**
- "Remove that cymbal crash at 1:23 and fill smoothly"
- "Clean up the cough at 0:45"
- "Replace the guitar solo with something more melodic"

**Mood & Style**
- "Make this section sound darker / more cinematic"
- "Give the chorus more energy"
- "Add reverb to the vocals — make it sound like a cathedral"

**Podcast / Dialogue**
- "Remove all ums, ahs, and pauses longer than 0.5s"
- "Normalize all speakers to the same volume"
- "Split this interview by speaker"
- "Generate chapters from the transcript"

**Content Creator / Short-form**
- "Make a vertical 30-sec version for TikTok"
- "Add captions burned in"
- "Speed up the slow parts, keep exciting parts normal"

**Film / Video**
- "Clean up dialogue in this noisy scene"
- "Match room tone across all clips"
- "Separate dialogue, music, and SFX into stems"
- "Sync my voiced ADR to lip movements"

### ✋ Manual Mode (Fallback)
For fine-grained control when prompting isn't precise enough:
- Waveform + stem visualization
- Manual region selection
- Volume curves, fades, crossfades
- Drag-to-trim handles

### 📦 Export & Integration
- Export as MP3, WAV, FLAC, AAC, stems as ZIP
- Export to Premiere Pro / DaVinci Resolve / Final Cut XML
- Share link / download
- Batch processing

---

## Project Structure

```
audio-trim/
├── mobile/                          # React Native (Expo) app
│   ├── app/                         # Screens (file-based routing)
│   ├── components/                  # Reusable UI
│   ├── services/                    # API calls
│   └── assets/
│
├── website/                         # Next.js web app
│   ├── app/                         # Pages (App Router)
│   ├── components/
│   ├── services/
│   ├── hooks/
│   └── public/
│
├── server/                          # Backend API (FastAPI / Node.js?)
│   ├── routes/
│   ├── services/                    # Business logic
│   ├── ml/                          # ML model inference
│   │   ├── source_separation/
│   │   ├── transcription/
│   │   ├── diarization/
│   │   ├── audio_understanding/     # Instrument ID, structure, mood
│   │   └── prompting_engine/        # NL → audio operations
│   └── utils/
│
├── shared/                          # Shared types, validation, constants
│
├── docs/
│   └── architecture.md
│
└── README.md
```

---

## Work Plan

### Phase 1 — ML Backbone (Server) ✅ (implemented in `server/ml/`)
| Task | Details |
|------|---------|
| Source separation | Demucs / Hybrid Demucs — isolate vocals, drums, bass, other |
| Instrument classification | Feature-based detector — library includes a heuristic `instrument_classifier.py`; swappable for a pretrained model (musicnn/YamNet) |
| Song structure | `song_structure.py` — intro/verse/chorus/bridge/outro via beat-synced chroma+MFCC recurrence |
| Mood/energy curve | `mood_curve.py` — energy/tension/brightness/loudness over time + summary mood |
| Transcription | Whisper (openai/whisper-tiny) — word-level transcript |
| Speaker diarization | `diarization.py` — VAD + MFCC clustering; pyannote-ready (needs HF token) |
| Prompting engine | `prompt_llm.py` LLM-first with deterministic regex fallback (see `server/ml/prompt_engine.py`) |
| Audio inpainting | `inpainter.py` — seamless crossfade fill for removed/replaced ranges (paint intent) |

### Phase 2 — Server API & Processing Pipeline ✅ (implemented in `server/services/`)
- Async job queue — `jobs.py` thread-pool; `POST /api/jobs/{process,separate,transcribe,understand,diarize,inpaint}` + `GET /api/jobs/{id}`
- File upload / storage — `storage.py` local (default) + optional S3; uploads return a storage key
- Audio manipulation backend — pydub, librosa, ffmpeg (`audio_operations.py`, `converter.py`, `video_extractor.py`)
- Export pipeline — `exporter.py`: stems ZIP, FCPXML (Final Cut), CMX3600 EDL (Premiere/DaVinci); served via `GET /api/export/download?path=`

### Phase 3 — Website (Next.js)
- Upload audio (drag & drop, file picker, URL)
- Prompt input (text box, suggested prompts, voice input)
- Audio analysis results display (instruments, structure, transcript)
- Preview player with waveform + stems overlay
- Manual mode editor (fallback)
- Export / share

### Phase 4 — Mobile App (React Native / Expo)
- Same flow as web, optimized for touch
- File picker / share sheet audio import
- Offline processing queue (upload when connected)
- Local playback + prompt history
- Push notifications for job completion

### Phase 5 — Platform & Polish
- User accounts & project save/load
- Prompt history / favorites
- Batch processing
- API for NLE plugins (Premiere, DaVinci, Final Cut)
- Fine-tuning from user feedback — improve model over time
- Community prompt library

---
