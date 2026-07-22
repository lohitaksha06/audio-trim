# EchoEdit

Edit audio by typing what you want — not by learning editing software.

> **Naming note:** "EchoEdit" is a placeholder pick from our shortlist (Trimly, Waveform, HybridCut, IntentMix future names). It's a straightforward find-and-replace throughout this file if the team lands on something else.

## What this is

EchoEdit takes an existing audio file and a plain-English instruction — "cut the first 10 seconds", "remove the background hiss", "make the voice louder" — and produces the edited audio automatically. No timeline scrubbing, no menus, no learning curve.

This is **editing**, not generation. Generation means creating new audio from nothing ("make the sound of rain"). EchoEdit only modifies audio that already exists.

## Why it's worth building

- Podcasters and YouTube creators repeatedly perform the same handful of edits manually — this removes that friction entirely.
- Natural-language audio editing is an active, recent research area (2025 papers like SAO-Instruct and "Guiding Audio Editing with Audio Language Model" work on adjacent problems), but there's no dominant commercial product yet.
- Existing approaches lean on diffusion models, which are accurate but slow and heavy for common edits.
- EchoEdit's angle: route deterministic operations (trim, gain, EQ) to fast classical DSP, and reserve ML models only for tasks that actually need them (denoising, source separation). Faster and more practical than diffusion-only pipelines for everyday edits.

## Supported operations (MVP)

| Operation | How it's executed |
| --- | --- |
| Trim / cut a section | Classical DSP (pydub / ffmpeg) |
| Remove silence | Classical DSP |
| Adjust volume / loudness | Classical DSP |
| Basic EQ (bass/treble) | Classical DSP |
| Reduce background noise | Pretrained model (DeepFilterNet / noisereduce) |
| Isolate vocals from music | Pretrained model (Demucs) |

## How it works

```text
User instruction (text)
        │
        ▼
┌───────────────────┐
│  Intent Parser     │   LLM converts free text → structured command
│  (LLM-based)       │   e.g. {"action": "trim", "start": 0, "end": 10}
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Execution Layer   │   Routes command to the right tool:
│                    │   - Deterministic ops → DSP (pydub/ffmpeg)
│                    │   - Noise reduction   → DeepFilterNet
│                    │   - Vocal isolation   → Demucs
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Interface Layer   │   Upload → instruction → before/after preview → download
└───────────────────┘
```

The intent parser is the only place an LLM is used — audio itself is never generated or synthesized, only transformed.

## Tech stack

- **Intent parsing:** LLM via API (structured JSON output)
- **DSP:** pydub, ffmpeg
- **Noise reduction:** DeepFilterNet or noisereduce
- **Source separation:** Demucs (pretrained inference only, no training from scratch)
- **Interface:** Minimal web app (upload → instruct → preview → download)

## Getting started

```bash
# clone the repo
git clone <repo-url>
cd echoedit

# install dependencies
pip install -r requirements.txt
npm install   # if the web interface lives in this repo too

# run the app
python app.py
```

Update this section once the actual project scaffold (folder structure, entry point, env vars) is in place — this is a placeholder for the eventual setup.

## Example usage

- **Upload:** `podcast_episode_04.wav`
- **Instruction:** "cut the first 10 seconds and remove the background hiss"
- **Output:** `podcast_episode_04_edited.wav`

## Evaluation

We evaluate two layers separately:

- **Intent parser:** accuracy / F1 on a held-out set of labeled instruction–action pairs, compared against a rule-based/keyword-matching baseline (ablation).
- **Audio quality:** SNR improvement, PESQ, and STOI for noise reduction; SDR for source separation (via MUSDB18 and internal test clips).
- **Latency:** end-to-end pipeline latency vs. reported latency of diffusion-based baselines (e.g., SAO-Instruct), to support the "faster and more practical" claim.

Full methodology is in the project report (`audio_editing_project_report.docx`).

## Roadmap

| Weeks | Milestone |
| --- | --- |
| 1–2 | Core MVP scope + intent parser working end-to-end |
| 3–5 | Wire up real audio engine (DSP + pretrained models) |
| 6 | Minimal web interface |
| 7–8 | Polish, demo to real podcast/YouTube editors, fix common failure cases |

## Related work

- SAO-Instruct (2025) — diffusion-based, instruction-guided audio editing
- "Guiding Audio Editing with Audio Language Model" (2025)
- WavCraft — LLM-orchestrated audio generation/editing pipelines

EchoEdit's contribution relative to this work is a hybrid, lower-latency pipeline: classical DSP for deterministic edits, learned models only where necessary.

## Status

Early-stage / coursework + prototype build. Not yet production-ready. Feedback from real podcast/content creators is actively being collected — reach out if you'd like to try it on your own audio.
