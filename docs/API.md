# Audelle backend API

FastAPI service. Interactive schema: **`/docs`** (Swagger) and
**`/openapi.json`**.

Base URL in production is the host you deploy to (e.g.
`https://audelle-api.onrender.com`). The Vercel frontend reads it at build time
from `NEXT_PUBLIC_API_URL`.

> **Status:** the backend is implemented and tested but **not deployed yet**.
> Until it is, the public surface a visitor actually uses is the browser build
> (local DSP). Nothing below is reachable on `audelle.vercel.app`.

## Auth

None. There is no account system and no billing. Abuse protection is:

- `AUDIDELLE_RATE_LIMIT` (default 120 requests/min per client) → **429** +
  `Retry-After` on heavy POSTs.
- `AUDIDELLE_MAX_UPLOAD_MB` (default 200) → **413** for oversized uploads.
- `AUDIDELLE_MAX_HEAVY_JOBS` (default 1) concurrent Demucs/Whisper jobs.
- `AUDIDELLE_RETENTION_HOURS` (default 24) storage cleanup.

## Sync endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | — | status, version, uptime, storage counts, rate limit |
| POST | `/api/upload` | multipart `file` | filename, size, `audio_path`, `source_video_path`, analysis |
| POST | `/api/process` | `{audio_path, prompt, stem_path?, stem_level?, source_video_path?}` | `intent`, `params`, `download_key`, `stems_keys`, `video_download_key`, `metadata` |
| POST | `/api/suggest` | `{prompt}` | concrete follow-up prompts |
| GET | `/api/ml/catalog` | — | instrument/beat/style catalogue |
| GET | `/api/ml/separate/sources` | — | available stems for the loaded model |
| POST | `/api/ml/separate` | `{audio_path, model?, stems?}` | stem paths (Demucs) |
| POST | `/api/ml/transcribe` | `{audio_path}` | transcript + segments (Whisper) |
| POST | `/api/ml/understand` | `{audio_path}` | instruments, structure, mood, genre, energy |
| POST | `/api/ml/diarize` | `{audio_path}` | speaker turns |
| POST | `/api/ml/optimize` | `{audio_path, ...}` | measured mix advice |
| POST | `/api/ml/inpaint` | `{audio_path, start, end}` | repaired audio |
| POST | `/api/lab/tweak` | `{audio_path, eq, pitch, speed, gain, reverb}` | rendered WAV |
| POST | `/api/lab/suggest` | `{audio_path}` | measured preset suggestion |
| POST | `/api/export/zip` | `{paths}` | ZIP of stems/exports |
| POST | `/api/export/fcpxml` | `{clips, ...}` | FCPXML |
| POST | `/api/export/edl` | `{clips, ...}` | EDL |
| GET | `/api/export/download?key=…` | — | file stream |
| POST | `/api/projects` | project document | saved project |
| GET | `/api/projects` | — | project list |
| GET/PUT/DELETE | `/api/projects/{id}` | project document | one project |
| GET | `/api/eval/*` | — | model/eval metrics |

### `POST /api/process` — the prompt endpoint

```json
{
  "audio_path": "uploads/abc.wav",
  "prompt": "add guitar chords: Am F C G",
  "stem_path": null,
  "stem_level": null,
  "source_video_path": null
}
```

The prompt is parsed by `plan_from_prompt` (LLM if `USE_LLM=1`, otherwise the
deterministic regex engine) and executed by `execute_plan`. Named chords,
`polish`, timed adds and the rest of the syntax live in
`docs/REAL_PROMPTS.md`.

## Async job endpoints

Use these for Demucs/Whisper work that must not block the request.

| Method | Path | Body |
|---|---|---|
| POST | `/api/jobs/process` | `{audio_path, prompt}` |
| POST | `/api/jobs/separate` | `{audio_path}` |
| POST | `/api/jobs/transcribe` | `{audio_path}` |
| POST | `/api/jobs/understand` | `{audio_path}` |
| POST | `/api/jobs/diarize` | `{audio_path}` |
| POST | `/api/jobs/inpaint` | `{audio_path, start, end}` |
| GET | `/api/jobs/{job_id}` | — |

Submit returns `{ "job_id": "…" }`; poll the GET until `ready` is true. Job
state is in memory and does not survive a restart (single-instance limit).

## Errors

| Code | Meaning |
|---|---|
| 400 | unsupported file / bad request |
| 413 | upload over the cap |
| 422 | request body validation |
| 429 | rate limit; honour `Retry-After` |
| 500 | processing failure (audio is not returned) |

## Local alternative

Every local edit is documented in the browser build. If the backend is not
configured, the frontend never calls it and runs the local DSP path instead.
