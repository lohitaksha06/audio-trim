# Audelle roadmap to 10/10

A 10/10 here means **the best audio editing product for people who think in
words, not waveforms** — not a Reaper/Logic replacement. The DAW features exist
to serve that user, not to win a spec sheet.

## Principles (the "don't break it" rules)

1. **The local editor never regresses.** Every server feature has a labelled
   local fallback or an honest "needs the server" message. No dead buttons.
2. **Measure before claiming.** NLU accuracy, render latency, clipping and
   loudness are numbers in a test, not adjectives.
3. **One change, one gate.** Every change runs the full suite + build before it
   ships. `python -m scripts.release_check` is the gate.
4. **Feature flags for risky work.** New render paths ship behind a flag, then
   become the default after A/B.
5. **No fake capability.** If the model can't split two voices, the UI says so.
6. **Local-first, server-optional.** A visitor can trim and export a WAV with no
   backend. The backend adds separation, synthesis and interchange.

## Scorecard today (honest)

| Dimension | Today | 10/10 target |
|---|---|---|
| Browser editing | 6 | 8.5 |
| Studio / DAW-lite | 6.5 | 8.5 |
| Combine | 7 | 9 |
| Backend / ML | 7.5 code, 0 live | 9 live |
| Prompt understanding | 5.5 → 7 after chord/polish work | 9 |
| Design / UX | 7 | 9 |
| Reliability / QA | 6 → 6.5 | 9 |
| Production readiness | 3 → 4 | 9 |

A realistic ceiling is **8.5–9** for a focused product. 10/10 means "no obvious
reason to use anything else for this job", not "has every DAW feature".

## Phase 0 — done in this pass

- [x] Named chords: `Am F C G`, `Cmaj7`, `C/G`, 30 qualities, strummed per bar.
- [x] Server **polish** intent: denoise → reverb → normalize in one request.
- [x] Timed adds: "add drums from 30 seconds" starts at 30 s.
- [x] "fade out slowly" gets a 4 s fade.
- [x] Add-verb targeting: "add drums … below the voice" adds drums, not vocals.
- [x] Upload cap (413), rate limit (429), storage cleanup, health/timing.
- [x] Empty-session Studio buttons disabled.
- [x] Real-prompt NLU benchmark (`REAL_BENCH`) alongside the curated set.
- [x] Blind A/B/X harness (`scripts/ab_pairs.py`).

## Phase 1 — make the promise real (2–3 weeks)

**Goal: the server features work on the live site, and we know how they sound.**

1. **Deploy the backend** (blocked on a host account + disk).
   - Render blueprint is ready (`render.yaml`). Set `NEXT_PUBLIC_API_URL`.
   - Acceptance: `/api/health` reachable from the Vercel site; a separation job
     returns stems from the live prompt box.
2. **Listening validation on a real corpus** (`docs/VALIDATION.md`).
   - 3 music, 3 speech, 3 mixed tracks. Loudness-match, blind A/B, ≥10 trials.
   - Acceptance: every reported win has a count/CI, and regressions are fixed.
3. **Prompt NLU v2**
   - Expand `REAL_BENCH` to 50+ real prompts; target **≥90% intent** and
     **≥90% param** accuracy.
   - Add: Roman numerals with a detected key, "same but" follow-ups, explicit
     unsupported reasons (`male voice`, time automation).
   - Acceptance: benchmark score is printed in `release_check`.
4. **Studio render off the UI thread**
   - Move `renderProject` into a worker (project + assets via transferables).
   - Acceptance: a 5-minute project stays responsive; preview < 2 s on 4 cores.
5. **Server-side Studio bridge**
   - Upload a project bundle, render stems/mix with Demucs on the server.
   - Acceptance: same project produces the same clip layout locally and remotely.

## Phase 2 — production reliability (4–6 weeks)

1. **Jobs that survive a restart**: persist job metadata (SQLite/Redis) and
   recover or fail cleanly. Keep the in-memory pool for single-instance dev.
2. **Object storage**: S3-compatible for uploads/exports; local disk only a
   cache. Retention policy enforced there.
3. **Auth-free quotas** (as requested, no billing): per-IP daily minutes,
   signed upload tokens, and a "try locally" fallback when the quota is hit.
4. **Observability**: request/job metrics (count, latency, failure), a
   `/api/metrics` endpoint, and alert thresholds.
5. **Error budget**: a documented SLO (99.5% successful edits, p95 < 30 s for
   a 3-minute separation on the reference host).
6. **Mobile + accessibility pass**: 390 px, touch targets, keyboard, focus,
   reduced motion, screen-reader labels.

## Phase 3 — the pro layer (2–3 months)

1. **Buses, sends and groups** in Studio; parallel compression; sidechain buses.
2. **MIDI piano roll** with note drag/resize/velocity; real tempo map.
3. **Automation for pan, EQ, sends**, not just volume.
4. **Stem-aware editing**: per-stem solo/mute, per-stem limiter, sidechain in
   stems.
5. **Interchange fidelity**: AAF/OMF decision, FCPXML round-trip tests, EDL
   markers.
6. **Collaboration-lite**: share a project bundle link, comments (no accounts).

## Phase 4 — native / scale (4–6 months)

1. **Desktop shell (JUCE/Electron)** for VST/AU/AAX hosting, ARA, low-latency
   monitoring. This is where "real DAW" features land; the browser stays the
   fast path.
2. **GPU workers** for separation/transcription at scale, with a queue.
3. **Video timeline** and video export/muxing beyond audio replacement.

## Metrics that define "10/10"

| Metric | Target |
|---|---|
| NLU intent accuracy (real bench) | ≥ 90% |
| NLU param accuracy | ≥ 90% |
| Live edit success rate | ≥ 99.5% |
| p95 render, 3-min separation | < 30 s (reference host) |
| Preview latency, 5-min project | < 2 s |
| Blind A/B win rate vs previous | ≥ 60% with CI, ≥10 trials |
| Crash-free sessions | ≥ 99.5% |
| Local-only task completion (no backend) | 100% of advertised local edits |

## Risk register

| Risk | Mitigation |
|---|---|
| New render path clips | peak/true-peak gate in tests + `tanh` safety |
| Backend outage | local editor stays fully functional; clear "server needed" labels |
| Model licence | pin model versions; document licences before commercial launch |
| Cost spikes | per-IP quotas, job caps, retention cleanup |
| Prompt regression | `REAL_BENCH` runs in `release_check`; block on a score drop |
| Deployment breakage | staged deploy, smoke test, rollback to previous Vercel deploy |

## Immediate next 10 tasks

- [ ] Create the Render blueprint service and set `NEXT_PUBLIC_API_URL`.
- [ ] Run the listening protocol on 9 tracks.
- [ ] Expand `REAL_BENCH` to 50 cases and fix the failures.
- [ ] Add Roman-numeral chords with a detected key.
- [ ] Add explicit "unsupported" reasons for male-voice and time-automation asks.
- [ ] Move `renderProject` into a worker.
- [ ] Persist job metadata.
- [ ] Add `/api/metrics` + SLO doc.
- [ ] Mobile/accessibility audit with Playwright.
- [ ] Add per-IP daily minute quota with a local fallback.

## How this stays safe

`scripts/release_check.py` runs the server suite, the frontend suite, ESLint and
`next build`. A phase is only marked done when that gate is green and the
relevant metric moved. Risky render changes ship behind a flag and are compared
with the A/B harness before becoming default.
