# Listening and validation harness

This is the protocol for validating studio renders. It separates technical
checks from perceptual claims.

## Technical checks (automated)

The Studio master panel runs these after every mix render:

| Check | Threshold | Why |
|---|---|---|
| True peak | ≤ −1.0 dBTP | Avoid inter-sample clipping on lossy encoders |
| Integrated loudness | > −40 LUFS | Catch a silent or nearly silent render |
| DC offset | < 0.01 | Catch a biased waveform |
| Stereo correlation | > −0.2 | Catch a phase-inverted or mono-incompatible mix |

The renderer also reports integrated LUFS, short-term LUFS, sample peak, true
peak and crest factor. The limiter is a 5 ms lookahead limiter with a 90 ms
release; it is not a lookahead brickwall with programme-dependent release.

## Perceptual (listening) checks

These require a human and must not be reported as passed by code.

### Corpus

Use at least three categories:

1. **Music** — 3 tracks with different tempo, key and density.
2. **Speech / podcast** — 3 recordings with pauses and steady noise.
3. **Mixed material** — 3 tracks with music under speech.

Record source, licence, duration, sample rate and processing chain in a
`validation/corpus.json` file. Do not commit copyrighted commercial tracks.

### Loudness matching

Before comparing two renders, match integrated LUFS to within **0.2 LU**. A
louder render is usually preferred even when it is worse. The Studio report
gives the value to match.

### Blind A/B

1. Render version A and version B.
2. Copy both to a player that hides their names, or randomise the labels.
3. Match loudness.
4. Compare the same 10–20 s excerpt, looped.
5. Record a preference and a confidence rating (1–5).
6. Run at least 10 trials across the corpus.
7. Report the preference count and the confidence interval. A 6/10 result is
   not evidence of an improvement.

### ABX

For a claimed audible difference (for example “the limiter is transparent”),
run an ABX test with at least 10 trials. Do not claim transparency from a null
result with fewer than 10 trials.

## Current status

- Technical checks: implemented in the Studio master panel.
- Automated render/loudness tests: `website/src/lib/studio/studio.test.ts`.
- Human listening test: **not run**.
- Real-song perceptual validation: **not run**.

The absence of listening results is deliberate. A measured LUFS value is not a
quality claim.
