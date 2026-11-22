# Real prompts users ask Audelle

These are verbatim-style requests from real sessions, with what they map to
and what is actually possible. They are kept so the prompt parser and the
server pipeline can be built to answer them honestly instead of pretending.

The running theme: people describe a *performance* ("her voice", "the guitar",
"the backing", "the drums"), not audio terms. The system must map roles to
stems, not ask for faders.

## 1. Vocal clarity + duck the backing under the lead

> Make her voice clearer and louder. Raise the guitar / backing instrument in
> the gaps where she isn't singing, and fade them back down just before she
> starts singing again. Reduce the male voice in the background if you can.

What it means:
- separate into stems (Demucs),
- enhance the vocal (high-pass, presence, air, light denoise) and lift it,
- build a vocal-activity envelope and duck the accompaniment by a few dB during
  singing, lift it in the gaps, with a lookahead so it fades *before* the vocal
  re-enters.

Honest limit: "reduce the male voice" is **not possible** with a 4/6-stem
model — every voice lands in one "vocals" stem. Do not claim to remove it. The
perceptual fix is to push the lead forward and duck everything else.

## 2. Instrumental is too quiet — make it audible in the gaps

> Increase the background guitar/instrument volume when she isn't singing and
> fade it back before she sings again. Amplify the instrument music.

Same as #1, but the complaint is that the backing was *inaudible*, so the gap
lift must be large (e.g. +8 dB), not a subtle nudge, and the backing base level
raised. The vocal must stay clearly on top.

## 3. Level automation over time (a section is too quiet)

> From 27 seconds the music is too quiet — increase the music and keep the
> vocals loud from there, with a smooth transition (don't jump it). Fade out
> slowly at the end.

What it means:
- a time-based gain ramp on the accompaniment starting at 27 s (smooth ~1 s),
- leave the vocal untouched,
- a slow cosine fade over the final few seconds of the whole mix.

## 4. Add drums that build in, at background level

> Add drums from 30 seconds when the music builds up, at the same level as the
> background instrument but still lower than the singer. Use a real kick
> (boom) and cymbals, not just thin noise.

What it means:
- synthesize a groove (kick, snare, hats, crash) aligned to the detected beat
  grid,
- enter at the requested time with a fade-in, and a crash to mark the entry,
- level it with the accompaniment and duck it under the vocal the same way.

Real feedback that shaped this: a drum layer made only of hats reads as "thin
noise"; the kick needs a low pitch-drop "boom" and the crash needs a band-passed
noise wash. Level drums *below* the lead and *with* the backing, never louder
than the song.

## Canonical phrasing to support

- "make her voice clearer and louder"
- "raise the guitar in the gaps where she isn't singing"
- "fade it back down just before she sings again"
- "increase the music from 0:27, keep the vocals loud"
- "fade out slowly at the end"
- "add drums from 0:30, same level as the backing, lower than the voice"
- "reduce the male/backing voice" (→ explain the limit, don't fake it)

## 5. Named guitar / piano chords

> add guitar chords: Am F C G
> play Cmaj7 then Am7 then Fmaj7 then G7
> use a C/G

What it means:
- parse chord names (root + quality + slash bass) into MIDI notes,
- strum one chord per bar on the detected beat grid,
- render with the trained guitar timbre (fallback: plucked-string synth).

Supported vocabulary: `C`, `Am`, `F#m`, `Bbmaj7`, `G7`, `Dsus4`, `C/G`,
`maj7/m7/7/m7b5/dim7/maj9/m9/9/6/m6/add9/sus2/sus4/7sus4`, and so on.

Still not supported: Roman-numeral progressions without a key ("I–V–vi–IV")
and chord *voicings* (guitar tab shapes). Named chords are in-key-independent —
they play exactly what was written.

## Capability notes

- Everything above needs **source separation** (server-side, Demucs). The
  browser-only build can only approximate "boost vocals" with a band EQ and
  cannot do time-based ducking of a real instrumental.
- Tempo detection for drums is `librosa.beat`; halve/double the detected BPM
  when it lands outside 60–180 to avoid a half-time or double-time groove.
