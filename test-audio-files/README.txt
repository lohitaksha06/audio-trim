Audelle — test audio files
=========================

Outputs from processing the karaoke demo (yanakaraoke.mp3, a 90 s mono voice +
instrumental backing). All files are 44.1 kHz stereo.

karaoke_final.wav / .mp3
    Vocal clarity + level, instrumental ducking (louder in the gaps, tucked
    under the voice), a +4 dB music lift ramped in from 27 s, and a slow
    cosine fade over the final 5 s.

karaoke_final_with_drums.wav / .mp3
    Same mix plus drums: kick (low "boom"), snare, hats and a crash cymbal,
    aligned to the ~99 BPM beat grid, entering at 30 s with a fade-in. Drums
    sit at the same level as the backing and below the lead vocal.

These are personal recordings and are intentionally NOT tracked in git
(see the repo .gitignore). The prompts behind these edits are recorded in
../docs/REAL_PROMPTS.md.
