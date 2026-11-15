Audelle — test audio files
=========================

Outputs from processing the karaoke demo (yanakaraoke.mp3, a 90 s mono voice +
instrumental backing). All files are 44.1 kHz stereo.

karaoke_final.wav / .mp3
    Vocal clarity + level, instrumental ducking (louder in the gaps, tucked
    under the voice), a +4 dB music lift ramped in from 27 s, and a slow
    cosine fade over the final 5 s.

karaoke_final_with_drums.wav / .mp3
    Same mix plus drums: kick (low "boom") on beats 1 & 3, snare on 2 & 4,
    and quiet hi-hats, aligned to the ~99 BPM beat grid, entering at 30 s
    with a fade-in. No crash cymbal. Drums sit at the backing level and
    below the lead vocal.

karaoke_final.mp4 / karaoke_final_with_drums.mp4
    Both mixes as 1080x1920 (9:16) videos using the supplied image as the
    still frame, H.264 + AAC 192 kbps, 30 fps, 90.44 s.

These are personal recordings and are intentionally NOT tracked in git
(see the repo .gitignore). The prompts behind these edits are recorded in
../docs/REAL_PROMPTS.md.
