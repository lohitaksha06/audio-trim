/**
 * Instrument + beat catalogue shown in the UI.
 *
 * This list is deliberately kept in step with the backend by a test
 * (`server/tests/test_synthesis.py::TestCatalogMatchesBackend`) that parses this
 * file and fails if it advertises anything the backend cannot actually render.
 * That is what stops the UI from promising instruments that do not exist.
 *
 * Trained timbres come from `server/ml/models/timbre_model.pt`; drum hits are
 * real recorded one-shots in `server/ml/models/drum_bank/`.
 */

export type Instrument = {
  /** The full prompt to send. Self-contained, so it can be tested as-is. */
  prompt: string;
  /** Button label. */
  label: string;
  /** Trained timbre family base, e.g. "guitar". */
  family: string;
  /** What the trained model actually learned for it. */
  blurb: string;
};

/** Instruments rendered by the trained timbre model. */
export const TRAINED_INSTRUMENTS: Instrument[] = [
  { prompt: "add guitar", label: "Guitar", family: "guitar", blurb: "Acoustic, electronic and synthetic variants" },
  { prompt: "add piano", label: "Piano & keys", family: "keyboard", blurb: "Acoustic, electronic and synthetic" },
  { prompt: "add bass", label: "Bass", family: "bass", blurb: "All three source types" },
  { prompt: "add flute", label: "Flute", family: "flute", blurb: "Acoustic, electronic, synthetic" },
  { prompt: "add strings", label: "Strings", family: "string", blurb: "Held chord section" },
  { prompt: "add organ", label: "Organ", family: "organ", blurb: "Sustained chords" },
  { prompt: "add marimba", label: "Marimba & bells", family: "mallet", blurb: "Arpeggio figure" },
  { prompt: "add brass", label: "Brass", family: "brass", blurb: "Sustained horn section" },
  { prompt: "add sax", label: "Sax & reeds", family: "reed", blurb: "Held line" },
  { prompt: "add vocals", label: "Vocal pad", family: "vocal", blurb: "Sustained layer" },
  { prompt: "add synth lead", label: "Synth lead", family: "synth_lead", blurb: "Melodic figure" },
];

/**
 * Sounds with no trained equivalent. These still work, but they are plain DSP
 * (oscillator + filter), not learned — the label says so.
 */
export const DSP_INSTRUMENTS: Instrument[] = [
  { prompt: "add an acid bassline", label: "Acid bassline", family: "", blurb: "DSP synth — not trained" },
  { prompt: "add an 808 bass", label: "808 sub bass", family: "", blurb: "DSP synth — not trained" },
  { prompt: "add supersaw", label: "Supersaw", family: "", blurb: "DSP synth — not trained" },
  { prompt: "add a reese bass", label: "Reese bass", family: "", blurb: "DSP synth — not trained" },
];

export type BeatStyle = { prompt: string; label: string; blurb: string };

/** Beat styles. Drums are real recorded one-shots, tempo-locked to your BPM. */
export const BEAT_STYLES: BeatStyle[] = [
  { prompt: "add house drums", label: "House", blurb: "Four-on-the-floor" },
  { prompt: "add techno drums", label: "Techno", blurb: "Driving offbeat hats" },
  { prompt: "add trap drums", label: "Trap", blurb: "Slow, hard 808 weight" },
  { prompt: "add dnb drums", label: "Drum & bass", blurb: "Amen-style breaks" },
  { prompt: "add edm drums", label: "EDM", blurb: "Big festival kick" },
  { prompt: "add hiphop drums", label: "Hip-hop", blurb: "Boom-bap backbeat" },
  { prompt: "add afrobeat drums", label: "Afrobeat", blurb: "Syncopated rim pattern" },
  { prompt: "add garage drums", label: "UK garage", blurb: "Shuffled two-step" },
  { prompt: "add breakbeat drums", label: "Breakbeat", blurb: "Broken groove" },
  { prompt: "add pop drums", label: "Pop", blurb: "Simple backbeat" },
];

/** Drum types available in the recorded bank. */
export const DRUM_TYPES = [
  "kick",
  "snare",
  "clap",
  "hat",
  "cymbal",
  "crash",
  "ride",
  "tom",
  "conga",
  "rim",
] as const;