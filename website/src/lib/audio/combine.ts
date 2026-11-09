/**
 * Multi-file combine engine.
 *
 * The job is small and concrete: take 2–5 recordings of the same performance
 * (a voice take and a piano take, a karaoke file and a vocal, several band
 * stems) and put them on one timeline. Two things matter:
 *
 * 1. Timing. Each file has an offset. It can be typed in, nudged, or found by
 *    correlating onset envelopes. Envelope correlation is used instead of raw
 *    waveform correlation because the files contain *different* material; what
 *    they share is when things happen, not the waveform itself.
 * 2. Balance. Each file has a gain. Natural-language instructions such as
 *    "highlight the vocals over the piano by 3 dB" are parsed into gain
 *    changes, so the user does not have to translate the request into faders.
 *
 * Everything here is pure and synchronous so it can be tested without a
 * browser. Decoding and encoding live in `decode.ts` and `wav.ts`.
 */

import type { LocalAudio } from "./types";

export const MAX_COMBINE_TRACKS = 5;

export type CombineRole =
  | "vocals"
  | "piano"
  | "guitar"
  | "drums"
  | "bass"
  | "backing"
  | "other";

export const ROLE_OPTIONS: { value: CombineRole; label: string }[] = [
  { value: "vocals", label: "Vocals" },
  { value: "piano", label: "Piano / keys" },
  { value: "guitar", label: "Guitar" },
  { value: "drums", label: "Drums" },
  { value: "bass", label: "Bass" },
  { value: "backing", label: "Backing / karaoke" },
  { value: "other", label: "Other" },
];

export interface CombineTrack {
  id: string;
  name: string;
  role: CombineRole;
  audio: LocalAudio;
  /** Seconds. Positive starts later; negative starts earlier. */
  offset: number;
  gainDb: number;
  muted: boolean;
}

export interface AlignResult {
  /** Seconds the track should move to line up with the reference. */
  offset: number;
  /** 0–1. Below ~0.2 the result is not trustworthy. */
  confidence: number;
}

const ROLE_KEYWORDS: Record<CombineRole, string[]> = {
  vocals: [
    "vocal", "vocals", "voice", "voices", "singing", "singer", "sung",
    "acapella", "a cappella",
  ],
  piano: ["piano", "keys", "keyboard", "pianoforte"],
  guitar: ["guitar", "guitars", "strum", "strumming"],
  drums: ["drum", "drums", "beat", "drums", "percussion"],
  bass: ["bass", "bassline", "bass line"],
  backing: [
    "backing", "karaoke", "instrumental", "accompaniment", "background",
    "music", "playback", "minus", "bed",
  ],
  other: ["other"],
};

const ROLE_ORDER: CombineRole[] = ["vocals", "piano", "guitar", "drums", "bass", "backing", "other"];

export function roleLabel(role: CombineRole): string {
  return ROLE_OPTIONS.find((option) => option.value === role)?.label ?? role;
}

/** Guess a role from the filename so the first pass is usually right. */
export function suggestRole(name: string): CombineRole {
  const lower = name.toLowerCase();
  for (const role of ROLE_ORDER) {
    if (ROLE_KEYWORDS[role].some((word) => lower.includes(word))) return role;
  }
  return "other";
}

function dbToGain(db: number): number {
  return 10 ** (db / 20);
}

export function rmsDb(audio: LocalAudio): number {
  let sum = 0;
  let count = 0;
  for (const channel of audio.channels) {
    for (let i = 0; i < channel.length; i++) {
      sum += channel[i] * channel[i];
      count++;
    }
  }
  const rms = count ? Math.sqrt(sum / count) : 0;
  return rms > 1e-9 ? 20 * Math.log10(rms) : -Infinity;
}

/** Mix down to one channel for analysis only. */
function mono(audio: LocalAudio): Float32Array {
  const channels = audio.channels;
  if (channels.length === 1) return channels[0];
  const length = channels[0]?.length ?? 0;
  const out = new Float32Array(length);
  for (const channel of channels) {
    for (let i = 0; i < length; i++) out[i] += channel[i] / channels.length;
  }
  return out;
}

/**
 * Onset-strength envelope at `rate` Hz.
 *
 * RMS per frame, then the positive change in log energy. A sustained note
 * produces one onset; repeated chords and words produce peaks where they
 * actually happen. That is what two different recordings of the same take
 * share.
 */
export function onsetEnvelope(audio: LocalAudio, rate = 50): Float32Array {
  const source = mono(audio);
  const hop = Math.max(1, Math.round(audio.sampleRate / rate));
  const frames = Math.max(1, Math.floor(source.length / hop));
  const energy = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame++) {
    let sum = 0;
    const start = frame * hop;
    const end = Math.min(source.length, start + hop);
    for (let i = start; i < end; i++) sum += source[i] * source[i];
    energy[frame] = Math.sqrt(sum / Math.max(1, end - start));
  }
  const onset = new Float32Array(frames);
  for (let i = 1; i < frames; i++) {
    const rise = Math.log1p(100 * energy[i]) - Math.log1p(100 * energy[i - 1]);
    onset[i] = Math.max(0, rise);
  }
  // Zero-mean, unit-ish scale so the correlation is a cosine similarity.
  let mean = 0;
  for (let i = 0; i < frames; i++) mean += onset[i] / frames;
  let variance = 0;
  for (let i = 0; i < frames; i++) variance += (onset[i] - mean) ** 2 / frames;
  const scale = Math.sqrt(variance) || 1;
  for (let i = 0; i < frames; i++) onset[i] = (onset[i] - mean) / scale;
  return onset;
}

function correlate(a: Float32Array, b: Float32Array, shift: number): { score: number; overlap: number } {
  // Compare a[i] with b[i - shift]; shift is how far b moves on the timeline.
  const start = Math.max(0, shift);
  const end = Math.min(a.length, b.length + shift);
  let dot = 0;
  let energyA = 0;
  let energyB = 0;
  let overlap = 0;
  for (let i = start; i < end; i++) {
    const x = a[i];
    const y = b[i - shift];
    dot += x * y;
    energyA += x * x;
    energyB += y * y;
    overlap++;
  }
  const denominator = Math.sqrt(energyA * energyB);
  return { score: denominator > 1e-9 ? dot / denominator : 0, overlap };
}

function bestShift(a: Float32Array, b: Float32Array, maxShift: number): { shift: number; score: number } {
  const shortest = Math.min(a.length, b.length);
  let bestShiftValue = 0;
  let bestScore = -Infinity;
  for (let shift = -maxShift; shift <= maxShift; shift++) {
    const { score, overlap } = correlate(a, b, shift);
    // A tiny overlap can score 1.0 by matching three samples. Weight by how
    // much of the shorter file actually participates, and ignore the tail.
    if (overlap < shortest * 0.25) continue;
    const weighted = score * Math.sqrt(overlap / shortest);
    if (weighted > bestScore) {
      bestScore = weighted;
      bestShiftValue = shift;
    }
  }
  return { shift: bestShiftValue, score: bestScore };
}

/**
 * Find how far `track` must move to line up with `reference`.
 *
 * Coarse pass at 25 Hz over ±60 s, then a fine pass at 200 Hz around the
 * coarse answer. The result stays useful for long performances without an
 * O(n²) search over the whole file.
 */
export function alignTo(
  reference: LocalAudio,
  track: LocalAudio,
  options: { maxSeconds?: number } = {},
): AlignResult {
  const maxSeconds = Math.max(2, Math.min(120, options.maxSeconds ?? 60));
  const coarseRate = 25;
  const coarseA = onsetEnvelope(reference, coarseRate);
  const coarseB = onsetEnvelope(track, coarseRate);
  const coarseMax = Math.min(coarseA.length, coarseB.length, Math.round(maxSeconds * coarseRate));
  const coarse = bestShift(coarseA, coarseB, coarseMax);
  const coarseSeconds = coarse.shift / coarseRate;

  const fineRate = 200;
  const fineA = onsetEnvelope(reference, fineRate);
  const fineB = onsetEnvelope(track, fineRate);
  const fineWindow = Math.round(0.35 * fineRate);
  const centre = Math.round(coarseSeconds * fineRate);
  const shortest = Math.min(fineA.length, fineB.length);
  let best = { shift: centre, score: 0, weighted: -Infinity };
  for (let shift = centre - fineWindow; shift <= centre + fineWindow; shift++) {
    if (shift < -fineA.length || shift > fineB.length) continue;
    const { score, overlap } = correlate(fineA, fineB, shift);
    if (overlap < shortest * 0.25) continue;
    const weighted = score * Math.sqrt(overlap / shortest);
    if (weighted > best.weighted) best = { shift, score, weighted };
  }

  return {
    offset: best.shift / fineRate,
    confidence: Math.max(0, Math.min(1, best.score)),
  };
}

export interface AlignOutcome {
  tracks: CombineTrack[];
  notes: string[];
}

/**
 * Align every unmuted track to the first unmuted track.
 *
 * The reference keeps its place; the rest move. A low-confidence result is
 * reported, not silently applied as if it were certain.
 */
export function autoAlign(tracks: CombineTrack[]): AlignOutcome {
  const reference = tracks.find((track) => !track.muted) ?? tracks[0];
  if (!reference || tracks.length < 2) {
    return { tracks, notes: ["Add at least two files to align."] };
  }
  const notes: string[] = [];
  const next = tracks.map((track) => {
    if (track.id === reference.id || track.muted) return track;
    const result = alignTo(reference.audio, track.audio);
    if (result.confidence < 0.2) {
      notes.push(
        `${track.name}: no confident match (${Math.round(result.confidence * 100)}%). Kept its current offset - adjust it by hand.`,
      );
      return track;
    }
    notes.push(
      `${track.name}: moved ${result.offset >= 0 ? "+" : ""}${result.offset.toFixed(3)} s (${Math.round(result.confidence * 100)}% match).`,
    );
    return { ...track, offset: Number(result.offset.toFixed(3)) };
  });
  return {
    tracks: next.map((track) => (track.id === reference.id ? { ...track, offset: 0 } : track)),
    notes,
  };
}

/**
 * Bring every file to roughly the same RMS so the first listen is usable.
 *
 * This is deliberately RMS, not perceived loudness: a full LUFS meter per file
 * would be heavier than the job needs, and the user can still move every fader.
 */
export function matchLevels(tracks: CombineTrack[], targetDb = -18): CombineTrack[] {
  return tracks.map((track) => {
    const rms = rmsDb(track.audio);
    if (!Number.isFinite(rms)) return track;
    return { ...track, gainDb: clampGain(track.gainDb + (targetDb - rms)) };
  });
}

export interface MixResult {
  audio: LocalAudio;
  peak: number;
  start: number;
  duration: number;
  /** Set when a safety scale was applied to keep the mix under 0 dBFS. */
  scaledBy: number;
}

/** Sum the tracks on a shared timeline, keeping their offsets and gains. */
export function mixTracks(tracks: CombineTrack[], sampleRate?: number): MixResult {
  const audible = tracks.filter((track) => !track.muted && track.audio.channels.length > 0);
  const rate = sampleRate ?? audible[0]?.audio.sampleRate ?? 44100;
  if (!audible.length) {
    return { audio: { channels: [new Float32Array(0), new Float32Array(0)], sampleRate: rate }, peak: 0, start: 0, duration: 0, scaledBy: 1 };
  }
  const start = Math.min(0, ...audible.map((track) => track.offset));
  const end = Math.max(...audible.map((track) => track.offset + (track.audio.channels[0]?.length ?? 0) / track.audio.sampleRate));
  const frames = Math.max(1, Math.ceil((end - start) * rate));
  const left = new Float32Array(frames);
  const right = new Float32Array(frames);

  for (const track of audible) {
    const gain = dbToGain(track.gainDb);
    const destStart = Math.round((track.offset - start) * rate);
    const sourceRate = track.audio.sampleRate;
    const ratio = sourceRate === rate ? 1 : sourceRate / rate;
    const frameCount = track.audio.channels[0]?.length ?? 0;
    const sourceLeft = track.audio.channels[0];
    const sourceRight = track.audio.channels[Math.min(1, track.audio.channels.length - 1)];
    for (let i = 0; i < frameCount; i++) {
      const dest = destStart + i;
      if (dest < 0 || dest >= frames) continue;
      const sourceIndex = ratio === 1 ? i : Math.min(frameCount - 1, Math.floor(i * ratio));
      left[dest] += (sourceLeft[sourceIndex] ?? 0) * gain;
      right[dest] += (sourceRight[sourceIndex] ?? 0) * gain;
    }
  }

  let peak = 0;
  for (let i = 0; i < frames; i++) {
    peak = Math.max(peak, Math.abs(left[i]), Math.abs(right[i]));
  }
  // Balance is preserved because every track gets the same scale. This only
  // catches the arithmetic overflow of summing several hot files.
  const ceiling = 10 ** (-1 / 20);
  const scaledBy = peak > ceiling ? ceiling / peak : 1;
  if (scaledBy !== 1) {
    for (let i = 0; i < frames; i++) {
      left[i] *= scaledBy;
      right[i] *= scaledBy;
    }
    peak = ceiling;
  }

  return {
    audio: { channels: [left, right], sampleRate: rate },
    peak,
    start,
    duration: frames / rate,
    scaledBy,
  };
}

export interface CombineInstruction {
  /** Per-track gain delta in dB, keyed by track id. */
  gainsDb: Record<string, number>;
  summary: string;
  matched: CombineRole[];
  unmatched: CombineRole[];
}

function clampGain(value: number): number {
  return Math.max(-24, Math.min(12, value));
}

function rolesIn(text: string, tracks: CombineTrack[]): CombineRole[] {
  const present = new Set(tracks.map((track) => track.role));
  const found: CombineRole[] = [];
  for (const role of ROLE_ORDER) {
    if (!present.has(role) && role !== "other") continue;
    const hit = ROLE_KEYWORDS[role].some((word) => text.includes(word));
    if (hit) found.push(role);
  }
  return found;
}

function firstDb(text: string): number | null {
  const match = text.match(/([+-]?\d+(?:\.\d+)?)\s*(?:db|decibels?)\b/);
  if (!match) return null;
  return Number(match[1]);
}

function ordinalIndex(text: string): number | null {
  const words: [string, number][] = [
    ["first", 0], ["second", 1], ["third", 2], ["fourth", 3], ["fifth", 4],
    ["1st", 0], ["2nd", 1], ["3rd", 2], ["4th", 3], ["5th", 4],
  ];
  for (const [word, index] of words) if (text.includes(word)) return index;
  return null;
}

/**
 * Translate a balance instruction into per-track gain changes.
 *
 * Supported forms, including the ones people actually type:
 *
 *   "highlight the vocals over the piano by 3 dB"
 *   "bring the singing up over the backing"
 *   "duck the piano under the vocals by 6 dB"
 *   "make the vocals louder and the piano quieter"
 *   "piano -4 dB, vocals +2 dB"
 *   "feature the second file"
 *
 * Gains are *deltas* on top of the current fader values.
 */
export function parseCombineInstruction(text: string, tracks: CombineTrack[]): CombineInstruction {
  const lower = text.toLowerCase().replace(/\s+/g, " ").trim();
  const deltas: Record<string, number> = {};
  const matched = new Set<CombineRole>();
  const notes: string[] = [];

  const applyRole = (role: CombineRole, db: number) => {
    const targets = tracks.filter((track) => track.role === role);
    if (!targets.length) return false;
    for (const track of targets) deltas[track.id] = clampGain((deltas[track.id] ?? 0) + db);
    matched.add(role);
    return true;
  };

  // 1. "highlight X over Y by N dB" — split the difference so X ends up N dB
  //    above Y without turning the whole mix up.
  const boundary = /\b(over|above|against|on top of|in front of|than)\b/.exec(lower);
  if (boundary) {
    const left = lower.slice(0, boundary.index);
    const right = lower.slice(boundary.index + boundary[0].length);
    const leftRoles = rolesIn(left, tracks);
    const rightRoles = rolesIn(right, tracks);
    const focusVerb = /(highlight|bring|raise|boost|feature|prioriti[sz]e|louder|more|forward|on top)/.test(left);
    if (focusVerb && leftRoles.length && rightRoles.length && leftRoles[0] !== rightRoles[0]) {
      const amount = Math.abs(firstDb(lower) ?? 3);
      for (const role of leftRoles) applyRole(role, amount / 2);
      for (const role of rightRoles) {
        if (!leftRoles.includes(role)) applyRole(role, -amount / 2);
      }
      const focusNames = leftRoles.map(roleLabel).join(" + ");
      const duckNames = rightRoles.filter((role) => !leftRoles.includes(role)).map(roleLabel).join(" + ");
      notes.push(
        `${focusNames} +${(amount / 2).toFixed(1)} dB, ${duckNames} -${(amount / 2).toFixed(1)} dB (${amount} dB apart).`,
      );
      return finish(deltas, matched, notes);
    }
  }

  // 2. "duck / lower X under Y by N dB" — move X down; Y only needs to stay put.
  const rolePattern =
    "(vocals?|voices?|singing|singer|lead|piano|keys|keyboard|guitar|drums?|bass|backing|karaoke|instrumental|accompaniment|background|music|other)";
  const duck = new RegExp(
    `(?:duck|push\\s+back|turn\\s+down|make\\s+quieter)\\s+(?:the\\s+)?${rolePattern}(?:\\s+(?:under|below|behind)\\s+(?:the\\s+)?${rolePattern})?`,
    "i",
  ).exec(lower);
  if (duck) {
    const role = roleFromWord(duck[1]);
    if (role) {
      const amount = firstDb(lower) ?? 6;
      applyRole(role, -Math.abs(amount));
      notes.push(`${roleLabel(role)} -${Math.abs(amount)} dB.`);
      const other = roleFromWord(duck[2] ?? "");
      if (other) matched.add(other);
      return finish(deltas, matched, notes);
    }
  }

  // 3. "X louder / quieter", "X +3 dB", including two clauses in one sentence.
  const clauses = lower.split(/,|\band\b|\bbut\b|\bthen\b/).map((item) => item.trim()).filter(Boolean);
  for (const clause of clauses) {
    const roles = rolesIn(clause, tracks);
    if (!roles.length) continue;
    const amount = Math.abs(firstDb(clause) ?? 3);
    const signed = clause.match(/[+-]\s*\d+(?:\.\d+)?\s*(?:db|decibels?)/);
    const down = /(quieter|softer|lower|reduce|duck|push back|turn down|too loud|behind|under)/.test(clause);
    const up = /(louder|boost|raise|bring up|highlight|feature|prioriti[sz]e|forward|more)/.test(clause);
    const db = signed
      ? Number(signed[0].replace(/\s*(?:db|decibels?)/, "").replace(/\s+/g, ""))
      : down
        ? -amount
        : up
          ? amount
          : null;
    if (db === null) continue;
    const role = roles[0];
    applyRole(role, db);
    notes.push(`${roleLabel(role)} ${db >= 0 ? "+" : ""}${db.toFixed(1)} dB.`);
  }

  // 4. "feature the second file" / "highlight the first one"
  if (!Object.keys(deltas).length) {
    const index = ordinalIndex(lower);
    const focus = /(highlight|feature|louder|bring up|raise|boost|prioriti[sz]e)/.test(lower);
    if (index !== null && tracks[index] && focus) {
      const amount = firstDb(lower) ?? 3;
      deltas[tracks[index].id] = clampGain(amount);
      notes.push(`${tracks[index].name} +${amount.toFixed(1)} dB.`);
    }
  }

  return finish(deltas, matched, notes);
}

function roleFromWord(word: string): CombineRole | null {
  const lower = word.toLowerCase();
  for (const role of ROLE_ORDER) {
    if (ROLE_KEYWORDS[role].some((keyword) => lower.includes(keyword))) return role;
  }
  return null;
}

function finish(
  deltas: Record<string, number>,
  matched: Set<CombineRole>,
  notes: string[],
): CombineInstruction {
  const unmatched: CombineRole[] = [];
  if (!Object.keys(deltas).length) {
    return {
      gainsDb: {},
      summary:
        "I could not turn that into a balance change. Try “highlight the vocals over the piano by 3 dB” or “duck the backing by 4 dB”.",
      matched: [],
      unmatched,
    };
  }
  return {
    gainsDb: deltas,
    summary: notes.join(" "),
    matched: [...matched],
    unmatched,
  };
}

/** Apply an instruction to the track list, returning a new list. */
export function applyInstruction(tracks: CombineTrack[], instruction: CombineInstruction): CombineTrack[] {
  return tracks.map((track) => {
    const delta = instruction.gainsDb[track.id];
    if (delta === undefined) return track;
    return { ...track, gainDb: clampGain(track.gainDb + delta) };
  });
}
