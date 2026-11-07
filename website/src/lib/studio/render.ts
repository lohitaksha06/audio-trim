/**
 * Offline timeline renderer.
 *
 * The browser preview plays a rendered mix rather than scheduling a graph of
 * live WebAudio nodes: the mix you hear is the mix you export, and there is one
 * code path for clips, fades, pan, EQ, compression and the master ceiling.
 */

import type { LocalAudio } from "@/lib/audio";
import { measureLoudness } from "./loudness";
import {
  clipEnd,
  projectDuration,
  type LoudnessReport,
  type StudioAsset,
  type StudioClip,
  type StudioProject,
  type StudioTrack,
} from "./types";

export interface RenderOptions {
  start?: number;
  end?: number;
  /** Render the master bus. Stems leave it out so they can be re-mixed. */
  master?: boolean;
  /** Render only these tracks. */
  onlyTrackIds?: string[];
}

export interface RenderResult {
  audio: LocalAudio;
  report: LoudnessReport;
  duration: number;
}

export interface StemResult {
  trackId: string;
  name: string;
  audio: LocalAudio;
}

interface Biquad {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

export function dbToGain(db: number): number {
  return Math.pow(10, db / 20);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function lowShelf(freq: number, gainDb: number, sr: number, q = 0.707): Biquad {
  const a = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * Math.min(freq, sr * 0.45)) / sr;
  const cos = Math.cos(w0);
  const sin = Math.sin(w0);
  const alpha = (sin / 2) * Math.sqrt((a + 1 / a) * (1 / q - 1) + 2);
  const twoSqrtAAlpha = 2 * Math.sqrt(a) * alpha;
  const a0 = (a + 1) + (a - 1) * cos + twoSqrtAAlpha;
  return {
    b0: (a * ((a + 1) - (a - 1) * cos + twoSqrtAAlpha)) / a0,
    b1: (2 * a * ((a - 1) - (a + 1) * cos)) / a0,
    b2: (a * ((a + 1) - (a - 1) * cos - twoSqrtAAlpha)) / a0,
    a1: (-2 * ((a - 1) + (a + 1) * cos)) / a0,
    a2: ((a + 1) + (a - 1) * cos - twoSqrtAAlpha) / a0,
  };
}

function highShelf(freq: number, gainDb: number, sr: number, q = 0.707): Biquad {
  const a = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * Math.min(freq, sr * 0.45)) / sr;
  const cos = Math.cos(w0);
  const sin = Math.sin(w0);
  const alpha = (sin / 2) * Math.sqrt((a + 1 / a) * (1 / q - 1) + 2);
  const twoSqrtAAlpha = 2 * Math.sqrt(a) * alpha;
  const a0 = (a + 1) - (a - 1) * cos + twoSqrtAAlpha;
  return {
    b0: (a * ((a + 1) + (a - 1) * cos + twoSqrtAAlpha)) / a0,
    b1: (-2 * a * ((a - 1) + (a + 1) * cos)) / a0,
    b2: (a * ((a + 1) + (a - 1) * cos - twoSqrtAAlpha)) / a0,
    a1: (2 * ((a - 1) - (a + 1) * cos)) / a0,
    a2: ((a + 1) - (a - 1) * cos - twoSqrtAAlpha) / a0,
  };
}

function peaking(freq: number, gainDb: number, q: number, sr: number): Biquad {
  const a = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * Math.min(freq, sr * 0.45)) / sr;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha / a;
  return {
    b0: (1 + alpha * a) / a0,
    b1: (-2 * cos) / a0,
    b2: (1 - alpha * a) / a0,
    a1: (-2 * cos) / a0,
    a2: (1 - alpha / a) / a0,
  };
}

function applyBiquad(x: Float32Array, filter: Biquad): Float32Array {
  const out = new Float32Array(x.length);
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const y = filter.b0 * x[i] + filter.b1 * x1 + filter.b2 * x2 - filter.a1 * y1 - filter.a2 * y2;
    x2 = x1;
    x1 = x[i];
    y2 = y1;
    y1 = y;
    out[i] = y;
  }
  return out;
}

function applyChannelEq(audio: LocalAudio, eq: { low: number; mid: number; high: number }): LocalAudio {
  const filters: Biquad[] = [];
  if (Math.abs(eq.low) > 0.01) filters.push(lowShelf(120, eq.low, audio.sampleRate));
  if (Math.abs(eq.mid) > 0.01) filters.push(peaking(1000, eq.mid, 1, audio.sampleRate));
  if (Math.abs(eq.high) > 0.01) filters.push(highShelf(8000, eq.high, audio.sampleRate));
  if (!filters.length) return audio;
  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((channel) => filters.reduce((current, filter) => applyBiquad(current, filter), channel)),
  };
}

function compress(audio: LocalAudio, settings: StudioTrack["compressor"]): LocalAudio {
  if (!settings.enabled) return audio;
  const threshold = dbToGain(settings.thresholdDb);
  const ratio = Math.max(1, settings.ratio);
  const attack = Math.exp(-1 / Math.max(1, (settings.attackMs / 1000) * audio.sampleRate));
  const release = Math.exp(-1 / Math.max(1, (settings.releaseMs / 1000) * audio.sampleRate));
  const length = Math.max(...audio.channels.map((channel) => channel.length), 0);
  if (!length) return audio;

  // One detector drives every channel, so the stereo image does not shift.
  const gains = new Float32Array(length);
  let envelope = 0;
  for (let i = 0; i < length; i++) {
    let level = 0;
    for (const channel of audio.channels) level = Math.max(level, Math.abs(channel[i] ?? 0));
    envelope = level > envelope ? attack * envelope + (1 - attack) * level : release * envelope + (1 - release) * level;
    let gain = 1;
    if (envelope > threshold) {
      const over = envelope / threshold;
      gain = Math.pow(over, 1 / ratio - 1);
    }
    gains[i] = gain;
  }
  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((channel) => {
      const out = new Float32Array(channel.length);
      for (let i = 0; i < channel.length; i++) out[i] = channel[i] * gains[i];
      return out;
    }),
  };
}

function applyTrackGainAndPan(track: StudioTrack, audio: LocalAudio): LocalAudio {
  const gain = dbToGain(track.gainDb);
  const pan = clamp(track.pan, -1, 1);
  const left = gain * Math.min(1, 1 - Math.max(0, pan));
  const right = gain * Math.min(1, 1 + Math.min(0, pan));
  const channels: Float32Array[] = [];
  const source = audio.channels;
  const leftSource = source[0] ?? new Float32Array(0);
  const rightSource = source[1] ?? leftSource;
  for (let i = 0; i < 2; i++) {
    const src = i === 0 ? leftSource : rightSource;
    const out = new Float32Array(src.length);
    const scale = i === 0 ? left : right;
    for (let j = 0; j < src.length; j++) out[j] = src[j] * scale;
    channels.push(out);
  }
  return { sampleRate: audio.sampleRate, channels };
}

function applyVolumeAutomation(
  audio: LocalAudio,
  points: StudioTrack["automation"],
  rangeStart: number,
): LocalAudio {
  if (!points?.length) return audio;
  const sorted = [...points].sort((a, b) => a.time - b.time);
  const gainAt = (time: number): number => {
    if (time <= sorted[0].time) return dbToGain(sorted[0].gainDb);
    if (time >= sorted[sorted.length - 1].time) return dbToGain(sorted[sorted.length - 1].gainDb);
    for (let i = 1; i < sorted.length; i++) {
      const right = sorted[i];
      const left = sorted[i - 1];
      if (time <= right.time) {
        const fraction = (time - left.time) / Math.max(0.0001, right.time - left.time);
        return dbToGain(left.gainDb + (right.gainDb - left.gainDb) * fraction);
      }
    }
    return 1;
  };
  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((channel) => {
      const out = new Float32Array(channel.length);
      for (let i = 0; i < channel.length; i++) out[i] = channel[i] * gainAt(rangeStart + i / audio.sampleRate);
      return out;
    }),
  };
}

function slidingMax(values: Float32Array, window: number): Float32Array {
  const length = values.length;
  const out = new Float32Array(length);
  if (!length || window <= 0) return out;
  const left = new Float32Array(length);
  const right = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    left[i] = i % window === 0 ? values[i] : Math.max(left[i - 1], values[i]);
  }
  for (let i = length - 1; i >= 0; i--) {
    right[i] = i === length - 1 || (i + 1) % window === 0 ? values[i] : Math.max(right[i + 1], values[i]);
  }
  for (let i = 0; i < length; i++) {
    const end = Math.min(length - 1, i + window - 1);
    out[i] = Math.max(right[i], left[end]);
  }
  return out;
}

function lookaheadLimit(audio: LocalAudio, ceilingDb: number, lookaheadMs = 5, releaseMs = 90): LocalAudio {
  const ceiling = dbToGain(ceilingDb);
  const length = audio.channels[0]?.length ?? 0;
  if (!length) return audio;
  const sampleRate = audio.sampleRate;
  const look = Math.max(1, Math.round((lookaheadMs / 1000) * sampleRate));
  const linked = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    let peak = 0;
    for (const channel of audio.channels) peak = Math.max(peak, Math.abs(channel[i] ?? 0));
    linked[i] = peak;
  }
  const futurePeak = slidingMax(linked, look + 1);
  const release = Math.exp(-1 / Math.max(1, (releaseMs / 1000) * sampleRate));
  const gains = new Float32Array(length);
  let gain = 1;
  for (let i = 0; i < length; i++) {
    const target = futurePeak[i] > ceiling ? ceiling / futurePeak[i] : 1;
    gain = target < gain ? target : release * gain + (1 - release) * target;
    gains[i] = gain;
  }
  const limited: LocalAudio = {
    sampleRate,
    channels: audio.channels.map((channel) => {
      const out = new Float32Array(channel.length);
      for (let i = 0; i < channel.length; i++) out[i] = channel[i] * gains[i];
      return out;
    }),
  };
  // The lookahead handles sample peaks. A final micro-scale catches the rare
  // intersample peak without changing the limiting decision.
  let truePeak = 0;
  for (const channel of limited.channels) {
    for (let i = 0; i < channel.length - 1; i++) {
      truePeak = Math.max(truePeak, Math.abs(channel[i]), Math.abs((channel[i] + channel[i + 1]) / 2));
    }
  }
  if (truePeak > ceiling) {
    const scale = ceiling / truePeak;
    for (const channel of limited.channels) {
      for (let i = 0; i < channel.length; i++) channel[i] *= scale;
    }
  }
  return limited;
}

function applySidechain(target: LocalAudio, source: LocalAudio, amountDb: number): LocalAudio {
  const depth = 1 - dbToGain(-Math.max(0, Math.min(24, amountDb)));
  if (depth <= 0) return target;
  let peak = 0;
  for (const channel of source.channels) {
    for (let i = 0; i < channel.length; i++) peak = Math.max(peak, Math.abs(channel[i]));
  }
  if (peak <= 1e-9) return target;
  const length = target.channels[0]?.length ?? 0;
  const attack = Math.exp(-1 / Math.max(1, 0.005 * target.sampleRate));
  const release = Math.exp(-1 / Math.max(1, 0.12 * target.sampleRate));
  let envelope = 0;
  const gains = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    let level = 0;
    for (const channel of source.channels) level = Math.max(level, Math.abs(channel[i] ?? 0));
    const normalized = level / peak;
    envelope = normalized > envelope ? attack * envelope + (1 - attack) * normalized : release * envelope + (1 - release) * normalized;
    gains[i] = 1 - depth * Math.min(1, envelope);
  }
  return {
    sampleRate: target.sampleRate,
    channels: target.channels.map((channel) => {
      const out = new Float32Array(channel.length);
      for (let i = 0; i < channel.length; i++) out[i] = channel[i] * gains[i];
      return out;
    }),
  };
}

function emptyStereo(length: number, sr: number): LocalAudio {
  return { sampleRate: sr, channels: [new Float32Array(length), new Float32Array(length)] };
}

function midiFrequency(note: number): number {
  return 440 * Math.pow(2, (note - 69) / 12);
}

function mixMidiClip(
  target: LocalAudio,
  clip: StudioClip,
  rangeStart: number,
  rangeEnd: number,
): void {
  const sr = target.sampleRate;
  const sourceStart = Math.max(clip.start, rangeStart);
  const sourceEnd = Math.min(clip.start + clip.duration, rangeEnd);
  if (sourceEnd <= sourceStart) return;
  const clipGain = dbToGain(clip.gainDb);
  for (const note of clip.notes ?? []) {
    const noteStart = clip.start + note.start;
    const noteEnd = Math.min(clip.start + clip.duration, noteStart + note.duration);
    if (noteEnd <= rangeStart || noteStart >= rangeEnd) continue;
    const start = Math.max(rangeStart, noteStart);
    const end = Math.min(rangeEnd, noteEnd);
    const frequency = midiFrequency(note.midi);
    const velocity = Math.max(1, Math.min(127, note.velocity)) / 127;
    const attack = Math.min(0.01, note.duration * 0.2);
    const release = Math.min(0.08, note.duration * 0.3);
    const first = Math.round((start - rangeStart) * sr);
    const last = Math.min(target.channels[0].length, Math.round((end - rangeStart) * sr));
    for (let i = first; i < last; i++) {
      const timeline = rangeStart + i / sr;
      const local = timeline - noteStart;
      const fadeIn = attack > 0 ? Math.min(1, local / attack) : 1;
      const fadeOut = release > 0 ? Math.min(1, (noteEnd - timeline) / release) : 1;
      const clipFadeIn = clip.fadeIn > 0 ? Math.min(1, (timeline - clip.start) / clip.fadeIn) : 1;
      const clipFadeOut = clip.fadeOut > 0 ? Math.min(1, (clip.start + clip.duration - timeline) / clip.fadeOut) : 1;
      const envelope = velocity * Math.max(0, Math.min(1, fadeIn, fadeOut, clipFadeIn, clipFadeOut));
      const phase = 2 * Math.PI * frequency * local;
      const value = (Math.sin(phase) + 0.22 * Math.sin(phase * 2) + 0.08 * Math.sin(phase * 3)) * envelope * 0.18 * clipGain;
      target.channels[0][i] += value;
      target.channels[1][i] += value;
    }
  }
}

function mixClip(
  target: LocalAudio,
  asset: StudioAsset,
  clip: StudioClip,
  rangeStart: number,
  rangeEnd: number,
  trackGainDb: number,
): void {
  const sr = target.sampleRate;
  const source = asset.audio;
  const outputStart = Math.max(clip.start, rangeStart);
  const outputEnd = Math.min(clipEnd(clip), rangeEnd);
  if (outputEnd <= outputStart) return;

  const first = Math.round((outputStart - rangeStart) * sr);
  const last = Math.min(target.channels[0].length, Math.round((outputEnd - rangeStart) * sr));
  const clipGain = dbToGain(clip.gainDb) * dbToGain(trackGainDb);
  const fadeIn = Math.max(0, clip.fadeIn);
  const fadeOut = Math.max(0, clip.fadeOut);
  const sourceRate = source.sampleRate;

  for (let i = first; i < last; i++) {
    const timeline = rangeStart + i / sr;
    const local = timeline - clip.start;
    if (local < 0 || local > clip.duration) continue;
    let fade = 1;
    if (fadeIn > 0 && local < fadeIn) fade *= local / fadeIn;
    if (fadeOut > 0 && clip.duration - local < fadeOut) fade *= (clip.duration - local) / fadeOut;
    const position = (clip.offset + local) * sourceRate;
    const index = Math.floor(position);
    if (index < 0 || index >= (source.channels[0]?.length ?? 0)) continue;
    const fraction = position - index;
    const read = (channel: Float32Array) => {
      const a = channel[index] ?? 0;
      const b = channel[index + 1] ?? a;
      return a + (b - a) * fraction;
    };
    const left = read(source.channels[0]);
    const right = source.channels.length > 1 ? read(source.channels[1]) : left;
    target.channels[0][i] += left * clipGain * fade;
    target.channels[1][i] += right * clipGain * fade;
  }
}

export function renderProject(
  project: StudioProject,
  assets: Record<string, StudioAsset>,
  options: RenderOptions = {},
): RenderResult {
  const rangeStart = Math.max(0, options.start ?? 0);
  const naturalEnd = Math.max(rangeStart + 0.01, options.end ?? projectDuration(project));
  const duration = Math.max(0.01, naturalEnd - rangeStart);
  const sr = project.sampleRate;
  const length = Math.max(1, Math.round(duration * sr));
  const master = emptyStereo(length, sr);
  const soloActive = project.tracks.some((track) => track.solo);
  const only = options.onlyTrackIds ? new Set(options.onlyTrackIds) : null;

  const activeTracks = project.tracks.filter((track) => {
    if (only) return only.has(track.id);
    return !track.mute && (!soloActive || track.solo);
  });

  // Build each channel first so a sidechain can read another track's envelope.
  const renderedTracks = new Map<string, LocalAudio>();
  for (const track of activeTracks) {
    const trackBuffer = emptyStereo(length, sr);
    for (const clip of track.clips) {
      if (clip.kind === "midi") {
        mixMidiClip(trackBuffer, clip, rangeStart, naturalEnd);
        continue;
      }
      const asset = clip.assetId ? assets[clip.assetId] : undefined;
      if (!asset) continue;
      mixClip(trackBuffer, asset, clip, rangeStart, naturalEnd, 0);
    }
    const eq = applyChannelEq(trackBuffer, track.eq);
    const compressed = compress(eq, track.compressor);
    const panned = applyTrackGainAndPan(track, compressed);
    renderedTracks.set(track.id, applyVolumeAutomation(panned, track.automation, rangeStart));
  }

  for (const track of activeTracks) {
    let output = renderedTracks.get(track.id);
    if (!output) continue;
    const sidechain = track.sidechain;
    if (sidechain?.enabled && sidechain.sourceTrackId !== track.id) {
      const source = renderedTracks.get(sidechain.sourceTrackId);
      if (source) output = applySidechain(output, source, sidechain.amountDb);
    }
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < length; i++) master.channels[c][i] += output.channels[c][i];
    }
  }

  let output = master;
  if (options.master !== false) {
    const gain = dbToGain(project.master.gainDb);
    for (const channel of output.channels) {
      for (let i = 0; i < channel.length; i++) channel[i] *= gain;
    }
    if (project.master.limiter) {
      output = lookaheadLimit(output, project.master.ceilingDb);
    }
  }

  const report = measureLoudness(output);
  return { audio: output, report, duration };
}

export function renderStems(project: StudioProject, assets: Record<string, StudioAsset>): StemResult[] {
  return project.tracks.map((track) => ({
    trackId: track.id,
    name: track.name,
    audio: renderProject(project, assets, { onlyTrackIds: [track.id], master: false }).audio,
  }));
}
