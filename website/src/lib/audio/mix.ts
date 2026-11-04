/** Local mixing helpers: a second stem, and 4-band envelopes for the mixer. */

import { cloneAudio } from "./ops";
import type { LocalAudio } from "./types";
import { peakOf } from "./wav";

export type StemBand = "vocals" | "drums" | "bass" | "other";

export interface BandPeaks {
  vocals: Float32Array;
  drums: Float32Array;
  bass: Float32Array;
  other: Float32Array;
}

function resampleChannel(input: Float32Array, from: number, to: number): Float32Array {
  if (from === to || input.length === 0) return new Float32Array(input);
  const ratio = to / from;
  const length = Math.max(1, Math.round(input.length * ratio));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const source = i / ratio;
    const i0 = Math.floor(source);
    const i1 = Math.min(input.length - 1, i0 + 1);
    const frac = source - i0;
    out[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return out;
}

function fitOverlay(overlay: LocalAudio, sampleRate: number, length: number): Float32Array[] {
  const channels = overlay.channels.map((c) =>
    overlay.sampleRate === sampleRate ? new Float32Array(c) : resampleChannel(c, overlay.sampleRate, sampleRate),
  );
  return channels.map((c) => {
    if (c.length === length) return c;
    const out = new Float32Array(length);
    out.set(c.subarray(0, Math.min(c.length, length)));
    return out;
  });
}

/**
 * Sum a second file into the base track (a simple stem import).
 *
 * This is a real audio mix, not a separator: the imported file is resampled to
 * the base rate, padded or cut to the base length, and added at `levelDb`.
 * Tempo/key matching lives on the server.
 */
export function mixAudio(
  base: LocalAudio,
  overlay: LocalAudio,
  levelDb: number,
): { audio: LocalAudio; appliedDb: number; mixedSeconds: number } {
  const length = base.channels[0]?.length ?? 0;
  const fitted = fitOverlay(overlay, base.sampleRate, length);
  const out = cloneAudio(base);
  const gain = Math.pow(10, Math.max(-36, Math.min(12, levelDb)) / 20);

  if (fitted.length === 1 && out.channels.length === 2) {
    for (let c = 0; c < out.channels.length; c++) {
      const channel = out.channels[c];
      const source = fitted[0];
      for (let i = 0; i < channel.length; i++) channel[i] += source[i] * gain;
    }
  } else if (fitted.length >= 2 && out.channels.length === 1) {
    const channel = out.channels[0];
    const left = fitted[0];
    const right = fitted[1];
    for (let i = 0; i < channel.length; i++) channel[i] += ((left[i] + right[i]) * 0.5) * gain;
  } else {
    for (let c = 0; c < out.channels.length; c++) {
      const channel = out.channels[c];
      const source = fitted[Math.min(c, fitted.length - 1)];
      for (let i = 0; i < channel.length; i++) channel[i] += source[i] * gain;
    }
  }

  const peak = peakOf(out);
  if (peak > 0.98) {
    const scale = 0.98 / peak;
    for (const channel of out.channels) {
      for (let i = 0; i < channel.length; i++) channel[i] *= scale;
    }
  }

  return {
    audio: out,
    appliedDb: levelDb,
    mixedSeconds: Math.min(length, fitted[0]?.length ?? 0) / base.sampleRate,
  };
}

function lowpassSample(x: number, state: number, coefficient: number): number {
  return coefficient * state + (1 - coefficient) * x;
}

/**
 * Four approximate envelopes for the fader lanes.
 *
 * One pass, one-pole filters: low = bass, low-mid = drums, 2-6 kHz = vocals,
 * the rest = other. They are visual guides, not isolated stems.
 */
export function bandPeaks(audio: LocalAudio, buckets = 160): BandPeaks {
  const data = audio.channels[0] ?? new Float32Array(0);
  const sr = audio.sampleRate || 44100;
  const size = Math.max(8, buckets);
  const out: BandPeaks = {
    vocals: new Float32Array(size),
    drums: new Float32Array(size),
    bass: new Float32Array(size),
    other: new Float32Array(size),
  };
  if (!data.length) return out;

  const a180 = Math.exp((-2 * Math.PI * 180) / sr);
  const a2000 = Math.exp((-2 * Math.PI * 2000) / sr);
  const a6000 = Math.exp((-2 * Math.PI * 6000) / sr);
  let low = 0;
  let lowMid = 0;
  let presence = 0;
  const block = Math.max(1, Math.floor(data.length / size));

  for (let i = 0; i < data.length; i++) {
    const x = data[i];
    low = lowpassSample(x, low, a180);
    const rest = x - low;
    lowMid = lowpassSample(rest, lowMid, a2000);
    const high = rest - lowMid;
    presence = lowpassSample(high, presence, a6000);
    const air = high - presence;
    const bucket = Math.min(size - 1, Math.floor(i / block));
    const drum = lowMid + low * 0.45;
    out.bass[bucket] = Math.max(out.bass[bucket], Math.abs(low));
    out.drums[bucket] = Math.max(out.drums[bucket], Math.abs(drum));
    out.vocals[bucket] = Math.max(out.vocals[bucket], Math.abs(presence));
    out.other[bucket] = Math.max(out.other[bucket], Math.abs(air));
  }

  for (const key of Object.keys(out) as StemBand[]) {
    let max = 0;
    for (let i = 0; i < out[key].length; i++) max = Math.max(max, out[key][i]);
    if (max > 1e-6) {
      for (let i = 0; i < out[key].length; i++) out[key][i] /= max;
    }
  }
  return out;
}

/** Map the four faders onto the existing 5-band EQ renderer. */
export function stemGainsToEq(gains: Record<StemBand, number>): Record<string, number> {
  const bass = gains.bass ?? 0;
  const drums = gains.drums ?? 0;
  const vocals = gains.vocals ?? 0;
  const other = gains.other ?? 0;
  return {
    "60": bass + drums * 0.35,
    "230": bass * 0.7 + drums,
    "910": drums * 0.35 + vocals,
    "4000": vocals + drums * 0.3,
    "14000": other,
  };
}
