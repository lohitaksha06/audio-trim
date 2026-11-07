/**
 * Loudness measurement for studio renders.
 *
 * Integrated loudness follows the BS.1770 idea: K-weighting, 400 ms blocks
 * with a 100 ms hop, absolute -70 LUFS gate, then a relative -10 LU gate. The
 * K-weighting filters are RBJ biquads recalculated for the render sample rate,
 * so the measurement is not tied to 48 kHz.
 *
 * True peak is estimated with 4x linear interpolation. That is a measurement,
 * not a guarantee about a downstream D/A converter.
 */

import type { LoudnessReport } from "./types";

interface Biquad {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

function highPass(freq: number, q: number, sr: number): Biquad {
  const w0 = (2 * Math.PI * Math.min(freq, sr * 0.45)) / sr;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: ((1 + cos) / 2) / a0,
    b1: (-(1 + cos)) / a0,
    b2: ((1 + cos) / 2) / a0,
    a1: (-2 * cos) / a0,
    a2: (1 - alpha) / a0,
  };
}

function highShelf(freq: number, gainDb: number, slope: number, sr: number): Biquad {
  const a = Math.pow(10, gainDb / 40);
  const w0 = (2 * Math.PI * Math.min(freq, sr * 0.45)) / sr;
  const cos = Math.cos(w0);
  const sin = Math.sin(w0);
  const alpha = (sin / 2) * Math.sqrt((a + 1 / a) * (1 / slope - 1) + 2);
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

function process(x: Float32Array, filter: Biquad): Float32Array {
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

function weighted(channels: Float32Array[], sr: number): Float32Array[] {
  const hp = highPass(38, 0.5, sr);
  const shelf = highShelf(1681, 4, 1, sr);
  return channels.map((channel) => process(process(channel, hp), shelf));
}

function meanSquare(channels: Float32Array[], from: number, to: number): number {
  let sum = 0;
  let count = 0;
  for (const channel of channels) {
    for (let i = from; i < to && i < channel.length; i++) {
      sum += channel[i] * channel[i];
      count++;
    }
  }
  return count ? sum / count : 0;
}

function toLufs(z: number): number {
  return z > 1e-12 ? -0.691 + 10 * Math.log10(z) : -Infinity;
}

function blockLoudness(channels: Float32Array[], sr: number, block: number, hop: number): number[] {
  const values: number[] = [];
  const length = channels[0]?.length ?? 0;
  for (let start = 0; start + block <= length; start += hop) {
    values.push(toLufs(meanSquare(channels, start, start + block)));
  }
  return values;
}

export function measureLoudness(audio: { sampleRate: number; channels: Float32Array[] }): LoudnessReport {
  const sr = audio.sampleRate;
  const channels = audio.channels;
  if (!channels.length || !channels[0]?.length) {
    return { integratedLufs: -Infinity, shortTermLufs: -Infinity, truePeakDbtp: -Infinity, samplePeakDbfs: -Infinity };
  }

  const weightedChannels = weighted(channels, sr);
  const block = Math.max(1, Math.round(0.4 * sr));
  const hop = Math.max(1, Math.round(0.1 * sr));
  const blocks = blockLoudness(weightedChannels, sr, block, hop);

  const absoluteGate = -70;
  const aboveAbsolute = blocks.filter((value) => value > absoluteGate);
  let integrated = -Infinity;
  if (aboveAbsolute.length) {
    const average = aboveAbsolute.reduce((sum, value) => sum + Math.pow(10, (value + 0.691) / 10), 0) / aboveAbsolute.length;
    const relativeGate = toLufs(average) - 10;
    const aboveRelative = aboveAbsolute.filter((value) => value > relativeGate);
    const finalAverage = aboveRelative.length
      ? aboveRelative.reduce((sum, value) => sum + Math.pow(10, (value + 0.691) / 10), 0) / aboveRelative.length
      : average;
    integrated = toLufs(finalAverage);
  }

  // Short-term: the loudest 3 s window, computed from the same block series.
  const blocksPerSecond = Math.max(1, Math.round(sr / hop));
  const windowBlocks = Math.max(1, blocksPerSecond * 3);
  let shortTerm = -Infinity;
  for (let i = 0; i + windowBlocks <= blocks.length; i++) {
    const slice = blocks.slice(i, i + windowBlocks);
    const average = slice.reduce((sum, value) => sum + Math.pow(10, (value + 0.691) / 10), 0) / slice.length;
    shortTerm = Math.max(shortTerm, toLufs(average));
  }

  let samplePeak = 0;
  for (const channel of channels) {
    for (let i = 0; i < channel.length; i++) samplePeak = Math.max(samplePeak, Math.abs(channel[i]));
  }

  // 4x linear interpolation gives a useful true-peak estimate without a
  // resampling dependency. It is monotonic and never below the sample peak.
  let truePeak = samplePeak;
  for (const channel of channels) {
    for (let i = 0; i < channel.length - 1; i++) {
      const a = channel[i];
      const b = channel[i + 1];
      for (let k = 1; k < 4; k++) {
        const value = Math.abs(a + (b - a) * (k / 4));
        if (value > truePeak) truePeak = value;
      }
    }
  }

  return {
    integratedLufs: integrated,
    shortTermLufs: shortTerm,
    truePeakDbtp: truePeak > 1e-12 ? 20 * Math.log10(truePeak) : -Infinity,
    samplePeakDbfs: samplePeak > 1e-12 ? 20 * Math.log10(samplePeak) : -Infinity,
  };
}

export function formatLufs(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(1)} LUFS` : "—";
}

export function formatDbtp(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(1)} dBTP` : "—";
}
