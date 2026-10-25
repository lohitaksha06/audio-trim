/**
 * Tempo and brightness measurement for browser-only mode.
 *
 * The server uses librosa's beat tracker. The browser has no such library, so
 * this is a small onset-flux + autocorrelation estimator. It is deliberately
 * conservative: it returns `null` rather than guessing when there is no
 * periodic energy, because a wrong BPM would silently stretch a file to the
 * wrong tempo.
 */

import { fft } from "./fft";
import type { LocalAudio } from "./types";

export interface TempoEstimate {
  bpm: number;
  /** Normalised autocorrelation peak, 0..1. */
  confidence: number;
}

function mixToMono(audio: LocalAudio, maxSamples: number): Float32Array {
  const n = Math.min(audio.channels[0]?.length ?? 0, maxSamples);
  const mono = new Float32Array(n);
  if (n === 0) return mono;

  const channels = audio.channels.length;
  for (const channel of audio.channels) {
    for (let i = 0; i < n; i++) mono[i] += channel[i];
  }
  if (channels > 1) {
    const scale = 1 / channels;
    for (let i = 0; i < n; i++) mono[i] *= scale;
  }
  return mono;
}

/**
 * Estimate tempo from the onset envelope.
 *
 * Frames are 1024 samples with a 128-sample hop. The hop sets the lag
 * resolution: at 44.1 kHz one frame is 2.9 ms, so 120 BPM (500 ms) lands on a
 * lag of ~172 frames, and parabolic interpolation refines the fractional part.
 */
export function estimateTempo(audio: LocalAudio, maxSeconds = 60): TempoEstimate | null {
  const sr = audio.sampleRate;
  const maxSamples = Math.max(0, Math.round(maxSeconds * sr));
  const mono = mixToMono(audio, maxSamples);
  const nFft = 1024;
  const hop = 128;
  if (mono.length < sr * 2 || mono.length < nFft * 2) return null;

  const frames = 1 + Math.floor((mono.length - nFft) / hop);
  if (frames < 16) return null;

  const re = new Float32Array(nFft);
  const im = new Float32Array(nFft);
  const previous = new Float32Array(nFft / 2);
  const flux = new Float32Array(frames);
  const binLimit = Math.min(nFft / 2, Math.ceil((5000 * nFft) / sr));
  const window = new Float32Array(nFft);
  for (let i = 0; i < nFft; i++) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft);
  }

  let maxFlux = 0;
  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    for (let i = 0; i < nFft; i++) {
      re[i] = mono[start + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);

    let sum = 0;
    for (let b = 1; b < binLimit; b++) {
      const magnitude = Math.hypot(re[b], im[b]);
      const delta = magnitude - previous[b];
      if (delta > 0) sum += delta;
      previous[b] = magnitude;
    }
    flux[f] = sum;
    if (sum > maxFlux) maxFlux = sum;
  }

  // A flat or silent signal has no onsets. Returning null here is what stops
  // "What BPM is this?" from inventing a number for a pad or a sine tone.
  if (maxFlux <= 1e-9) return null;

  let mean = 0;
  for (let i = 0; i < frames; i++) mean += flux[i];
  mean /= frames;
  for (let i = 0; i < frames; i++) flux[i] = Math.max(0, flux[i] - mean);

  const minBpm = 50;
  const maxBpm = 200;
  const minLag = Math.max(1, Math.floor((60 * sr) / (maxBpm * hop)));
  const maxLag = Math.min(frames - 2, Math.ceil((60 * sr) / (minBpm * hop)));
  if (maxLag <= minLag + 2) return null;

  // Normalised autocorrelation. Computing up to 2 * maxLag gives the harmonic
  // term below something to read without a second pass.
  const acLimit = Math.min(frames - 2, maxLag * 2);
  const ac = new Float32Array(acLimit + 1);
  for (let lag = 1; lag <= acLimit; lag++) {
    let numerator = 0;
    let left = 0;
    let right = 0;
    for (let i = 0; i + lag < frames; i++) {
      numerator += flux[i] * flux[i + lag];
      left += flux[i] * flux[i];
      right += flux[i + lag] * flux[i + lag];
    }
    const denominator = Math.sqrt(left * right);
    ac[lag] = denominator > 1e-12 ? numerator / denominator : 0;
  }

  let bestLag = 0;
  let bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    // Reward the two-beat lag as well. A sub-beat bonus was tried and removed:
    // it made the double lag score higher than the true beat on a plain click
    // track, so 120 BPM read as 60 BPM. The octave term alone still favours a
    // real beat over a strong eighth-note pattern without that bias.
    const score = ac[lag] + 0.5 * (lag * 2 <= acLimit ? ac[lag * 2] : 0);
    if (score > bestScore) {
      bestScore = score;
      bestLag = lag;
    }
  }
  if (bestLag === 0 || bestScore < 0.12) return null;

  // Parabolic interpolation around the integer lag. Without this, 120 BPM at
  // 22.05 kHz can read as 120.2 or 118.8 depending on which frame it lands on.
  let refinedLag = bestLag;
  if (bestLag > 1 && bestLag < acLimit) {
    const left = ac[bestLag - 1];
    const centre = ac[bestLag];
    const right = ac[bestLag + 1];
    const denominator = left - 2 * centre + right;
    if (Math.abs(denominator) > 1e-9) {
      const delta = (0.5 * (left - right)) / denominator;
      if (Number.isFinite(delta) && Math.abs(delta) < 1) refinedLag = bestLag + delta;
    }
  }

  const bpm = (60 * sr) / (refinedLag * hop);
  if (!Number.isFinite(bpm) || bpm < 40 || bpm > 240) return null;

  return {
    bpm,
    confidence: Math.max(0, Math.min(1, bestScore)),
  };
}

/**
 * Average spectral centroid over the first `maxSeconds`.
 *
 * Used only to pick a starting preset in the Lab. It is a measured descriptor,
 * not a mood classifier; the UI says "brightness", not "emotion".
 */
export function estimateBrightness(audio: LocalAudio, maxSeconds = 20): number {
  const sr = audio.sampleRate;
  const mono = mixToMono(audio, Math.max(0, Math.round(maxSeconds * sr)));
  const nFft = 1024;
  const hop = 512;
  if (mono.length < nFft) return 0;

  const frames = 1 + Math.floor((mono.length - nFft) / hop);
  const re = new Float32Array(nFft);
  const im = new Float32Array(nFft);
  const window = new Float32Array(nFft);
  for (let i = 0; i < nFft; i++) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft);
  }

  let weighted = 0;
  let total = 0;
  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    for (let i = 0; i < nFft; i++) {
      re[i] = mono[start + i] * window[i];
      im[i] = 0;
    }
    fft(re, im);

    for (let b = 1; b < nFft / 2; b++) {
      const magnitude = Math.hypot(re[b], im[b]);
      weighted += ((b * sr) / nFft) * magnitude;
      total += magnitude;
    }
  }
  return total > 1e-9 ? weighted / total : 0;
}
