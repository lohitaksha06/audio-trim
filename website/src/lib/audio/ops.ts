/**
 * Time-domain and spectral edits that run in the browser.
 *
 * These mirror the server's behaviour where that matters (peak normalisation to
 * 0.9 after a boost, pitch-preserving speed changes) and are explicit where it
 * cannot (the browser has no stem separator or instrument generator).
 */

import { denoiseChannel } from "./denoise";
import { fft, ifft } from "./fft";
import { applySpectralGain, DEFAULT_HOP, DEFAULT_N_FFT, hann } from "./stft";
import type { LocalAudio } from "./types";
import { peakOf } from "./wav";

export function cloneAudio(audio: LocalAudio): LocalAudio {
  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((c) => new Float32Array(c)),
  };
}

/**
 * Join segments with a short crossfade at each boundary.
 *
 * Used by remove-section and loop. The crossfade is linear and removes
 * `crossfadeSeconds` from the total length per join, which is why the tests
 * allow a few milliseconds of tolerance around the nominal duration.
 */
export function concatAudio(segments: LocalAudio[], crossfadeSeconds = 0.01): LocalAudio {
  const valid = segments.filter((segment) => (segment.channels[0]?.length ?? 0) > 0);
  if (valid.length === 0) {
    const sampleRate = segments[0]?.sampleRate ?? 44100;
    return { sampleRate, channels: [new Float32Array(0)] };
  }
  if (valid.length === 1) return cloneAudio(valid[0]);

  const sampleRate = valid[0].sampleRate;
  const maxCrossfade = Math.max(0, Math.round(crossfadeSeconds * sampleRate));
  const overlaps = valid.slice(1).map((segment, index) =>
    Math.min(
      maxCrossfade,
      Math.floor(valid[index].channels[0].length / 2),
      Math.floor(segment.channels[0].length / 2),
    ),
  );
  const outLength =
    valid.reduce((sum, segment) => sum + segment.channels[0].length, 0) -
    overlaps.reduce((sum, overlap) => sum + overlap, 0);

  const channels = valid[0].channels.map((_, channelIndex) => {
    const out = new Float32Array(outLength);
    out.set(valid[0].channels[channelIndex], 0);
    let position = valid[0].channels[channelIndex].length;

    for (let i = 1; i < valid.length; i++) {
      const segment = valid[i].channels[channelIndex];
      const overlap = overlaps[i - 1];
      if (overlap > 0) {
        const start = position - overlap;
        for (let j = 0; j < overlap; j++) {
          const mix = (j + 1) / (overlap + 1);
          out[start + j] = out[start + j] * (1 - mix) + segment[j] * mix;
        }
        out.set(segment.subarray(overlap), position);
        position += segment.length - overlap;
      } else {
        out.set(segment, position);
        position += segment.length;
      }
    }
    return out;
  });

  return { sampleRate, channels };
}

export function trim(audio: LocalAudio, startSec: number, endSec: number): LocalAudio {
  const sr = audio.sampleRate;
  const start = Math.max(0, Math.min(audio.channels[0].length, Math.round(startSec * sr)));
  const end = Math.max(start, Math.min(audio.channels[0].length, Math.round(endSec * sr)));
  return {
    sampleRate: sr,
    channels: audio.channels.map((c) => c.slice(start, end)),
  };
}

/** Raised-cosine fades. `inSec` and `outSec` are clamped to the duration. */
export function fade(audio: LocalAudio, inSec: number, outSec: number): LocalAudio {
  const sr = audio.sampleRate;
  const n = audio.channels[0]?.length ?? 0;
  const inN = Math.min(n, Math.max(0, Math.round(inSec * sr)));
  const outN = Math.min(n, Math.max(0, Math.round(outSec * sr)));
  const out = cloneAudio(audio);
  for (const channel of out.channels) {
    for (let i = 0; i < inN; i++) {
      channel[i] *= 0.5 - 0.5 * Math.cos((Math.PI * i) / inN);
    }
    for (let i = 0; i < outN; i++) {
      const idx = n - 1 - i;
      channel[idx] *= 0.5 - 0.5 * Math.cos((Math.PI * i) / outN);
    }
  }
  return out;
}

export interface GainResult {
  audio: LocalAudio;
  /** What the file actually got after the anti-clip safety net. */
  appliedDb: number;
}

export function applyGain(audio: LocalAudio, db: number): GainResult {
  const factor = Math.pow(10, db / 20);
  const out = cloneAudio(audio);
  for (const channel of out.channels) {
    for (let i = 0; i < channel.length; i++) channel[i] *= factor;
  }
  // A +12 dB request on an already-hot mix must not wrap or hard-clip, so pull
  // the whole file back instead. Report the number that was actually applied.
  const peak = peakOf(out);
  if (peak > 0.999) {
    const k = 0.99 / peak;
    for (const channel of out.channels) {
      for (let i = 0; i < channel.length; i++) channel[i] *= k;
    }
    return { audio: out, appliedDb: db + 20 * Math.log10(k) };
  }
  return { audio: out, appliedDb: db };
}

/** Peak-normalise to `peakDb` (default -1 dBFS, like the server's normaliser). */
export function normalize(audio: LocalAudio, peakDb = -1): { audio: LocalAudio; gainDb: number } {
  const peak = peakOf(audio);
  if (peak <= 1e-9) return { audio: cloneAudio(audio), gainDb: 0 };
  const target = Math.pow(10, peakDb / 20);
  const gain = target / peak;
  const out = cloneAudio(audio);
  for (const channel of out.channels) {
    for (let i = 0; i < channel.length; i++) channel[i] *= gain;
  }
  return { audio: out, gainDb: 20 * Math.log10(gain) };
}

export function reverse(audio: LocalAudio): LocalAudio {
  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((c) => {
      const out = new Float32Array(c.length);
      for (let i = 0; i < c.length; i++) out[i] = c[c.length - 1 - i];
      return out;
    }),
  };
}

/**
 * Pitch-preserving time stretch (phase vocoder).
 *
 * `factor > 1` is longer/slower, `< 1` is shorter/faster. Speed and stretch are
 * reciprocals at the call site.
 *
 * A plain resample would change pitch, which is not what "speed this up" means
 * for speech or music, so the phase advance is integrated per bin instead.
 */
export function timeStretch(
  x: Float32Array,
  factor: number,
  nFft: number = DEFAULT_N_FFT,
  hop: number = DEFAULT_HOP,
  onProgress?: (fraction: number) => void,
): Float32Array {
  if (x.length === 0 || Math.abs(factor - 1) < 1e-6) return new Float32Array(x);

  const win = hann(nFft);
  const bins = (nFft >> 1) + 1;
  const pad = nFft >> 1;
  const total = x.length + 2 * pad;
  const frames = 1 + Math.floor((total - nFft) / hop);
  onProgress?.(0);
  const synHop = Math.max(1, Math.round(hop * factor));
  const ratio = synHop / hop;
  const outLength = Math.max(1, Math.round(x.length * factor));
  const out = new Float32Array(outLength + nFft * 4);
  const wsum = new Float32Array(out.length);

  const fr = new Float32Array(nFft);
  const fi = new Float32Array(nFft);
  const prevPhase = new Float32Array(bins);
  const synthPhase = new Float32Array(bins);
  const omega = new Float32Array(bins);
  for (let b = 0; b < bins; b++) omega[b] = (2 * Math.PI * b * hop) / nFft;

  for (let f = 0; f < frames; f++) {
    const start = f * hop - pad;
    for (let i = 0; i < nFft; i++) {
      const src = start + i;
      fr[i] = src >= 0 && src < x.length ? x[src] * win[i] : 0;
      fi[i] = 0;
    }
    fft(fr, fi);

    for (let b = 0; b < bins; b++) {
      const mag = Math.hypot(fr[b], fi[b]);
      const p = Math.atan2(fi[b], fr[b]);
      if (f === 0) {
        synthPhase[b] = p;
      } else {
        let delta = p - prevPhase[b] - omega[b];
        delta -= 2 * Math.PI * Math.round(delta / (2 * Math.PI));
        const trueOmega = omega[b] + delta;
        synthPhase[b] += trueOmega * ratio;
      }
      prevPhase[b] = p;
      fr[b] = mag * Math.cos(synthPhase[b]);
      fi[b] = mag * Math.sin(synthPhase[b]);
    }
    for (let b = bins; b < nFft; b++) {
      fr[b] = fr[nFft - b];
      fi[b] = -fi[nFft - b];
    }
    ifft(fr, fi);

    const at = f * synHop;
    for (let i = 0; i < nFft; i++) {
      out[at + i] += fr[i] * win[i];
      wsum[at + i] += win[i] * win[i];
    }
    if (onProgress && (f & 63) === 0) onProgress(f / frames);
  }
  onProgress?.(1);

  const offset = Math.round(pad * ratio);
  const y = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const w = wsum[offset + i];
    y[i] = w > 1e-8 ? out[offset + i] / w : 0;
  }
  return y;
}

export function stretchAudio(
  audio: LocalAudio,
  factor: number,
  onProgress?: (fraction: number) => void,
): LocalAudio {
  if (Math.abs(factor - 1) < 1e-6) return cloneAudio(audio);
  const count = audio.channels.length;
  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((c, i) =>
      timeStretch(c, factor, DEFAULT_N_FFT, DEFAULT_HOP, onProgress ? (f) => onProgress((i + f) / count) : undefined),
    ),
  };
}

type BoostTarget = "vocals" | "drums" | "bass" | "other";

/** Raised-cosine ramp so the shelf does not ring at its edges. */
function ramp(freq: number, lo: number, hi: number, width: number): number {
  if (freq <= lo || freq >= hi) return 0;
  const up = Math.min(1, (freq - lo) / width);
  const down = Math.min(1, (hi - freq) / width);
  return 0.5 - 0.5 * Math.cos(Math.PI * Math.min(up, down));
}

/**
 * Turn up one element, matching the server's `_boost_target` bands:
 * vocals = presence 1-6 kHz, drums = sub 200 Hz + snap 2-4 kHz, bass = 250 Hz
 * down. Peak-normalised to 0.9 afterwards, same as the server.
 */
export function boostTarget(
  audio: LocalAudio,
  target: BoostTarget,
  direction: "up" | "down" = "up",
  onProgress?: (fraction: number) => void,
): LocalAudio {
  const sr = audio.sampleRate;
  const up = direction === "up";
  const count = audio.channels.length;
  // Shelf amounts: the "up" numbers are 1.7x / 1.9x / 1.5x / 1.8x, matching the
  // server's `_boost_target`; the cuts are gentler so a negative shelf does not
  // hollow out the band.
  const channels = audio.channels.map((channel, index) => {
    const out = applySpectralGain(channel, DEFAULT_N_FFT, DEFAULT_HOP, (_mag, bins, frames) => {
      const gain = new Float32Array(bins * frames);
      for (let b = 0; b < bins; b++) {
        const freq = (b * sr) / DEFAULT_N_FFT;
        let weight = 0;
        if (target === "vocals" || target === "other") {
          weight = (up ? 0.7 : 0.5) * ramp(freq, 1000, 6000, 200);
        } else if (target === "drums") {
          weight = (up ? 0.9 : 0.56) * ramp(freq, 20, 200, 60) + (up ? 0.5 : 0.29) * ramp(freq, 2000, 4000, 250);
        } else if (target === "bass") {
          weight = (up ? 0.8 : 0.5) * ramp(freq, 20, 250, 70);
        }
        const g = up ? 1 + weight : 1 - weight;
        const base = b * frames;
        for (let f = 0; f < frames; f++) gain[base + f] = g;
      }
      return gain;
    }, onProgress ? (fraction) => onProgress((index + fraction) / count) : undefined);
    return out;
  });

  const boosted: LocalAudio = { sampleRate: sr, channels };
  const peak = peakOf(boosted);
  if (peak > 0.9) {
    const k = 0.9 / peak;
    for (const channel of boosted.channels) {
      for (let i = 0; i < channel.length; i++) channel[i] *= k;
    }
  }
  return boosted;
}

export type ToneKind = "darker" | "brighter" | "calm" | "energetic" | "lofi";

/** Fraction of the way from `corner` up to `corner + width` (0 below the corner). */
function highShelf(freq: number, corner: number, width: number): number {
  if (freq <= corner) return 0;
  return Math.min(1, (freq - corner) / width);
}

/** Fraction of the way down from `corner` (0 above the corner). */
function lowShelf(freq: number, corner: number, width: number): number {
  if (freq >= corner) return 0;
  return Math.min(1, (corner - freq) / width);
}

/**
 * Mood shaping: a static spectral tilt plus a level change.
 *
 * These are the browser-only stand-ins for the server's mood renderer. They do
 * not invent instruments: "darker" rolls the highs off, "brighter" lifts them,
 * "energetic" adds level and presence, "calm" softens both. The result is
 * peak-limited so a lift cannot clip.
 */
export function shapeTone(
  audio: LocalAudio,
  kind: ToneKind,
  onProgress?: (fraction: number) => void,
): LocalAudio {
  const sr = audio.sampleRate;
  const count = audio.channels.length;

  const gainFor = (freq: number): number => {
    switch (kind) {
      case "darker":
        return (1 - 0.75 * highShelf(freq, 2000, 1500)) * (1 + 0.08 * lowShelf(freq, 250, 200));
      case "brighter":
        return (1 + 0.95 * highShelf(freq, 2500, 1500)) * (1 - 0.08 * lowShelf(freq, 120, 80));
      case "calm":
        return (1 - 0.5 * highShelf(freq, 2200, 1400)) * 0.78;
      case "energetic":
        return (1 + 0.5 * highShelf(freq, 2500, 1500)) * 1.3;
      case "lofi":
        return (1 - 0.85 * highShelf(freq, 3000, 1200)) * (1 + 0.12 * lowShelf(freq, 250, 150));
    }
  };

  const channels = audio.channels.map((channel, index) =>
    applySpectralGain(channel, DEFAULT_N_FFT, DEFAULT_HOP, (_mag, bins, frames) => {
      const gain = new Float32Array(bins * frames);
      for (let b = 0; b < bins; b++) {
        const g = gainFor((b * sr) / DEFAULT_N_FFT);
        const base = b * frames;
        for (let f = 0; f < frames; f++) gain[base + f] = g;
      }
      return gain;
    }, onProgress ? (fraction) => onProgress((index + fraction) / count) : undefined),
  );

  const out: LocalAudio = { sampleRate: sr, channels };
  const peak = peakOf(out);
  if (peak > 0.98) {
    const k = 0.98 / peak;
    for (const channel of out.channels) {
      for (let i = 0; i < channel.length; i++) channel[i] *= k;
    }
  }
  return out;
}

/** One-pole low-pass used to damp the reverb tail (darkens with each pass). */
function damped(value: number, state: number, damping: number): number {
  return value * (1 - damping) + state * damping;
}

/**
 * Small Schroeder reverb: four parallel feedback combs into two allpasses.
 *
 * The tail decays to −60 dB after `seconds`, and is mixed back under the dry
 * signal. It is deliberately modest: this is room/space, not a plate preset.
 */
export function reverbAudio(audio: LocalAudio, seconds = 2.2, mix = 0.25): LocalAudio {
  const sr = audio.sampleRate;
  const combDelays = [0.0297, 0.0371, 0.0411, 0.0437];
  const allpassDelays = [0.005, 0.0017];
  const wet = Math.max(0, Math.min(1, mix));
  const decay = Math.max(0.2, Math.min(8, seconds));

  return {
    sampleRate: sr,
    channels: audio.channels.map((channel, ch) => {
      const n = channel.length;
      const wetSignal = new Float32Array(n);
      // Slight delay offset per channel gives the tail a stereo width.
      const spread = ch === 0 ? 1 : 0.93;

      for (const secondsDelay of combDelays) {
        const delay = Math.max(1, Math.round(secondsDelay * spread * sr));
        const buffer = new Float32Array(delay);
        // gain per pass so the tail falls 60 dB in `decay` seconds
        const feedback = Math.pow(0.001, delay / (decay * sr));
        let index = 0;
        let state = 0;
        for (let i = 0; i < n; i++) {
          const delayed = buffer[index];
          state = damped(delayed, state, 0.55);
          buffer[index] = channel[i] + state * feedback;
          wetSignal[i] += delayed * 0.25;
          index = (index + 1) % delay;
        }
      }

      for (const allpassDelay of allpassDelays) {
        const delay = Math.max(1, Math.round(allpassDelay * sr));
        const buffer = new Float32Array(delay);
        const g = 0.5;
        let index = 0;
        for (let i = 0; i < n; i++) {
          const delayed = buffer[index];
          const out = -g * wetSignal[i] + delayed;
          buffer[index] = wetSignal[i] + g * out;
          wetSignal[i] = out;
          index = (index + 1) % delay;
        }
      }

      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        out[i] = channel[i] * (1 - wet) + wetSignal[i] * wet;
      }
      return out;
    }),
  };
}

export function denoiseAudio(
  audio: LocalAudio,
  strength = 2,
  onProgress?: (fraction: number) => void,
): LocalAudio {
  const count = audio.channels.length;
  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((c, i) =>
      denoiseChannel(
        c,
        audio.sampleRate,
        strength,
        DEFAULT_N_FFT,
        DEFAULT_HOP,
        onProgress ? (f) => onProgress((i + f) / count) : undefined,
      ),
    ),
  };
}

interface Biquad {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

/** RBJ cookbook peaking EQ, normalised so a0 = 1. */
function peaking(freq: number, q: number, gainDb: number, sr: number): Biquad {
  const w0 = (2 * Math.PI * Math.min(freq, sr * 0.45)) / sr;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a = Math.pow(10, gainDb / 40);
  const a0 = 1 + alpha / a;
  return {
    b0: (1 + alpha * a) / a0,
    b1: (-2 * cos) / a0,
    b2: (1 - alpha * a) / a0,
    a1: (-2 * cos) / a0,
    a2: (1 - alpha / a) / a0,
  };
}

function lowpass(freq: number, q: number, sr: number): Biquad {
  const w0 = (2 * Math.PI * Math.min(freq, sr * 0.45)) / sr;
  const cos = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: ((1 - cos) / 2) / a0,
    b1: (1 - cos) / a0,
    b2: ((1 - cos) / 2) / a0,
    a1: (-2 * cos) / a0,
    a2: (1 - alpha) / a0,
  };
}

function highpass(freq: number, q: number, sr: number): Biquad {
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

function applyBiquad(x: Float32Array, filter: Biquad): Float32Array {
  const y = new Float32Array(x.length);
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const value = filter.b0 * x[i] + filter.b1 * x1 + filter.b2 * x2 - filter.a1 * y1 - filter.a2 * y2;
    x2 = x1;
    x1 = x[i];
    y2 = y1;
    y1 = value;
    y[i] = value;
  }
  return y;
}

/**
 * 5-band peaking EQ followed by optional high-pass and low-pass filters.
 *
 * This is the same signal path the Lab preview builds with WebAudio biquads, so
 * what you hear is what gets rendered. Frequencies above Nyquist are clamped
 * rather than throwing (a 14 kHz band on a 22.05 kHz file is common).
 */
export function applyEq(
  audio: LocalAudio,
  gainsDb: Record<string, number>,
  highpassHz: number | null = null,
  lowpassHz: number | null = null,
  onProgress?: (fraction: number) => void,
): LocalAudio {
  const sr = audio.sampleRate;
  const bands = [60, 230, 910, 4000, 14000].filter((freq) => Math.abs(gainsDb[String(freq)] ?? 0) >= 0.01);
  const filters: Biquad[] = bands.map((freq) => peaking(freq, 1.1, gainsDb[String(freq)] ?? 0, sr));
  if (highpassHz && highpassHz > 0) filters.push(highpass(highpassHz, 0.707, sr));
  if (lowpassHz && lowpassHz > 0) filters.push(lowpass(lowpassHz, 0.707, sr));

  const count = audio.channels.length;
  const channels = audio.channels.map((channel, channelIndex) => {
    // Copy into a fresh ArrayBuffer-backed array; `new Float32Array(channel)`
    // preserves the input's ArrayBufferLike type and upsets the TS 5.9 generic.
    let out: Float32Array = new Float32Array(channel.length);
    out.set(channel);
    filters.forEach((filter, filterIndex) => {
      out = applyBiquad(out, filter);
      const done = (channelIndex * filters.length + filterIndex + 1) / (count * Math.max(1, filters.length));
      onProgress?.(done);
    });
    return out;
  });

  const equalised: LocalAudio = { sampleRate: sr, channels };
  // A +12 dB band can push a hot mix over full scale. Pull the whole file back
  // rather than clipping individual samples.
  const peak = peakOf(equalised);
  if (peak > 0.999) {
    const scale = 0.999 / peak;
    for (const channel of equalised.channels) {
      for (let i = 0; i < channel.length; i++) channel[i] *= scale;
    }
  }
  return equalised;
}

/** Linear-interpolation resampler. `step > 1` reads faster and raises pitch. */
function resampleLinear(x: Float32Array, step: number): Float32Array {
  const length = Math.max(1, Math.floor((x.length - 1) / step));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const position = i * step;
    const index = Math.floor(position);
    const fraction = position - index;
    const a = x[index] ?? 0;
    const b = x[index + 1] ?? a;
    out[i] = a + (b - a) * fraction;
  }
  return out;
}

/**
 * Pitch shift with duration preserved.
 *
 * Resampling changes pitch and length together; a phase-vocoder stretch then
 * restores the original length at the new pitch. This is the standard
 * resample-plus-stretch approach, and it is why a +12 st shift lands on 880 Hz
 * for a 440 Hz tone rather than on a different note.
 */
export function pitchShiftAudio(
  audio: LocalAudio,
  semitones: number,
  onProgress?: (fraction: number) => void,
): LocalAudio {
  const shift = Math.max(-12, Math.min(12, semitones));
  if (Math.abs(shift) < 1e-6) return cloneAudio(audio);
  const ratio = Math.pow(2, shift / 12);
  const count = audio.channels.length;

  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((channel, index) => {
      onProgress?.(index / count);
      const resampled = resampleLinear(channel, ratio);
      const stretched = timeStretch(
        resampled,
        ratio,
        DEFAULT_N_FFT,
        DEFAULT_HOP,
        onProgress ? (f) => onProgress((index + 0.1 + 0.9 * f) / count) : undefined,
      );
      return stretched;
    }),
  };
}

export interface SilenceRemoval {
  audio: LocalAudio;
  removedSeconds: number;
  pauses: number;
  /** Silence threshold in dB relative to the loudest frame. */
  thresholdDb: number;
}

/**
 * Remove pauses longer than `minPauseSeconds`.
 *
 * The detector uses 20 ms RMS frames with a 10 ms hop, and a threshold between
 * the measured noise floor and the peak. It keeps an 80 ms margin at each cut,
 * so a pause is shortened rather than clipped at the first consonant. Short
 * gaps stay in place (the server bug this mirrors removed every gap).
 *
 * This is energy-based, not a speech recogniser. It does not know whether a
 * quiet passage is a pause or a soft word; the threshold is deliberately low to
 * prefer leaving audio in over cutting speech.
 */
export function removeSilenceAudio(
  audio: LocalAudio,
  minPauseSeconds = 0.5,
  keepMarginSeconds = 0.08,
  onProgress?: (fraction: number) => void,
): SilenceRemoval {
  const sr = audio.sampleRate;
  const length = audio.channels[0]?.length ?? 0;
  const minPause = Math.max(0.05, Math.min(20, minPauseSeconds));
  const margin = Math.max(0, keepMarginSeconds);
  const unchanged = (): SilenceRemoval => ({
    audio: cloneAudio(audio),
    removedSeconds: 0,
    pauses: 0,
    thresholdDb: 0,
  });
  if (length === 0) return unchanged();

  const frame = Math.max(1, Math.round(0.02 * sr));
  const hop = Math.max(1, Math.round(0.01 * sr));
  const frames = Math.max(1, Math.floor((length - frame) / hop) + 1);

  // Mono frame energy; one channel is enough to decide where speech is.
  const energies = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    let sum = 0;
    for (const channel of audio.channels) {
      for (let i = 0; i < frame && start + i < length; i++) {
        const value = channel[start + i];
        sum += value * value;
      }
    }
    energies[f] = Math.sqrt(sum / (frame * Math.max(1, audio.channels.length)));
  }

  const sorted = Array.from(energies).sort((a, b) => a - b);
  const peak = sorted[sorted.length - 1] ?? 0;
  if (peak <= 1e-6) return unchanged();
  const floor = sorted[Math.floor(sorted.length * 0.1)] ?? 0;
  // Cap at half the peak: a continuous pad has floor == peak, and without the
  // cap it would be classified as silence and deleted.
  const threshold = Math.min(
    peak * 0.5,
    Math.max(peak * 0.01, floor * 2, 1e-4),
  );
  const thresholdDb = 20 * Math.log10(threshold / peak);

  const removals: Array<[number, number]> = [];
  let runStart = -1;
  for (let f = 0; f < frames; f++) {
    const silent = energies[f] < threshold;
    if (silent && runStart < 0) runStart = f;
    if (runStart >= 0 && (!silent || f === frames - 1)) {
      const runEnd = silent ? f : f - 1;
      const startSample = runStart * hop;
      const endSample = Math.min(length, runEnd * hop + frame);
      if ((endSample - startSample) / sr >= minPause) {
        const cutStart = startSample + Math.round(margin * sr);
        const cutEnd = endSample - Math.round(margin * sr);
        if (cutEnd > cutStart) removals.push([cutStart, cutEnd]);
      }
      runStart = -1;
    }
  }

  if (removals.length === 0) {
    return { ...unchanged(), thresholdDb };
  }

  const kept: Array<[number, number]> = [];
  let cursor = 0;
  for (const [start, end] of removals) {
    if (start > cursor) kept.push([cursor, start]);
    cursor = Math.max(cursor, end);
  }
  if (cursor < length) kept.push([cursor, length]);

  const outLength = kept.reduce((sum, [start, end]) => sum + (end - start), 0);
  const channels = audio.channels.map((channel, channelIndex) => {
    const out = new Float32Array(outLength);
    let position = 0;
    kept.forEach(([start, end]) => {
      out.set(channel.subarray(start, end), position);
      position += end - start;
    });

    // 2 ms fades at the joins. The 80 ms margin means these sit in silence, so
    // this only removes any residual click from a noisy pause.
    const fade = Math.min(Math.round(0.002 * sr), Math.floor(outLength / 2));
    if (fade > 0) {
      let offset = 0;
      kept.forEach(([start, end], index) => {
        const segment = end - start;
        if (index > 0) {
          for (let i = 0; i < fade; i++) out[offset + i] *= i / fade;
        }
        if (index < kept.length - 1) {
          for (let i = 0; i < fade; i++) out[offset + segment - 1 - i] *= i / fade;
        }
        offset += segment;
      });
    }
    onProgress?.((channelIndex + 1) / audio.channels.length);
    return out;
  });

  return {
    audio: { sampleRate: sr, channels },
    removedSeconds: (length - outLength) / sr,
    pauses: removals.length,
    thresholdDb,
  };
}
