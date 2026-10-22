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
