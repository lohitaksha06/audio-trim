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
