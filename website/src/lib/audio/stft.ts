/**
 * Short-time Fourier transform with a Hann window and overlap-add resynthesis.
 *
 * `applySpectralGain` is the only entry point the operations use: it takes the
 * signal, hands the magnitude matrix to a gain function, and resynthesises.
 * Analysis and synthesis both use the same window, and the overlap-add is
 * divided by the accumulated squared window, so reconstruction is exact when
 * the gain is flat (no COLA assumption baked in).
 *
 * Magnitudes and gains are laid out bin-major (`bin * frames + frame`) because
 * every caller works per bin across time (noise percentiles, smoothing).
 */

import { fft, ifft } from "./fft";

export const DEFAULT_N_FFT = 1024;
export const DEFAULT_HOP = 256;

/** Periodic Hann window, matching `scipy.signal.get_window("hann", n)`. */
export function hann(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/** Number of frames a centred STFT produces for a signal of `length` samples. */
export function frameCount(length: number, nFft: number, hop: number): number {
  const pad = nFft >> 1;
  return 1 + Math.floor((length + 2 * pad - nFft) / hop);
}

/**
 * Apply a real, non-negative gain to every time-frequency cell.
 *
 * `gainFor` receives the magnitude matrix and must return one gain per cell in
 * the same layout. Gains above 1 are allowed (used by the presence shelf).
 */
export function applySpectralGain(
  x: Float32Array,
  nFft: number,
  hop: number,
  gainFor: (mag: Float32Array, bins: number, frames: number) => Float32Array,
): Float32Array {
  const n = x.length;
  if (n === 0) return new Float32Array(0);

  const win = hann(nFft);
  const bins = (nFft >> 1) + 1;
  const pad = nFft >> 1;
  const total = n + 2 * pad;
  const frames = 1 + Math.floor((total - nFft) / hop);

  // Forward transform.
  const re = new Float32Array(frames * bins);
  const im = new Float32Array(frames * bins);
  const fr = new Float32Array(nFft);
  const fi = new Float32Array(nFft);

  for (let f = 0; f < frames; f++) {
    const start = f * hop - pad;
    for (let i = 0; i < nFft; i++) {
      const src = start + i;
      fr[i] = src >= 0 && src < n ? x[src] * win[i] : 0;
      fi[i] = 0;
    }
    fft(fr, fi);
    for (let b = 0; b < bins; b++) {
      re[f * bins + b] = fr[b];
      im[f * bins + b] = fi[b];
    }
  }

  const mag = new Float32Array(bins * frames);
  for (let f = 0; f < frames; f++) {
    for (let b = 0; b < bins; b++) {
      mag[b * frames + f] = Math.hypot(re[f * bins + b], im[f * bins + b]);
    }
  }

  const gain = gainFor(mag, bins, frames);

  // Inverse transform + overlap-add.
  const out = new Float32Array(total);
  const wsum = new Float32Array(total);

  for (let f = 0; f < frames; f++) {
    for (let b = 0; b < bins; b++) {
      const g = gain[b * frames + f];
      fr[b] = re[f * bins + b] * g;
      fi[b] = im[f * bins + b] * g;
    }
    // Mirror the positive half onto the negative frequencies so the inverse
    // transform produces a real signal.
    for (let b = bins; b < nFft; b++) {
      fr[b] = fr[nFft - b];
      fi[b] = -fi[nFft - b];
    }
    ifft(fr, fi);

    const start = f * hop;
    for (let i = 0; i < nFft; i++) {
      out[start + i] += fr[i] * win[i];
      wsum[start + i] += win[i] * win[i];
    }
  }

  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const w = wsum[i + pad];
    y[i] = w > 1e-8 ? out[i + pad] / w : 0;
  }
  return y;
}
