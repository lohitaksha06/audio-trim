/**
 * Stationary-noise suppression for speech, ported from the server's
 * `denoise_speech` (server/ml/speech/voice.py) so browser and backend agree on
 * what "denoise" means.
 *
 * Per bin, the noise magnitude is the 12th percentile across time (minimum
 * statistics). The gain is a Wiener ratio with over-subtraction `alpha`, then
 * floored so the residual is not pure silence, then smoothed across time so the
 * gain does not flicker into musical noise.
 */

import { applySpectralGain, DEFAULT_HOP, DEFAULT_N_FFT } from "./stft";

/** Linear-interpolated percentile, matching `numpy.percentile` defaults. */
export function percentile(values: Float32Array, p: number): number {
  if (values.length === 0) return 0;
  const sorted = Float32Array.from(values);
  sorted.sort();
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/** Centred moving average with a Hann kernel, computed in place per bin. */
function smoothPerBin(gain: Float32Array, bins: number, frames: number, length: number): void {
  const win = Math.max(3, Math.round(length));
  if (frames < 3 || win < 3) return;
  const kernel = new Float32Array(win);
  let sum = 0;
  for (let i = 0; i < win; i++) {
    kernel[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / win);
    sum += kernel[i];
  }
  for (let i = 0; i < win; i++) kernel[i] /= sum;

  const half = win >> 1;
  const scratch = new Float32Array(frames);
  for (let b = 0; b < bins; b++) {
    const base = b * frames;
    for (let f = 0; f < frames; f++) {
      let acc = 0;
      for (let k = 0; k < win; k++) {
        // Clamp at the edges rather than zero-pad: a file that starts mid-noise
        // should not get a fade-in of suppression.
        const src = f + k - half;
        const idx = src < 0 ? 0 : src >= frames ? frames - 1 : src;
        acc += gain[base + idx] * kernel[k];
      }
      scratch[f] = acc;
    }
    for (let f = 0; f < frames; f++) gain[base + f] = scratch[f];
  }
}

/**
 * Denoise one channel. `strength` 1..3 scales the over-subtraction, same
 * mapping as the Python implementation.
 */
export function denoiseChannel(
  x: Float32Array,
  sampleRate: number,
  strength = 2,
  nFft: number = DEFAULT_N_FFT,
  hop: number = DEFAULT_HOP,
  onProgress?: (fraction: number) => void,
): Float32Array {
  if (x.length === 0) return x;
  const s = Math.min(3, Math.max(1, strength));
  const alpha = 0.8 + (s - 1) * 0.9;
  const floor = 0.1 - (s - 1) * 0.03;

  const out = applySpectralGain(x, nFft, hop, (mag, bins, frames) => {
    const gain = new Float32Array(bins * frames);
    const column = new Float32Array(frames);
    for (let b = 0; b < bins; b++) {
      const base = b * frames;
      for (let f = 0; f < frames; f++) column[f] = mag[base + f];
      const noise = Math.max(percentile(column, 12), 1e-10);
      const noise2 = noise * noise;
      for (let f = 0; f < frames; f++) {
        const power = mag[base + f] * mag[base + f];
        const g = power / (power + alpha * noise2 + 1e-20);
        gain[base + f] = Math.min(1, Math.max(floor, g));
      }
    }
    smoothPerBin(gain, bins, frames, (0.02 * sampleRate) / hop);
    for (let i = 0; i < gain.length; i++) gain[i] = Math.min(1, Math.max(floor, gain[i]));
    return gain;
  }, onProgress);

  // Keep a hot denoise from clipping the file on the way out.
  let peak = 0;
  for (let i = 0; i < out.length; i++) peak = Math.max(peak, Math.abs(out[i]));
  if (peak > 1) {
    const k = 0.99 / peak;
    for (let i = 0; i < out.length; i++) out[i] *= k;
  }
  return out;
}
