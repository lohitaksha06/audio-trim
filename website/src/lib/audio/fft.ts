/**
 * In-place iterative radix-2 FFT.
 *
 * The browser has no FFT primitive, and the denoiser needs one, so this is the
 * smallest correct implementation: bit-reversal permutation followed by
 * butterflies in log2(n) passes. Length must be a power of two.
 */

/** Reverse the low `bits` bits of `v`. */
function reverseBits(v: number, bits: number): number {
  let r = 0;
  for (let i = 0; i < bits; i++) {
    r = (r << 1) | (v & 1);
    v >>= 1;
  }
  return r;
}

/** Forward DFT, in place. `im` must be zeroed for a purely real input. */
export function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  if (n <= 1) return;
  if ((n & (n - 1)) !== 0) throw new Error(`fft length must be a power of two, got ${n}`);

  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    const j = reverseBits(i, bits);
    if (j > i) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = (-2 * Math.PI) / size;
    // Recompute each pass instead of a table: one fewer allocation, and the
    // trig is a rounding error next to the butterflies.
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const angle = step * k;
        const wr = Math.cos(angle);
        const wi = Math.sin(angle);
        const a = start + k;
        const b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
}

/** Inverse DFT (unscaled by 1/n here is applied on exit), in place. */
export function ifft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  // conj -> fft -> conj == inverse, with the 1/n scale on the way out.
  for (let i = 0; i < n; i++) im[i] = -im[i];
  fft(re, im);
  const scale = 1 / n;
  for (let i = 0; i < n; i++) {
    re[i] *= scale;
    im[i] = -im[i] * scale;
  }
}
