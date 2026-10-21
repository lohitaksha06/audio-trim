/**
 * Checks for the browser DSP. Run with `bun test src/lib/audio`.
 *
 * These are property/behaviour tests, not golden files: the point is that a
 * flat spectral gain reconstructs, a denoise actually raises SNR against the
 * clean reference, and the phase vocoder keeps pitch while changing length.
 */

import { describe, expect, test } from "bun:test";

import { fft, ifft } from "./fft";
import { applySpectralGain } from "./stft";
import { denoiseChannel } from "./denoise";
import {
  applyGain,
  boostTarget,
  fade,
  normalize,
  reverse,
  stretchAudio,
  timeStretch,
  trim,
} from "./ops";
import { LOCAL_PROMPTS, parseLocalPrompt, parseTime } from "./parse";
import { durationOf, encodeWav } from "./wav";
import { runLocalPlan } from "./engine";
import type { LocalAudio } from "./types";

const SR = 22050;

function sine(freq: number, seconds: number, amp = 0.5): Float32Array {
  const out = new Float32Array(Math.round(SR * seconds));
  for (let i = 0; i < out.length; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / SR);
  return out;
}

function noise(length: number, amp: number, seed = 7): Float32Array {
  // Small deterministic LCG so the test does not depend on Math.random.
  let state = seed >>> 0;
  const out = new Float32Array(length);
  for (let i = 0; i < out.length; i++) {
    state = (1664525 * state + 1013904223) >>> 0;
    out[i] = ((state / 0xffffffff) * 2 - 1) * amp;
  }
  return out;
}

function snrDb(signal: Float32Array, clean: Float32Array): number {
  const n = Math.min(signal.length, clean.length);
  let sig = 0;
  let err = 0;
  for (let i = 0; i < n; i++) {
    sig += signal[i] * signal[i];
    const d = signal[i] - clean[i];
    err += d * d;
  }
  return 10 * Math.log10((sig + 1e-20) / (err + 1e-20));
}

function bandEnergy(x: Float32Array, lo: number, hi: number): number {
  const n = 4096;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  re.set(x.subarray(0, Math.min(n, x.length)));
  fft(re, im);
  let sum = 0;
  for (let b = 0; b < n / 2; b++) {
    const f = (b * SR) / n;
    if (f >= lo && f <= hi) sum += re[b] * re[b] + im[b] * im[b];
  }
  return sum;
}

function dominantFreq(x: Float32Array): number {
  const n = 8192;
  const re = new Float32Array(n);
  const im = new Float32Array(n);
  re.set(x.subarray(0, Math.min(n, x.length)));
  fft(re, im);
  let best = 0;
  let bestBin = 0;
  for (let b = 1; b < n / 2; b++) {
    const mag = re[b] * re[b] + im[b] * im[b];
    if (mag > best) {
      best = mag;
      bestBin = b;
    }
  }
  return (bestBin * SR) / n;
}

describe("fft", () => {
  test("round-trips exactly", () => {
    const n = 1024;
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.sin(i / 7) + 0.3 * Math.cos(i / 3);
    const original = Float32Array.from(re);
    fft(re, im);
    ifft(re, im);
    let worst = 0;
    for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(re[i] - original[i]));
    expect(worst).toBeLessThan(1e-4);
  });
});

describe("stft", () => {
  test("a flat gain reconstructs the input", () => {
    const x = new Float32Array(SR);
    for (let i = 0; i < x.length; i++) x[i] = Math.sin((2 * Math.PI * 300 * i) / SR) * 0.4;
    const y = applySpectralGain(x, 1024, 256, (_mag, bins, frames) => {
      const g = new Float32Array(bins * frames);
      g.fill(1);
      return g;
    });
    let worst = 0;
    for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(y[i] - x[i]));
    expect(worst).toBeLessThan(1e-4);
  });
});

describe("denoise", () => {
  test("raises SNR against the clean tone", () => {
    // Gated tone, not a continuous one: minimum statistics needs to see the
    // noise floor during gaps, exactly like the server's denoiser. A constant
    // tone has no gaps, so its own magnitude *is* the 12th percentile and the
    // filter would treat it as noise (see the docs note in the README).
    const clean = new Float32Array(Math.round(SR * 2));
    for (let i = 0; i < clean.length; i++) {
      const t = i / SR;
      const gate = Math.max(0, Math.sin(2 * Math.PI * 3 * t));
      clean[i] = 0.4 * gate * Math.sin((2 * Math.PI * 440 * i) / SR);
    }
    const noisy = new Float32Array(clean.length);
    const n = noise(clean.length, 0.2);
    for (let i = 0; i < clean.length; i++) noisy[i] = clean[i] + n[i];

    const before = snrDb(noisy, clean);
    const after = snrDb(denoiseChannel(noisy, SR, 2), clean);
    // Measured on this fixture the gain is several dB; 1 dB is a deliberately
    // loose floor so the test asserts the direction, not a tuning constant.
    expect(after).toBeGreaterThan(before + 1);
  });
});

describe("time ops", () => {
  const source: LocalAudio = { channels: [sine(220, 2, 0.5), sine(330, 2, 0.5)], sampleRate: SR };

  test("trim keeps the requested window", () => {
    const out = trim(source, 0.5, 1.5);
    expect(out.channels[0].length).toBe(SR);
    expect(durationOf(out)).toBeCloseTo(1, 2);
  });

  test("fade reaches zero at the edge", () => {
    const out = fade(source, 0.5, 0.5);
    expect(Math.abs(out.channels[0][0])).toBeLessThan(1e-6);
    expect(Math.abs(out.channels[0][out.channels[0].length - 1])).toBeLessThan(1e-6);
    expect(Math.abs(out.channels[0][15000])).toBeGreaterThan(0.1);
  });

  test("gain is applied and reported", () => {
    const result = applyGain(source, -6);
    expect(result.appliedDb).toBeCloseTo(-6, 5);
    expect(Math.abs(result.audio.channels[0][100])).toBeLessThan(Math.abs(source.channels[0][100]));
  });

  test("normalize hits -1 dBFS", () => {
    const quiet: LocalAudio = { channels: [sine(220, 1, 0.1)], sampleRate: SR };
    const out = normalize(quiet, -1);
    let peak = 0;
    for (const v of out.audio.channels[0]) peak = Math.max(peak, Math.abs(v));
    expect(peak).toBeCloseTo(0.891, 2);
  });

  test("reverse is an involution", () => {
    const once = reverse(source);
    const twice = reverse(once);
    for (let i = 0; i < source.channels[0].length; i++) {
      expect(twice.channels[0][i]).toBeCloseTo(source.channels[0][i], 6);
    }
  });

  test("time stretch changes length but keeps pitch", () => {
    const tone = sine(440, 1.5, 0.5);
    const stretched = timeStretch(tone, 1.5);
    expect(stretched.length / tone.length).toBeCloseTo(1.5, 2);
    expect(Math.abs(dominantFreq(stretched) - 440)).toBeLessThan(12);
  });

  test("speed up shortens without changing pitch", () => {
    const out = stretchAudio({ channels: [sine(440, 2, 0.5)], sampleRate: SR }, 0.5);
    expect(durationOf(out)).toBeCloseTo(1, 1);
    expect(Math.abs(dominantFreq(out.channels[0]) - 440)).toBeLessThan(12);
  });

  test("a 1.25x speed prompt shortens the file through the engine", async () => {
    // Regression: the engine briefly passed the speed multiplier straight to
    // the stretcher, so "Speed up 1.25x" made the file 25% longer.
    const source: LocalAudio = { channels: [sine(440, 2, 0.5)], sampleRate: SR };
    const result = await runLocalPlan(source, parseLocalPrompt("Speed up 1.25x", 2));
    expect(durationOf(result.audio)).toBeCloseTo(1.6, 1);
  });
});

describe("boost", () => {
  test("vocals lift the presence band relative to the low end", () => {
    // Two tones: 250 Hz "body" and 3 kHz "presence". The shelf should move the
    // ratio, not just the level.
    const a = sine(250, 1, 0.3);
    const b = sine(3000, 1, 0.3);
    const mixed = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) mixed[i] = a[i] + b[i];
    const source: LocalAudio = { channels: [mixed], sampleRate: SR };
    const out = boostTarget(source, "vocals", "up");
    const before = bandEnergy(source.channels[0], 2000, 4000) / bandEnergy(source.channels[0], 150, 350);
    const after = bandEnergy(out.channels[0], 2000, 4000) / bandEnergy(out.channels[0], 150, 350);
    expect(after).toBeGreaterThan(before * 1.3);
  });
});

describe("wav", () => {
  test("writes a readable 16-bit header", async () => {
    const audio: LocalAudio = { channels: [sine(220, 0.5, 0.5), sine(330, 0.5, 0.5)], sampleRate: SR };
    const blob = encodeWav(audio);
    const bytes = new Uint8Array(await blob.arrayBuffer());
    const view = new DataView(bytes.buffer);
    expect(String.fromCharCode(...bytes.slice(0, 4))).toBe("RIFF");
    expect(String.fromCharCode(...bytes.slice(8, 12))).toBe("WAVE");
    expect(view.getUint16(20, true)).toBe(1);
    expect(view.getUint16(22, true)).toBe(2);
    expect(view.getUint32(24, true)).toBe(SR);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(audio.channels[0].length * 4);
    expect(bytes.length).toBe(44 + audio.channels[0].length * 4);
  });
});

describe("prompt parsing", () => {
  test("parses time formats", () => {
    expect(parseTime("1:15")).toBe(75);
    expect(parseTime("1:02:03")).toBe(3723);
    expect(parseTime("90s")).toBe(90);
    expect(parseTime("2 min")).toBe(120);
    expect(parseTime("1 minute 30 seconds")).toBe(90);
    expect(parseTime("42")).toBe(42);
  });

  test("maps the local prompt set", () => {
    const cases: [string, string][] = [
      ["Trim from 0:30 to 1:15", "trim"],
      ["Cut the first 5 seconds", "trim"],
      ["Cut the last 10 seconds", "trim"],
      ["Fade in over 2 seconds", "fade"],
      ["Fade out over 3 seconds", "fade"],
      ["Make it 3 dB louder", "gain"],
      ["Normalize the audio", "normalize"],
      ["Reverse it", "reverse"],
      ["Speed up 1.25x", "speed"],
      ["Remove the background noise", "denoise"],
      ["Boost the vocals", "boost"],
      ["Make the drums louder", "boost"],
      ["Boost the bass", "boost"],
    ];
    for (const [prompt, intent] of cases) {
      expect(parseLocalPrompt(prompt, 180).intent).toBe(intent);
    }
  });

  test("trim parameters are seconds", () => {
    const plan = parseLocalPrompt("Trim from 0:30 to 1:15", 180);
    expect(plan.params.start).toBe(30);
    expect(plan.params.end).toBe(75);

    const first = parseLocalPrompt("Cut the first 5 seconds", 180);
    expect(first.params.start).toBe(5);
    expect(first.params.end).toBe(180);

    const last = parseLocalPrompt("Cut the last 10 seconds", 180);
    expect(last.params.end).toBe(170);
  });

  test("speed reads multipliers", () => {
    expect(parseLocalPrompt("Speed up 1.25x", 60).params.factor).toBe(1.25);
    expect(parseLocalPrompt("Slow down to 0.8x", 60).params.factor).toBe(0.8);
  });

  test("boost keeps its direction", () => {
    expect(parseLocalPrompt("Boost the vocals", 60).params.direction).toBe("up");
    expect(parseLocalPrompt("Lower the bass", 60).params.direction).toBe("down");
  });

  test("refuses what the browser cannot do", () => {
    for (const prompt of [
      "Separate the stems",
      "Add funky drums",
      "Make the drums louder and the vocals quieter",
      "Convert to mp3",
    ]) {
      const plan = parseLocalPrompt(prompt, 60);
      expect(plan.unsupported).toBeTruthy();
    }
  });

  test("every advertised chip parses to an operation", () => {
    for (const prompt of LOCAL_PROMPTS) {
      const plan = parseLocalPrompt(prompt, 180);
      expect(plan.intent).not.toBe("unknown");
      expect(plan.unsupported).toBeUndefined();
    }
  });
});
