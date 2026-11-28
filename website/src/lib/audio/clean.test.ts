/**
 * Behaviour tests for the "fix my recording" diagnosis and repair chain.
 *
 * These check the properties the page relies on: clipping is counted, declipping
 * removes the flat top, normalise hits the target, and the plan only asks for a
 * repair that the measurements actually justify.
 */

import { describe, expect, test } from "bun:test";

import {
  CLIP_LEVEL,
  TARGET_PEAK_DB,
  applyCleanPlan,
  declipAudio,
  declipChannel,
  detectClipping,
  diagnose,
  frameLevels,
  normalizePeak,
  peakDbOf,
  planClean,
} from "./clean";
import type { LocalAudio } from "./types";

const SR = 16000;

function mono(samples: Float32Array): LocalAudio {
  return { channels: [samples], sampleRate: SR };
}

function sine(freq: number, seconds: number, amp = 0.5): Float32Array {
  const n = Math.round(SR * seconds);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / SR);
  return out;
}

/** Deterministic pseudo-noise so the test never depends on Math.random. */
function noise(length: number, amp: number, seed = 12345): Float32Array {
  const out = new Float32Array(length);
  let state = seed;
  for (let i = 0; i < length; i++) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = ((state / 0xffffffff) * 2 - 1) * amp;
  }
  return out;
}

function withPauses(tone: Float32Array, noiseAmp: number): Float32Array {
  const out = Float32Array.from(tone);
  const block = Math.round(SR * 0.5);
  for (let i = 0; i < out.length; i++) {
    const inPause = Math.floor(i / block) % 2 === 1;
    if (inPause) out[i] = 0;
  }
  const hiss = noise(out.length, noiseAmp);
  for (let i = 0; i < out.length; i++) out[i] += hiss[i];
  return out;
}

function hardClip(x: Float32Array, level: number): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.max(-level, Math.min(level, x[i]));
  return out;
}

describe("clipping detection", () => {
  test("counts flat-topped samples and their runs", () => {
    const clipped = hardClip(sine(100, 0.5, 1.3), 1);
    const report = detectClipping(mono(clipped));
    expect(report.clippedSamples).toBeGreaterThan(0);
    expect(report.percent).toBeGreaterThan(0.1);
    expect(report.runs).toBeGreaterThan(0);
    expect(report.longestRun).toBeGreaterThan(2);
  });

  test("a clean sine reports no clipping", () => {
    const report = detectClipping(mono(sine(440, 0.5, 0.8)));
    expect(report.clippedSamples).toBe(0);
    expect(report.percent).toBe(0);
    expect(report.runs).toBe(0);
  });
});

describe("declipping", () => {
  test("removes the flat top without changing length", () => {
    const clipped = hardClip(sine(100, 0.5, 1.3), 1);
    const beforeReport = detectClipping(mono(clipped));
    const fixed = declipChannel(clipped);
    const afterReport = detectClipping(mono(fixed));

    expect(fixed.length).toBe(clipped.length);
    expect(afterReport.longestRun).toBeLessThan(beforeReport.longestRun);
    for (let i = 0; i < fixed.length; i++) expect(Number.isFinite(fixed[i])).toBe(true);
  });

  test("restores headroom above the clip level", () => {
    const clipped = hardClip(sine(100, 0.2, 1.3), 1);
    const fixed = declipChannel(clipped);

    let peak = 0;
    for (let i = 0; i < fixed.length; i++) peak = Math.max(peak, Math.abs(fixed[i]));
    expect(peak).toBeGreaterThan(CLIP_LEVEL);
  });

  test("leaves silence and tiny buffers alone", () => {
    expect(declipChannel(new Float32Array(0)).length).toBe(0);
    const silence = new Float32Array(64);
    expect(declipChannel(silence)).toEqual(silence);
  });

  test("works on every channel of a stereo file", () => {
    const left = hardClip(sine(100, 0.2, 1.3), 1);
    const right = hardClip(sine(120, 0.2, 1.3), 1);
    const fixed = declipAudio({ channels: [left, right], sampleRate: SR });
    expect(fixed.channels.length).toBe(2);
    expect(fixed.channels[0].length).toBe(left.length);
    expect(fixed.channels[1].length).toBe(right.length);
  });
});

describe("normalise", () => {
  test("moves the peak to the target", () => {
    const quiet = mono(sine(440, 0.25, 0.1));
    const loud = normalizePeak(quiet);
    expect(peakDbOf(loud)).toBeCloseTo(TARGET_PEAK_DB, 1);
  });

  test("never pushes a sample past full scale", () => {
    const loud = normalizePeak(mono(sine(440, 0.25, 1.4)));
    for (const channel of loud.channels) {
      for (let i = 0; i < channel.length; i++) expect(Math.abs(channel[i])).toBeLessThanOrEqual(1);
    }
  });

  test("is a no-op on digital silence", () => {
    const silent = mono(new Float32Array(256));
    expect(normalizePeak(silent)).toBe(silent);
  });
});

describe("level analysis", () => {
  test("finds a noise floor when the file has pauses", () => {
    const paused = withPauses(sine(220, 2, 0.4), 0.002);
    const levels = frameLevels(mono(paused));
    expect(levels.dynamicRangeDb).toBeGreaterThan(6);
    expect(levels.activeLevelDb).toBeGreaterThan(levels.noiseFloorDb);
  });

  test("flags a continuous signal as unreliable", () => {
    const steady = sine(220, 2, 0.4);
    const diagnosis = diagnose(mono(steady));
    expect(diagnosis.noiseReliable).toBe(false);
  });

  test("reports a lower SNR for a noisier recording", () => {
    const clean = diagnose(mono(withPauses(sine(220, 2, 0.4), 0.001)));
    const dirty = diagnose(mono(withPauses(sine(220, 2, 0.4), 0.05)));
    expect(clean.snrDb).toBeGreaterThan(dirty.snrDb);
  });
});

describe("repair plan", () => {
  test("declips a flat-topped file and always normalises last", () => {
    const diagnosis = diagnose(mono(hardClip(sine(100, 1, 1.4), 1)));
    const plan = planClean(diagnosis);
    expect(plan.steps[0].key).toBe("declip");
    expect(plan.steps[plan.steps.length - 1].key).toBe("normalize");
  });

  test("skips denoise when the file is mostly continuous", () => {
    const diagnosis = diagnose(mono(sine(220, 1, 0.4)));
    const plan = planClean(diagnosis);
    expect(plan.steps.some((step) => step.key === "denoise")).toBe(false);
    expect(plan.skipped.join(" ")).toContain("no pauses");
  });

  test("adds denoise for a noisy recording with pauses", () => {
    const diagnosis = diagnose(mono(withPauses(sine(220, 2, 0.4), 0.05)));
    const plan = planClean(diagnosis);
    expect(plan.steps.some((step) => step.key === "denoise")).toBe(true);
  });
});

describe("full chain", () => {
  test("produces a finite, full-scale-safe file", () => {
    const source = mono(hardClip(withPauses(sine(220, 1, 1.4), 0.01), 1));
    const before = diagnose(source);
    const result = applyCleanPlan(source, before, planClean(before));

    expect(result.audio.channels[0].length).toBe(source.channels[0].length);
    expect(result.audio.sampleRate).toBe(SR);
    for (const channel of result.audio.channels) {
      for (let i = 0; i < channel.length; i++) expect(Number.isFinite(channel[i])).toBe(true);
    }
    expect(peakDbOf(result.audio)).toBeLessThanOrEqual(TARGET_PEAK_DB + 0.5);
  });

  test("reduces the clipped fraction it measured", () => {
    const source = mono(hardClip(sine(100, 1, 1.4), 1));
    const before = diagnose(source);
    const result = applyCleanPlan(source, before, planClean(before));
    expect(result.after.clipping.percent).toBeLessThan(before.clipping.percent);
  });
});
