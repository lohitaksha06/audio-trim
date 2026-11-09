import { describe, expect, test } from "bun:test";
import { applyPlan } from "./engine";
import { parseLocalPrompt } from "./parse";
import { describePolish, parsePolishPrompt, polishAudio, polishParams } from "./polish";
import type { LocalAudio } from "./types";

const SR = 8000;

function tone(seconds = 1, amplitude = 0.1): LocalAudio {
  const data = new Float32Array(Math.round(seconds * SR));
  for (let i = 0; i < data.length; i++) data[i] = Math.sin((2 * Math.PI * 220 * i) / SR) * amplitude;
  return { channels: [data], sampleRate: SR };
}

function peak(audio: LocalAudio): number {
  let value = 0;
  for (const channel of audio.channels) {
    for (let i = 0; i < channel.length; i++) value = Math.max(value, Math.abs(channel[i]));
  }
  return value;
}

describe("polish prompt", () => {
  test("parses the four-part cleanup sentence as one chain", () => {
    const options = parsePolishPrompt(
      "clean up the recording. Add a little reverb, adjust the volume and reduce background noise.",
    );
    expect(options).not.toBeNull();
    expect(options!.denoise).toBe(true);
    expect(options!.denoiseStrength).toBe(1);
    expect(options!.reverbMix).toBeCloseTo(0.12, 3);
    expect(options!.reverbSeconds).toBeCloseTo(1.2, 3);
    expect(options!.normalize).toBe(true);
  });

  test("'make my music perfect' becomes a cleanup and level pass, not fake mastering", () => {
    const options = parsePolishPrompt("make my music perfect");
    expect(options).not.toBeNull();
    expect(options!.denoise).toBe(true);
    expect(options!.reverbMix).toBe(0);
    expect(options!.normalize).toBe(true);
  });

  test("two effects trigger the chain even without a polish word", () => {
    const options = parsePolishPrompt("reduce the background noise and adjust the volume");
    expect(options).not.toBeNull();
    expect(options!.denoise).toBe(true);
    expect(options!.reverbMix).toBe(0);
    expect(options!.normalize).toBe(true);
  });

  test("a single 'add reverb' stays a single-purpose edit", () => {
    expect(parsePolishPrompt("add reverb")).toBeNull();
  });

  test("describes exactly the steps that will run", () => {
    const text = describePolish({
      denoise: true,
      denoiseStrength: 2,
      reverbMix: 0.12,
      reverbSeconds: 1.2,
      normalize: true,
      targetPeakDb: -1,
    });
    expect(text).toContain("noise reduction");
    expect(text).toContain("reverb");
    expect(text).toContain("-1.0 dBFS");
  });

  test("plan params use the engine's snake-case keys", () => {
    const params = polishParams({
      denoise: true,
      denoiseStrength: 2,
      reverbMix: 0.12,
      reverbSeconds: 1.2,
      normalize: true,
      targetPeakDb: -1,
    });
    expect(params.denoise_strength).toBe(2);
    expect(params.reverb_mix).toBe(0.12);
    expect(params.target_peak_db).toBe(-1);
  });
});

describe("polish chain", () => {
  test("normalizes the peak to -1 dBFS", () => {
    const result = polishAudio(tone(1, 0.1), {
      denoise: false,
      denoiseStrength: 2,
      reverbMix: 0,
      reverbSeconds: 0,
      normalize: true,
      targetPeakDb: -1,
    });
    expect(peak(result.audio)).toBeCloseTo(10 ** (-1 / 20), 2);
    expect(result.report.some((line) => line.includes("normalized"))).toBe(true);
  });

  test("reverb leaves a tail after the source stops", () => {
    const source: LocalAudio = { channels: [new Float32Array(SR)], sampleRate: SR };
    for (let i = 0; i < 80; i++) source.channels[0][i] = 0.9;
    const result = polishAudio(source, {
      denoise: false,
      denoiseStrength: 2,
      reverbMix: 0.5,
      reverbSeconds: 1.5,
      normalize: false,
      targetPeakDb: -1,
    });
    let tail = 0;
    for (let i = Math.round(0.5 * SR); i < result.audio.channels[0].length; i++) {
      tail = Math.max(tail, Math.abs(result.audio.channels[0][i]));
    }
    expect(tail).toBeGreaterThan(0.0001);
  });

  test("the local parser routes the combined sentence to the polish intent", () => {
    const plan = parseLocalPrompt(
      "clean up the recording, add a little reverb, adjust the volume and reduce background noise",
      1,
    );
    expect(plan.intent).toBe("polish");
    expect(plan.params.denoise).toBe(true);
    expect(plan.params.normalize).toBe(true);
  });

  test("a standalone 'adjust the volume' still means normalize", () => {
    const plan = parseLocalPrompt("adjust the volume", 1);
    expect(plan.intent).toBe("normalize");
  });

  test("the engine renders the chain and reports it", () => {
    const plan = parseLocalPrompt("make my music perfect", 1);
    const output = applyPlan(tone(1, 0.1), plan);
    expect(output.audio.channels[0].length).toBeGreaterThan(0);
    expect(Array.isArray(output.metadata.polish_report)).toBe(true);
    expect(String(output.summary).toLowerCase()).toContain("polished");
  });
});
