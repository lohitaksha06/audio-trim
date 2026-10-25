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
  applyEq,
  applyGain,
  boostTarget,
  fade,
  normalize,
  pitchShiftAudio,
  removeSilenceAudio,
  reverse,
  reverbAudio,
  shapeTone,
  stretchAudio,
  timeStretch,
  trim,
} from "./ops";
import { estimateTempo } from "./tempo";
import { LOCAL_PROMPTS, localSupport, parseLocalPrompt, parseTime } from "./parse";
import { durationOf, encodeWav, encodeWavBuffer } from "./wav";
import { applyPlan, runLocalPlan } from "./engine";
import { runLocalPlanAsync } from "./client";
import type { LocalAudio, LocalPlan } from "./types";

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

/** Short decaying 1 kHz bursts on a steady beat. */
function clickTrack(bpm: number, seconds: number, amp = 0.8): Float32Array {
  const out = new Float32Array(Math.round(SR * seconds));
  const period = 60 / bpm;
  const burst = Math.round(0.03 * SR);
  for (let t = 0; t < seconds; t += period) {
    const start = Math.round(t * SR);
    for (let i = 0; i < burst && start + i < out.length; i++) {
      const envelope = Math.exp(-i / (0.004 * SR));
      out[start + i] += amp * envelope * Math.sin((2 * Math.PI * 1000 * i) / SR);
    }
  }
  return out;
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

describe("tempo", () => {
  test("detects a steady click track", () => {
    for (const bpm of [90, 120, 140]) {
      const source: LocalAudio = { channels: [clickTrack(bpm, 8)], sampleRate: SR };
      const estimate = estimateTempo(source);
      expect(estimate).not.toBeNull();
      expect(Math.abs((estimate as { bpm: number }).bpm - bpm)).toBeLessThan(2.5);
      expect((estimate as { confidence: number }).confidence).toBeGreaterThan(0.3);
    }
  });

  test("returns null for silence and for a steady tone", () => {
    const silence: LocalAudio = { channels: [new Float32Array(SR * 2)], sampleRate: SR };
    expect(estimateTempo(silence)).toBeNull();
    // A continuous sine has no onsets. It must not be assigned an invented BPM.
    const tone: LocalAudio = { channels: [sine(440, 3, 0.5)], sampleRate: SR };
    expect(estimateTempo(tone)).toBeNull();
  });

  test("target BPM shortens a 120 BPM file to 140 BPM", () => {
    const source: LocalAudio = { channels: [clickTrack(120, 8)], sampleRate: SR };
    const measured = estimateTempo(source);
    expect(measured).not.toBeNull();
    const plan = parseLocalPrompt("Set tempo to 140 BPM", 8);
    expect(plan.intent).toBe("speed");
    expect(plan.params.targetBpm).toBe(140);
    const result = applyPlan(source, plan);
    const expected = 8 * (measured as { bpm: number }).bpm / 140;
    expect(Math.abs(durationOf(result.audio) - expected)).toBeLessThan(0.1);
    expect(result.metadata.target_bpm).toBe(140);
  });

  test("a BPM question returns info and no new audio", () => {
    const source: LocalAudio = { channels: [clickTrack(120, 8)], sampleRate: SR };
    const plan = parseLocalPrompt("What BPM is this?", 8);
    expect(plan.intent).toBe("info");
    const result = applyPlan(source, plan);
    expect(result.metadata.info_only).toBe(true);
    expect(Math.abs(Number(result.metadata.bpm) - 120)).toBeLessThan(2.5);
    expect(result.audio.channels[0].length).toBe(source.channels[0].length);
  });
});

describe("EQ and pitch", () => {
  test("a low-band boost lifts 60 Hz relative to 4 kHz", () => {
    const low = sine(60, 1, 0.2);
    const high = sine(4000, 1, 0.2);
    const mixed = new Float32Array(low.length);
    for (let i = 0; i < mixed.length; i++) mixed[i] = low[i] + high[i];
    const source: LocalAudio = { channels: [mixed], sampleRate: SR };
    const out = applyEq(source, { "60": 6 });
    const before = bandEnergy(source.channels[0], 50, 70) / bandEnergy(source.channels[0], 3500, 4500);
    const after = bandEnergy(out.channels[0], 50, 70) / bandEnergy(out.channels[0], 3500, 4500);
    expect(after).toBeGreaterThan(before * 1.5);
  });

  test("high-pass and low-pass remove their stop bands", () => {
    const low = sine(60, 1, 0.2);
    const high = sine(4000, 1, 0.2);
    const mixed = new Float32Array(low.length);
    for (let i = 0; i < mixed.length; i++) mixed[i] = low[i] + high[i];
    const source: LocalAudio = { channels: [mixed], sampleRate: SR };

    const highpassed = applyEq(source, {}, 200, null);
    const hpBefore = bandEnergy(source.channels[0], 50, 70) / bandEnergy(source.channels[0], 3500, 4500);
    const hpAfter = bandEnergy(highpassed.channels[0], 50, 70) / bandEnergy(highpassed.channels[0], 3500, 4500);
    expect(hpAfter).toBeLessThan(hpBefore * 0.2);

    const lowpassed = applyEq(source, {}, null, 1000);
    const lpBefore = bandEnergy(source.channels[0], 3500, 4500) / bandEnergy(source.channels[0], 50, 70);
    const lpAfter = bandEnergy(lowpassed.channels[0], 3500, 4500) / bandEnergy(lowpassed.channels[0], 50, 70);
    expect(lpAfter).toBeLessThan(lpBefore * 0.2);
  });

  test("pitch shift raises the note and keeps the duration", () => {
    const source: LocalAudio = { channels: [sine(440, 1, 0.5)], sampleRate: SR };
    const up = pitchShiftAudio(source, 12);
    expect(durationOf(up)).toBeCloseTo(1, 1);
    expect(Math.abs(dominantFreq(up.channels[0]) - 880)).toBeLessThan(20);

    const down = pitchShiftAudio(source, -12);
    expect(durationOf(down)).toBeCloseTo(1, 1);
    expect(Math.abs(dominantFreq(down.channels[0]) - 220)).toBeLessThan(15);
  });

  test("removing a section shortens the file with a crossfade", () => {
    const source: LocalAudio = { channels: [sine(440, 10, 0.4)], sampleRate: SR };
    const plan = parseLocalPrompt("Remove the section from 0:02 to 0:04", 10);
    expect(plan.intent).toBe("remove");
    const result = applyPlan(source, plan);
    expect(durationOf(result.audio)).toBeCloseTo(8, 1);
    expect(result.metadata.removed_seconds).toBe(2);

    const whole = parseLocalPrompt("Remove the section from 0:00 to 0:10", 10);
    expect(() => applyPlan(source, whole)).toThrow(/whole file/);
  });

  test("pitch and transpose preserve duration", () => {
    const source: LocalAudio = { channels: [sine(440, 1, 0.5)], sampleRate: SR };
    const plan = parseLocalPrompt("Pitch it up 2 semitones", 1);
    expect(plan.intent).toBe("pitch");
    expect(plan.params.semitones).toBe(2);
    const result = applyPlan(source, plan);
    expect(durationOf(result.audio)).toBeCloseTo(1, 1);
    expect(Math.abs(dominantFreq(result.audio.channels[0]) - 493.88)).toBeLessThan(10);
  });

  test("loop repeats the range the requested number of times", () => {
    const source: LocalAudio = { channels: [sine(440, 4, 0.4)], sampleRate: SR };
    const plan = parseLocalPrompt("Loop from 0:01 to 0:02 3 times", 4);
    expect(plan.intent).toBe("loop");
    expect(plan.params.times).toBe(3);
    const result = applyPlan(source, plan);
    // 4 s original + two extra 1 s repeats, minus two 10 ms crossfades.
    expect(durationOf(result.audio)).toBeCloseTo(6, 1);
    expect(result.metadata.loop_times).toBe(3);
  });

  test("the Lab plan renders without changing duration", () => {
    const source: LocalAudio = { channels: [sine(220, 1, 0.4)], sampleRate: SR };
    const plan: LocalPlan = {
      intent: "lab",
      params: {
        gains: { "60": 3, "4000": -1 },
        highpass: 60,
        lowpass: 12000,
        pitch: 2,
        speed: 1.1,
        gainDb: -1,
        reverb: 0.1,
      },
      summary: "",
    };
    const result = applyPlan(source, plan);
    expect(durationOf(result.audio)).toBeCloseTo(1 / 1.1, 1);
    expect(result.metadata.lab_params).toBeTruthy();
    for (const value of result.audio.channels[0]) expect(Number.isFinite(value)).toBe(true);
  });
});

describe("pause removal", () => {
  function withPause(seconds = 2): LocalAudio {
    const first = sine(440, 1, 0.4);
    const gap = new Float32Array(Math.round(SR * seconds));
    const second = sine(660, 1, 0.4);
    const channel = new Float32Array(first.length + gap.length + second.length);
    channel.set(first);
    channel.set(gap, first.length);
    channel.set(second, first.length + gap.length);
    return { channels: [channel], sampleRate: SR };
  }

  test("removes a pause longer than the threshold and keeps the speech", () => {
    const source = withPause(2);
    const result = removeSilenceAudio(source, 0.5);
    expect(result.pauses).toBe(1);
    expect(result.removedSeconds).toBeGreaterThan(1.6);
    expect(durationOf(result.audio)).toBeGreaterThan(1.9);
    expect(durationOf(result.audio)).toBeLessThan(2.4);
    expect(bandEnergy(result.audio.channels[0], 400, 480)).toBeGreaterThan(0.01);
  });

  test("keeps a pause shorter than the threshold", () => {
    const source = withPause(0.25);
    const result = removeSilenceAudio(source, 0.5);
    expect(result.pauses).toBe(0);
    expect(durationOf(result.audio)).toBeCloseTo(2.25, 2);
  });

  test("does not delete a continuous tone", () => {
    const source: LocalAudio = { channels: [sine(440, 2, 0.4)], sampleRate: SR };
    const result = removeSilenceAudio(source, 0.5);
    expect(result.pauses).toBe(0);
    expect(durationOf(result.audio)).toBeCloseTo(2, 2);
  });

  test("parses the threshold and applies it", () => {
    const plan = parseLocalPrompt("Remove all long pauses and silences longer than 0.5 seconds", 10);
    expect(plan.intent).toBe("silence");
    expect(plan.params.minSeconds).toBe(0.5);
    expect(parseLocalPrompt("Cut pauses over 1s", 10).params.minSeconds).toBe(1);
    expect(parseLocalPrompt("Remove filler words", 10).intent).toBe("unknown");

    const result = applyPlan(withPause(2), plan);
    expect(result.metadata.pauses_removed).toBe(1);
    expect(result.metadata.duration_seconds as number).toBeLessThan(2.5);
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

describe("mood", () => {
  // 200 Hz "body" + 8 kHz "air", so a tilt changes the ratio, not just level.
  function mixed(): LocalAudio {
    const low = sine(200, 1, 0.3);
    const high = sine(8000, 1, 0.3);
    const sum = new Float32Array(low.length);
    for (let i = 0; i < sum.length; i++) sum[i] = low[i] + high[i];
    return { channels: [sum], sampleRate: SR };
  }

  test("darker rolls the highs off", () => {
    const source = mixed();
    const out = shapeTone(source, "darker");
    const before = bandEnergy(source.channels[0], 6000, 10000) / bandEnergy(source.channels[0], 150, 300);
    const after = bandEnergy(out.channels[0], 6000, 10000) / bandEnergy(out.channels[0], 150, 300);
    expect(after).toBeLessThan(before * 0.5);
  });

  test("brighter lifts the highs", () => {
    const source = mixed();
    const out = shapeTone(source, "brighter");
    const before = bandEnergy(source.channels[0], 6000, 10000) / bandEnergy(source.channels[0], 150, 300);
    const after = bandEnergy(out.channels[0], 6000, 10000) / bandEnergy(out.channels[0], 150, 300);
    expect(after).toBeGreaterThan(before * 1.5);
  });

  test("calm is quieter and energetic is louder", () => {
    const source = mixed();
    const calm = shapeTone(source, "calm");
    const energetic = shapeTone(source, "energetic");
    expect(bandEnergy(calm.channels[0], 20, 10000)).toBeLessThan(bandEnergy(source.channels[0], 20, 10000) * 0.8);
    expect(bandEnergy(energetic.channels[0], 20, 10000)).toBeGreaterThan(bandEnergy(source.channels[0], 20, 10000) * 1.2);
  });

  test("reverb leaves a decaying tail and stays bounded", () => {
    const impulse = new Float32Array(SR);
    impulse[0] = 1;
    const out = reverbAudio({ channels: [impulse], sampleRate: SR }, 1.5, 0.5);
    const tail = (lo: number, hi: number) => {
      let sum = 0;
      for (let i = Math.round(lo * SR); i < Math.round(hi * SR); i++) sum += out.channels[0][i] ** 2;
      return sum;
    };
    expect(tail(0.02, 0.2)).toBeGreaterThan(1e-6);
    // The dry impulse is silent after sample 0, so any later energy is the tail,
    // and it must fall rather than grow.
    expect(tail(0.2, 0.5)).toBeGreaterThan(tail(0.5, 0.9));
    let peak = 0;
    for (const v of out.channels[0]) peak = Math.max(peak, Math.abs(v));
    expect(peak).toBeLessThan(1.5);
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
      ["Make it darker", "tone"],
      ["Make this sound darker", "tone"],
      ["Make it brighter", "tone"],
      ["Make this more energetic", "tone"],
      ["Calm it down", "tone"],
      ["Make it lofi", "tone"],
      ["Add reverb", "reverb"],
      ["Add reverb to make it sound like a cathedral", "reverb"],
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
    expect(parseLocalPrompt("Speed up by 20%", 60).params.factor).toBeCloseTo(1.2, 5);
    expect(parseLocalPrompt("Slow it down by 25%", 60).params.factor).toBeCloseTo(0.75, 5);
    expect(parseLocalPrompt("Set speed to 150%", 60).params.factor).toBeCloseTo(1.5, 5);
  });

  test("exact BPM targets are not treated as a default 1.25x speed", () => {
    const target = parseLocalPrompt("Set tempo to 128 BPM", 60);
    expect(target.intent).toBe("speed");
    expect(target.params.targetBpm).toBe(128);
    expect(target.params.factor).toBeUndefined();

    expect(parseLocalPrompt("Speed up to 140 BPM", 60).params.targetBpm).toBe(140);
    expect(parseLocalPrompt("Slow down to 90 BPM", 60).params.targetBpm).toBe(90);
    expect(parseLocalPrompt("What BPM is this?", 60).intent).toBe("info");
    expect(parseLocalPrompt("Detect the tempo without changing anything", 60).intent).toBe("info");
  });

  test("localSupport marks server-only actions", () => {
    expect(localSupport("Separate into stems").supported).toBe(false);
    expect(localSupport("Split this interview by speaker").supported).toBe(false);
    expect(localSupport("Trim from 0:01 to 0:02").supported).toBe(true);
    expect(localSupport("Set tempo to 128 BPM").supported).toBe(true);
    expect(localSupport("What BPM is this?").supported).toBe(true);
    expect(localSupport("Remove all long pauses and silences longer than 0.5 seconds").supported).toBe(true);
    expect(localSupport("Cut pauses over 1s").supported).toBe(true);
    expect(localSupport("Remove filler words").supported).toBe(false);
    expect(localSupport("Remove the section from 0:02 to 0:04").supported).toBe(true);
    expect(localSupport("Pitch up 2 semitones").supported).toBe(true);
    expect(localSupport("Loop from 0:15 to 0:30 3 times").supported).toBe(true);
  });

  test("boost keeps its direction", () => {
    expect(parseLocalPrompt("Boost the vocals", 60).params.direction).toBe("up");
    expect(parseLocalPrompt("Lower the bass", 60).params.direction).toBe("down");
  });

  test("style conversion is refused instead of silently exported as WAV", () => {
    const plan = parseLocalPrompt("Convert this song into a house music style", 180);
    expect(plan.intent).toBe("unknown");
    expect(plan.unsupported).toMatch(/needs the server/i);

    // A real format request still routes to convert.
    expect(parseLocalPrompt("Convert to mp3", 60).intent).toBe("convert");
    expect(parseLocalPrompt("Export as WAV", 60).intent).toBe("convert");
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

describe("worker handoff", () => {
  test("applyPlan reports monotonic progress and ends at 100%", () => {
    const audio: LocalAudio = { sampleRate: SR, channels: [sine(440, 0.5, 0.4)] };
    const plan = parseLocalPrompt("Remove the background noise", 0.5);
    const seen: number[] = [];
    applyPlan(audio, plan, (fraction) => seen.push(fraction));

    expect(seen.length).toBeGreaterThan(1);
    for (let i = 1; i < seen.length; i++) {
      // A progress bar that goes backwards is worse than no progress bar.
      expect(seen[i]).toBeGreaterThanOrEqual(seen[i - 1]);
    }
    expect(seen[seen.length - 1]).toBe(1);
    for (const fraction of seen) {
      expect(fraction).toBeGreaterThanOrEqual(0);
      expect(fraction).toBeLessThanOrEqual(1);
    }
  });

  test("async client falls back to the main thread when there is no window", async () => {
    // Bun has no `window`, so this exercises the exact code path an old
    // browser would take: same audio, just computed on the calling thread.
    expect(typeof window).toBe("undefined");
    const audio: LocalAudio = {
      sampleRate: SR,
      channels: [Float32Array.from([0, 0.1, 0.2, 0.3])],
    };
    const plan: LocalPlan = { intent: "reverse", params: {}, summary: "Reversed" };
    const result = await runLocalPlanAsync(audio, plan);
    const out = Array.from(result.audio.channels[0]);
    expect(out.length).toBe(4);
    expect(out[0]).toBeCloseTo(0.3, 5);
    expect(out[1]).toBeCloseTo(0.2, 5);
    expect(out[2]).toBeCloseTo(0.1, 5);
    expect(out[3]).toBeCloseTo(0, 5);
    expect(result.plan.summary).toBe("Reversed");
  });

  test("encodeWavBuffer and encodeWav produce the same bytes", async () => {
    const audio: LocalAudio = { sampleRate: SR, channels: [sine(440, 0.05, 0.5)] };
    const buffer = encodeWavBuffer(audio);
    const blobBytes = new Uint8Array(await encodeWav(audio).arrayBuffer());
    expect(buffer.byteLength).toBe(blobBytes.byteLength);
    expect(new Uint8Array(buffer)).toEqual(blobBytes);
  });
});
