/**
 * "Fix my recording": measure what is wrong with a file, then repair it.
 *
 * Everything here is deterministic signal processing that runs on the user's
 * device. Nothing in this file is a trained model. The measurements are honest
 * estimates from the samples themselves, and every number the UI shows comes
 * from a function below — no invented values.
 *
 * The diagnosis is deliberately limited to what this build can actually fix:
 * broadband noise, hard clipping and level. Room echo is *not* measured here,
 * because there is no dereverberator yet and reporting a fault we cannot repair
 * would be dishonest.
 */

import { denoiseChannel, percentile } from "./denoise";
import type { LocalAudio } from "./types";

/** Above this magnitude a sample is treated as a clipped (flat-topped) peak. */
export const CLIP_LEVEL = 0.9999;

/** Target peak for the final normalise step, in dBFS. */
export const TARGET_PEAK_DB = -1;

const FRAME_SECONDS = 0.02;
const EPS = 1e-12;

function toDb(linear: number): number {
  return 20 * Math.log10(Math.max(linear, EPS));
}

function fromDb(db: number): number {
  return Math.pow(10, db / 20);
}

function clamp(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value;
}

/** Mono downmix used for level analysis only. */
export function downmix(audio: LocalAudio): Float32Array {
  const channels = audio.channels.length;
  if (channels === 0) return new Float32Array(0);
  if (channels === 1) return audio.channels[0];
  const length = audio.channels[0].length;
  const out = new Float32Array(length);
  for (let c = 0; c < channels; c++) {
    const channel = audio.channels[c];
    for (let i = 0; i < length; i++) out[i] += channel[i] ?? 0;
  }
  for (let i = 0; i < length; i++) out[i] /= channels;
  return out;
}

export function peakDbOf(audio: LocalAudio): number {
  let peak = 0;
  for (const channel of audio.channels) {
    for (let i = 0; i < channel.length; i++) {
      const v = Math.abs(channel[i]);
      if (v > peak) peak = v;
    }
  }
  return toDb(peak);
}

export function rmsDbOf(audio: LocalAudio): number {
  const mono = downmix(audio);
  if (mono.length === 0) return -Infinity;
  let sum = 0;
  for (let i = 0; i < mono.length; i++) sum += mono[i] * mono[i];
  return toDb(Math.sqrt(sum / mono.length));
}

export interface ClippingReport {
  clippedSamples: number;
  totalSamples: number;
  /** Percentage of all samples sitting on a flat-topped plateau. */
  percent: number;
  /** Number of separate clipped stretches. */
  runs: number;
  /** Longest clipped stretch, in samples. */
  longestRun: number;
}

/**
 * Find flat-topped plateaus at full scale.
 *
 * Level alone is not enough: a restored clip is still louder than the clip
 * level but follows a curve, and a legitimate master can touch 0 dBFS. What
 * marks real clipping is a *run of identical samples* pinned at the ceiling,
 * which is what a converter produces when the input exceeds its range.
 */
export function detectClipping(
  audio: LocalAudio,
  threshold = CLIP_LEVEL,
  minRun = 3,
): ClippingReport {
  let clippedSamples = 0;
  let totalSamples = 0;
  let runs = 0;
  let longestRun = 0;
  const tolerance = 1e-6;

  for (const channel of audio.channels) {
    totalSamples += channel.length;
    let runStart = -1;
    let flat = true;

    const closeRun = (end: number) => {
      if (runStart < 0) return;
      const length = end - runStart;
      if (length >= minRun && flat) {
        clippedSamples += length;
        runs++;
        if (length > longestRun) longestRun = length;
      }
      runStart = -1;
      flat = true;
    };

    for (let i = 0; i < channel.length; i++) {
      const value = channel[i];
      if (Math.abs(value) >= threshold) {
        if (runStart < 0) {
          runStart = i;
          flat = true;
        } else if (Math.abs(value - channel[runStart]) > tolerance) {
          flat = false;
        }
      } else {
        closeRun(i);
      }
    }
    closeRun(channel.length);
  }

  return {
    clippedSamples,
    totalSamples,
    percent: totalSamples > 0 ? (clippedSamples / totalSamples) * 100 : 0,
    runs,
    longestRun,
  };
}

export interface FrameLevels {
  /** 10th percentile frame level: the quiet part of the file. */
  noiseFloorDb: number;
  /** 90th percentile frame level: the loud, active part. */
  activeLevelDb: number;
  /** Spread between the two. Small values mean a continuous signal. */
  dynamicRangeDb: number;
}

/**
 * Frame-percentile level estimate. With pauses in the signal the quiet frames
 * expose the noise floor; a continuously loud file has no pauses, so the
 * estimate is flagged as unreliable by callers instead of being trusted.
 */
export function frameLevels(audio: LocalAudio): FrameLevels {
  const mono = downmix(audio);
  const frame = Math.max(64, Math.round(FRAME_SECONDS * audio.sampleRate));
  if (mono.length < frame) {
    const level = toDb(Math.sqrt(mono.reduce((s, v) => s + v * v, 0) / Math.max(1, mono.length)));
    return { noiseFloorDb: level, activeLevelDb: level, dynamicRangeDb: 0 };
  }

  const hop = Math.max(1, frame >> 1);
  const levels: number[] = [];
  for (let start = 0; start + frame <= mono.length; start += hop) {
    let sum = 0;
    for (let i = 0; i < frame; i++) sum += mono[start + i] * mono[start + i];
    levels.push(toDb(Math.sqrt(sum / frame)));
  }

  const sorted = Float32Array.from(levels);
  const noiseFloorDb = percentile(sorted, 10);
  const activeLevelDb = percentile(sorted, 90);
  return {
    noiseFloorDb,
    activeLevelDb,
    dynamicRangeDb: Math.max(0, activeLevelDb - noiseFloorDb),
  };
}

export interface CleanDiagnosis {
  durationSeconds: number;
  sampleRate: number;
  channels: number;
  peakDb: number;
  rmsDb: number;
  noiseFloorDb: number;
  activeLevelDb: number;
  /** Active level minus noise floor, clamped to 0..80 dB. */
  snrDb: number;
  /** False when the file has no pauses, so the noise floor cannot be trusted. */
  noiseReliable: boolean;
  clipping: ClippingReport;
  dcOffset: number;
  /** 0..1 problem scores, used for the bars in the UI. */
  severity: { noise: number; clipping: number; level: number };
}

export function diagnose(audio: LocalAudio): CleanDiagnosis {
  const levels = frameLevels(audio);
  const snrDb = clamp(levels.activeLevelDb - levels.noiseFloorDb, 0, 80);
  const peakDb = peakDbOf(audio);
  const clipping = detectClipping(audio);

  const mono = downmix(audio);
  let sum = 0;
  for (let i = 0; i < mono.length; i++) sum += mono[i];
  const dcOffset = mono.length > 0 ? sum / mono.length : 0;

  // A file needs about 6 dB of spread before "the quietest frames" means
  // anything other than "the quiet parts of the same continuous sound".
  const noiseReliable = levels.dynamicRangeDb >= 6 && levels.activeLevelDb > -70;

  return {
    durationSeconds: audio.sampleRate > 0 ? (audio.channels[0]?.length ?? 0) / audio.sampleRate : 0,
    sampleRate: audio.sampleRate,
    channels: audio.channels.length,
    peakDb,
    rmsDb: rmsDbOf(audio),
    noiseFloorDb: levels.noiseFloorDb,
    activeLevelDb: levels.activeLevelDb,
    snrDb,
    noiseReliable,
    clipping,
    dcOffset,
    severity: {
      noise: noiseReliable ? clamp((30 - snrDb) / 25, 0, 1) : 0,
      clipping: clamp(Math.log10(1 + clipping.percent * 1000) / 3, 0, 1),
      level: clamp((TARGET_PEAK_DB - peakDb) / 12, 0, 1),
    },
  };
}

/**
 * Restore flat-topped runs with a cubic Hermite curve through the surviving
 * neighbours. This is the textbook baseline for declipping: cheap, deterministic
 * and honest about being an interpolation rather than a reconstruction of the
 * lost peak.
 */
export function declipChannel(x: Float32Array, threshold = CLIP_LEVEL): Float32Array {
  const n = x.length;
  const out = Float32Array.from(x);
  if (n === 0) return out;

  let i = 0;
  while (i < n) {
    if (Math.abs(x[i]) < threshold) {
      i++;
      continue;
    }

    const start = i;
    while (i < n && Math.abs(x[i]) >= threshold) i++;
    const end = i; // exclusive
    const length = end - start;

    const left = start - 1;
    const right = end;

    // A run touching the first or last sample has no two-sided context; fade
    // it toward zero instead of inventing a slope.
    if (left < 0 || right >= n) {
      const anchor = left >= 0 ? x[left] : right < n ? x[right] : 0;
      const steps = length + 1;
      for (let k = 0; k < length; k++) {
        const t = (k + 1) / steps;
        const value = start === 0 ? anchor * (1 - t) : anchor * t;
        out[start + k] = clamp(value, -1, 1);
      }
      continue;
    }

    const y0 = x[left];
    const y1 = x[right];
    // Central-difference tangents, scaled so the curve spans the gap.
    const m0 = left > 0 ? (x[left] - x[left - 1]) * 0.5 : 0;
    const m1 = right + 1 < n ? (x[right + 1] - x[right]) * 0.5 : 0;
    const steps = length + 1;

    for (let k = 1; k <= length; k++) {
      const t = k / steps;
      const t2 = t * t;
      const t3 = t2 * t;
      const h00 = 2 * t3 - 3 * t2 + 1;
      const h10 = t3 - 2 * t2 + t;
      const h01 = -2 * t3 + 3 * t2;
      const h11 = t3 - t2;
      const value = h00 * y0 + h10 * m0 * steps + h01 * y1 + h11 * m1 * steps;
      // Keep the restored sample representable; the final normalise step sets
      // the ceiling, so allow a little overshoot past the clip level here.
      out[start + k - 1] = clamp(value, -1.2, 1.2);
    }
  }

  return out;
}

export function declipAudio(audio: LocalAudio, threshold = CLIP_LEVEL): LocalAudio {
  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((channel) => declipChannel(channel, threshold)),
  };
}

/** Scale the whole file so its loudest sample sits at `targetDb` dBFS. */
export function normalizePeak(audio: LocalAudio, targetDb = TARGET_PEAK_DB): LocalAudio {
  let peak = 0;
  for (const channel of audio.channels) {
    for (let i = 0; i < channel.length; i++) {
      const v = Math.abs(channel[i]);
      if (v > peak) peak = v;
    }
  }
  if (peak <= 0) return audio;
  const gain = fromDb(targetDb) / peak;
  if (Math.abs(gain - 1) < 1e-9) return audio;
  return {
    sampleRate: audio.sampleRate,
    channels: audio.channels.map((channel) => {
      const out = new Float32Array(channel.length);
      for (let i = 0; i < channel.length; i++) {
        out[i] = clamp(channel[i] * gain, -1, 1);
      }
      return out;
    }),
  };
}

export type CleanStepKey = "declip" | "denoise" | "normalize";

export interface CleanStep {
  key: CleanStepKey;
  label: string;
  /** Why this step was chosen, in one sentence, from the measurements. */
  detail: string;
}

export interface CleanPlan {
  steps: CleanStep[];
  /** Reasons a step was skipped, shown so the choice is inspectable. */
  skipped: string[];
}

/**
 * Turn a diagnosis into a repair chain.
 *
 * Order matters and is explainable: declipping runs before denoising because a
 * flat-topped peak is a broadband click that a spectral denoiser would smear
 * instead of remove. Normalising is always last so it sees the final peaks.
 */
export function planClean(diagnosis: CleanDiagnosis): CleanPlan {
  const steps: CleanStep[] = [];
  const skipped: string[] = [];

  const clipped = diagnosis.clipping.percent >= 0.001 || diagnosis.clipping.longestRun >= 32;
  if (clipped) {
    steps.push({
      key: "declip",
      label: "Declip",
      detail: `${diagnosis.clipping.percent.toFixed(2)}% of samples are flat-topped — restore them before anything spectral touches the file.`,
    });
  } else {
    skipped.push("No clipping found, so the samples are left untouched.");
  }

  if (diagnosis.noiseReliable && diagnosis.snrDb < 25) {
    steps.push({
      key: "denoise",
      label: "Denoise",
      detail: `Signal is about ${diagnosis.snrDb.toFixed(0)} dB above the quiet frames — reduce the stationary noise.`,
    });
  } else if (!diagnosis.noiseReliable) {
    skipped.push("The file has no pauses, so the noise floor cannot be measured — denoise skipped rather than guessed.");
  } else {
    skipped.push("Noise floor is already more than 25 dB down — denoise skipped.");
  }

  steps.push({
    key: "normalize",
    label: "Normalise",
    detail: `Peak is ${diagnosis.peakDb.toFixed(1)} dBFS — set it to ${TARGET_PEAK_DB} dBFS for a predictable export.`,
  });

  return { steps, skipped };
}

export interface CleanResult {
  audio: LocalAudio;
  before: CleanDiagnosis;
  after: CleanDiagnosis;
  plan: CleanPlan;
  /** Measured SNR gain, in dB. */
  snrGainDb: number;
}

export type CleanProgress = (fraction: number, label: string) => void;

export function applyCleanPlan(
  audio: LocalAudio,
  before: CleanDiagnosis,
  plan: CleanPlan,
  onProgress?: CleanProgress,
): CleanResult {
  let current = audio;
  const total = Math.max(1, plan.steps.length);

  plan.steps.forEach((step, index) => {
    const start = index / total;
    const span = 1 / total;
    if (step.key === "declip") {
      onProgress?.(start, "Restoring clipped peaks");
      current = declipAudio(current);
    } else if (step.key === "denoise") {
      current = {
        sampleRate: current.sampleRate,
        channels: current.channels.map((channel) =>
          denoiseChannel(channel, current.sampleRate, 2, undefined, undefined, (fraction) =>
            onProgress?.(start + span * fraction, "Reducing noise"),
          ),
        ),
      };
    } else {
      onProgress?.(start, `Normalising to ${TARGET_PEAK_DB} dBFS`);
      current = normalizePeak(current);
    }
    onProgress?.(start + span, step.label);
  });

  const after = diagnose(current);
  return { audio: current, before, after, plan, snrGainDb: after.snrDb - before.snrDb };
}
