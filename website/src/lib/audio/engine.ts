/**
 * The browser-only engine: plan -> audio -> WAV blob.
 *
 * Everything here runs on the user's machine. The only I/O is the WAV encode at
 * the end, which is why the deployed site can work with no backend at all.
 */

import {
  applyGain,
  boostTarget,
  cloneAudio,
  denoiseAudio,
  fade,
  normalize,
  reverse,
  reverbAudio,
  shapeTone,
  stretchAudio,
  trim,
  type ToneKind,
} from "./ops";
import { durationOf, encodeWav, peakOf } from "./wav";
import type { LocalAudio, LocalPlan, LocalResult } from "./types";

/** Yield once so React can paint the "processing" state before the DSP starts. */
function nextTick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function rms(audio: LocalAudio): number {
  let sum = 0;
  let count = 0;
  for (const channel of audio.channels) {
    for (let i = 0; i < channel.length; i++) {
      sum += channel[i] * channel[i];
      count++;
    }
  }
  return count ? Math.sqrt(sum / count) : 0;
}

export interface LocalAnalysis {
  durationSeconds: number;
  sampleRate: number;
  channels: number;
  peak: number;
  rms: number;
}

export function analyzeLocal(audio: LocalAudio): LocalAnalysis {
  return {
    durationSeconds: durationOf(audio),
    sampleRate: audio.sampleRate,
    channels: audio.channels.length,
    peak: peakOf(audio),
    rms: rms(audio),
  };
}

export type LocalProgress = (fraction: number, label: string) => void;

export interface PlanOutput {
  audio: LocalAudio;
  metadata: Record<string, unknown>;
  summary: string;
}

/**
 * Apply a plan with no I/O: channel buffers in, channel buffers out.
 *
 * Split from `runLocalPlan` so the same code can run inside a Web Worker (where
 * creating a Blob is fine but the UI thread must stay free) and in tests.
 * `onProgress` is 0..1 and may fire thousands of times; callers throttle.
 */
export function applyPlan(source: LocalAudio, plan: LocalPlan, onProgress?: LocalProgress): PlanOutput {
  // Spectral operations know how far along their STFT is; time-domain ones are
  // effectively instant, so they report nothing until the end.
  const spectral = (label: string) =>
    onProgress ? (fraction: number) => onProgress(fraction, label) : undefined;

  const duration = durationOf(source);
  let audio: LocalAudio = cloneAudio(source);
  let summary = plan.summary;
  const metadata: Record<string, unknown> = { local: true, method: "browser_dsp" };

  switch (plan.intent) {
    case "trim": {
      const start = Math.max(0, Math.min(duration, Number(plan.params.start ?? 0)));
      const end = Math.max(start, Math.min(duration, Number(plan.params.end ?? duration)));
      audio = trim(source, start, end);
      metadata.start_seconds = start;
      metadata.end_seconds = end;
      metadata.duration_seconds = durationOf(audio);
      break;
    }
    case "fade": {
      const inSec = Number(plan.params.fade_in_seconds ?? 0);
      const outSec = Number(plan.params.fade_out_seconds ?? 0);
      audio = fade(source, inSec, outSec);
      metadata.fade_in_seconds = inSec;
      metadata.fade_out_seconds = outSec;
      break;
    }
    case "gain": {
      const result = applyGain(source, Number(plan.params.db ?? 0));
      audio = result.audio;
      metadata.gain_db = Number(result.appliedDb.toFixed(2));
      if (Math.abs(result.appliedDb - Number(plan.params.db ?? 0)) > 0.05) {
        summary += ` (pull-back to avoid clipping: ${result.appliedDb > 0 ? "+" : ""}${result.appliedDb.toFixed(1)} dB applied)`;
      }
      break;
    }
    case "normalize": {
      const result = normalize(source, -1);
      audio = result.audio;
      metadata.gain_db = Number(result.gainDb.toFixed(2));
      summary = `Normalised to −1.0 dBFS peak (${result.gainDb >= 0 ? "+" : ""}${result.gainDb.toFixed(1)} dB)`;
      break;
    }
    case "reverse": {
      audio = reverse(source);
      metadata.reversed = true;
      break;
    }
    case "speed": {
      const factor = Number(plan.params.factor ?? 1);
      // `factor` is a speed multiplier (1.25 = 25% faster); the phase vocoder
      // takes a stretch factor, so it is the reciprocal. Getting this backwards
      // is exactly the bug the browser smoke test caught.
      const stretch = 1 / factor;
      audio = stretchAudio(source, stretch, spectral("Changing speed"));
      metadata.speed_factor = factor;
      metadata.stretch_factor = stretch;
      metadata.duration_seconds = durationOf(audio);
      break;
    }
    case "denoise": {
      audio = denoiseAudio(source, Number(plan.params.strength ?? 2), spectral("Denoising"));
      metadata.enhanced = "voice";
      metadata.method = "spectral_denoise";
      break;
    }
    case "boost": {
      const target = String(plan.params.target ?? "vocals") as "vocals" | "drums" | "bass" | "other";
      const direction = plan.params.direction === "down" ? "down" : "up";
      audio = boostTarget(source, target, direction, spectral(`${direction === "up" ? "Boosting" : "Cutting"} ${target}`));
      metadata.boosted = target;
      metadata.direction = direction;
      break;
    }
    case "tone": {
      const kind = String(plan.params.kind ?? "darker") as ToneKind;
      audio = shapeTone(source, kind, spectral("Shaping tone"));
      metadata.tone = kind;
      break;
    }
    case "reverb": {
      const seconds = Number(plan.params.seconds ?? 2.2);
      const mix = Number(plan.params.mix ?? 0.25);
      audio = reverbAudio(source, seconds, mix);
      metadata.reverb_seconds = seconds;
      metadata.reverb_mix = mix;
      break;
    }
    case "convert": {
      audio = cloneAudio(source);
      metadata.format = "wav";
      break;
    }
    default: {
      throw new Error("That edit is not available in browser mode.");
    }
  }

  // Keep the summary honest about the result, not just the request.
  if (plan.intent === "trim") {
    summary = `${summary} · ${durationOf(audio).toFixed(1)} s left`;
  }

  onProgress?.(1, "Done");
  return { audio, metadata, summary };
}

export async function runLocalPlan(
  source: LocalAudio,
  plan: LocalPlan,
  onProgress?: LocalProgress,
): Promise<LocalResult> {
  await nextTick();
  const output = applyPlan(source, plan, onProgress);
  onProgress?.(1, "Encoding WAV");
  return {
    audio: output.audio,
    plan: { ...plan, summary: output.summary },
    metadata: output.metadata,
    blob: encodeWav(output.audio),
  };
}

export * from "./types";
export { analyzeLocal as analyze };
export { encodeWav, durationOf } from "./wav";
export { decodeFile } from "./decode";
export { LOCAL_PROMPTS, normalizePrompt, parseLocalPrompt, parseTime } from "./parse";
