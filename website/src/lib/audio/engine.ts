/**
 * The browser-only engine: plan -> audio -> WAV blob.
 *
 * Everything here runs on the user's machine. The only I/O is the WAV encode at
 * the end, which is why the deployed site can work with no backend at all.
 */

import {
  applyEq,
  applyGain,
  boostTarget,
  cloneAudio,
  concatAudio,
  denoiseAudio,
  fade,
  normalize,
  pitchShiftAudio,
  removeSilenceAudio,
  reverse,
  reverbAudio,
  shapeTone,
  stretchAudio,
  trim,
  type ToneKind,
} from "./ops";
import { estimateBrightness, estimateTempo } from "./tempo";
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
  /** Null when there is no periodic onset structure (a pad, a sine, silence). */
  bpm: number | null;
  bpmConfidence: number;
  /** Average spectral centroid in Hz, a measured brightness descriptor. */
  brightnessHz: number;
}

export function analyzeLocal(audio: LocalAudio): LocalAnalysis {
  const tempo = estimateTempo(audio);
  return {
    durationSeconds: durationOf(audio),
    sampleRate: audio.sampleRate,
    channels: audio.channels.length,
    peak: peakOf(audio),
    rms: rms(audio),
    bpm: tempo?.bpm ?? null,
    bpmConfidence: tempo?.confidence ?? 0,
    brightnessHz: estimateBrightness(audio),
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
    case "remove": {
      const start = Math.max(0, Math.min(duration, Number(plan.params.start ?? 0)));
      const end = Math.max(start, Math.min(duration, Number(plan.params.end ?? duration)));
      if (end <= start) throw new Error("That section has no length to remove.");
      if (end - start >= duration - 1e-6) {
        throw new Error("That range covers the whole file, so there would be no audio left. Choose a shorter range.");
      }
      audio = concatAudio([trim(source, 0, start), trim(source, end, duration)], 0.01);
      metadata.removed_start_seconds = start;
      metadata.removed_end_seconds = end;
      metadata.removed_seconds = end - start;
      metadata.duration_seconds = durationOf(audio);
      summary = `Removed ${(end - start).toFixed(1)} s (${start.toFixed(1)}-${end.toFixed(1)} s), 10 ms crossfade`;
      break;
    }
    case "pitch": {
      const semitones = Math.max(-12, Math.min(12, Number(plan.params.semitones ?? 0)));
      if (Math.abs(semitones) < 0.01) {
        audio = cloneAudio(source);
        summary = "Pitch unchanged (0 semitones)";
      } else {
        const shifted = pitchShiftAudio(source, semitones, spectral("Shifting pitch"));
        const normalised = normalize(shifted, -1);
        audio = normalised.audio;
        metadata.semitones = semitones;
        metadata.gain_db = Number(normalised.gainDb.toFixed(2));
        summary = `Pitch ${semitones > 0 ? "+" : ""}${semitones} semitones, tempo preserved`;
      }
      break;
    }
    case "loop": {
      const start = Math.max(0, Math.min(duration, Number(plan.params.start ?? 0)));
      const end = Math.max(start, Math.min(duration, Number(plan.params.end ?? duration)));
      const times = Math.max(2, Math.min(8, Number(plan.params.times ?? 2)));
      if (end <= start) throw new Error("That loop range has no length.");
      const section = trim(source, start, end);
      const segments: LocalAudio[] = [trim(source, 0, end)];
      for (let i = 1; i < times; i++) segments.push(section);
      const tail = trim(source, end, duration);
      if (tail.channels[0].length > 0) segments.push(tail);
      audio = concatAudio(segments, 0.01);
      metadata.loop_start_seconds = start;
      metadata.loop_end_seconds = end;
      metadata.loop_times = times;
      metadata.duration_seconds = durationOf(audio);
      summary = `Looped ${start.toFixed(1)}-${end.toFixed(1)} s ${times} times total (section ${(end - start).toFixed(1)} s)`;
      break;
    }
    case "speed": {
      let factor = Number(plan.params.factor ?? 1);
      if (plan.params.targetBpm != null) {
        const targetBpm = Number(plan.params.targetBpm);
        const measured = estimateTempo(source);
        if (!measured) {
          throw new Error(
            "I can't find a steady beat in this file, so I can't convert it to an exact BPM. " +
              'Use a multiplier instead, for example "Speed up 1.25x".',
          );
        }
        factor = targetBpm / measured.bpm;
        if (!Number.isFinite(factor) || factor < 0.25 || factor > 4) {
          throw new Error(
            `That would need a ${factor.toFixed(2)}x stretch (measured ${measured.bpm.toFixed(1)} BPM ` +
              `to ${targetBpm} BPM). Browser tempo changes are limited to 0.25x-4x.`,
          );
        }
        summary = `Tempo ${measured.bpm.toFixed(1)} BPM to ${targetBpm} BPM (x${factor.toFixed(2)}), pitch preserved`;
        metadata.measured_bpm = Number(measured.bpm.toFixed(2));
        metadata.target_bpm = targetBpm;
        metadata.bpm_confidence = Number(measured.confidence.toFixed(3));
      }
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
    case "silence": {
      const minSeconds = Math.max(0.05, Math.min(20, Number(plan.params.minSeconds ?? 0.5)));
      const cleaned = removeSilenceAudio(source, minSeconds, 0.08, spectral("Removing pauses"));
      audio = cleaned.audio;
      metadata.min_pause_seconds = minSeconds;
      metadata.pauses_removed = cleaned.pauses;
      metadata.removed_seconds = Number(cleaned.removedSeconds.toFixed(2));
      metadata.silence_threshold_db = Number(cleaned.thresholdDb.toFixed(1));
      metadata.duration_seconds = durationOf(audio);
      summary = cleaned.pauses
        ? `Removed ${cleaned.pauses} pause${cleaned.pauses === 1 ? "" : "s"} longer than ${minSeconds} s (${cleaned.removedSeconds.toFixed(1)} s total)`
        : `No pauses longer than ${minSeconds} s found`;
      break;
    }
    case "convert": {
      audio = cloneAudio(source);
      metadata.format = "wav";
      break;
    }
    case "lab": {
      // Mirrors the server's /api/lab/tweak order: EQ -> pitch -> speed ->
      // gain -> reverb. Segmented progress keeps the bar moving forward.
      const gains = (plan.params.gains ?? {}) as Record<string, number>;
      const highpassHz = plan.params.highpass == null ? null : Number(plan.params.highpass);
      const lowpassHz = plan.params.lowpass == null ? null : Number(plan.params.lowpass);
      const pitch = Number(plan.params.pitch ?? 0);
      const speed = Math.max(0.5, Math.min(2, Number(plan.params.speed ?? 1)));
      const db = Number(plan.params.gainDb ?? 0);
      const reverb = Math.max(0, Math.min(1, Number(plan.params.reverb ?? 0)));
      const stage = (start: number, span: number, label: string) =>
        onProgress
          ? (fraction: number) => onProgress(start + span * Math.max(0, Math.min(1, fraction)), label)
          : undefined;

      let out = applyEq(cloneAudio(source), gains, highpassHz, lowpassHz, stage(0, 0.45, "Shaping EQ"));
      if (Math.abs(pitch) >= 0.01) {
        out = pitchShiftAudio(out, pitch, stage(0.45, 0.25, "Shifting pitch"));
      }
      if (Math.abs(speed - 1) >= 0.001) {
        out = stretchAudio(out, 1 / speed, stage(0.7, 0.18, "Changing speed"));
      }
      if (Math.abs(db) >= 0.01) {
        const applied = applyGain(out, db);
        out = applied.audio;
        metadata.gain_db = Number(applied.appliedDb.toFixed(2));
      }
      if (reverb > 0) {
        onProgress?.(0.94, "Adding reverb");
        out = reverbAudio(out, 2.2, reverb);
      }

      audio = out;
      const parts: string[] = [];
      const activeBands = Object.keys(gains).filter((key) => Math.abs(Number(gains[key]) || 0) >= 0.01);
      if (activeBands.length) parts.push(`EQ (${activeBands.length} band${activeBands.length === 1 ? "" : "s"})`);
      if (highpassHz) parts.push(`high-pass ${Math.round(highpassHz)} Hz`);
      if (lowpassHz) parts.push(`low-pass ${Math.round(lowpassHz)} Hz`);
      if (Math.abs(pitch) >= 0.01) parts.push(`pitch ${pitch > 0 ? "+" : ""}${pitch} st`);
      if (Math.abs(speed - 1) >= 0.001) parts.push(`speed ${speed.toFixed(2)}x`);
      if (Math.abs(db) >= 0.01) parts.push(`gain ${db > 0 ? "+" : ""}${db.toFixed(1)} dB`);
      if (reverb > 0) parts.push(`reverb ${Math.round(reverb * 100)}%`);
      summary = parts.length ? `Lab render: ${parts.join(", ")}` : "Lab render: all controls are flat (no change)";
      metadata.lab_params = { gains, highpass: highpassHz, lowpass: lowpassHz, pitch, speed, gainDb: db, reverb };
      metadata.duration_seconds = durationOf(audio);
      break;
    }
    case "info": {
      const tempo = estimateTempo(source);
      if (tempo) {
        summary = `Tempo: ${tempo.bpm.toFixed(1)} BPM (confidence ${Math.round(tempo.confidence * 100)}%)`;
        metadata.bpm = Number(tempo.bpm.toFixed(2));
        metadata.bpm_confidence = Number(tempo.confidence.toFixed(3));
      } else {
        summary = "I couldn't find a steady tempo in this file (no periodic onsets).";
        metadata.bpm = null;
        metadata.bpm_confidence = 0;
      }
      metadata.brightness_hz = Math.round(estimateBrightness(source));
      metadata.info_only = true;
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
