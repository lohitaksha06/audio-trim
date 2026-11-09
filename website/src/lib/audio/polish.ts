/**
 * Deterministic "polish" chain for a finished mix.
 *
 * This is not a mastering model and it does not know what "perfect" means. It
 * maps a plain-language request onto the four operations that actually exist in
 * the browser: noise reduction, reverb, peak normalization, and a safety gain.
 * The caller gets a report of exactly what ran.
 */

import { denoiseAudio, normalize, reverbAudio } from "./ops";
import type { LocalAudio } from "./types";

export interface PolishOptions {
  denoise: boolean;
  /** Wiener strength, 1–3. */
  denoiseStrength: number;
  /** 0 disables reverb. */
  reverbMix: number;
  reverbSeconds: number;
  /** Normalize the final peak. */
  normalize: boolean;
  targetPeakDb: number;
}

export interface PolishResult {
  audio: LocalAudio;
  report: string[];
  gainDb: number;
}

const NOISE = /\b(noise|noisy|hiss|hum|static|denoise|de-noise|background\s+(?:noise|sound|hum))\b/;
const REVERB = /\b(reverb|room|hall|space|ambien|cathedral)\b/;
const LEVEL = /\b(volume|level|loud|normaliz|normalis|louder|quieter|quiet)\b/;
const POLISH = /\b(perfect|polish|radio[- ]ready|release[- ]ready|master(?:ing)?|clean\s+up|cleanup|fix\s+(?:the\s+)?(?:mix|track|song|recording|audio))\b/;

function strengthFrom(text: string): number {
  if (/(strong|heavy|aggressive|very noisy|lots of noise|lot of noise)/.test(text)) return 3;
  if (/(a little|light|gentle|subtle|slight|mild)/.test(text)) return 1;
  return 2;
}

function reverbFrom(text: string): { mix: number; seconds: number } {
  if (/(a little|light|subtle|slight|tiny|hint of)/.test(text)) return { mix: 0.12, seconds: 1.2 };
  if (/(lots?|heavy|big|huge|wash|cathedral|hall)/.test(text)) return { mix: 0.32, seconds: 2.8 };
  return { mix: 0.2, seconds: 1.8 };
}

/**
 * Recognize a cleanup request. A single "add reverb" or "remove the noise"
 * stays a single-purpose edit; a combination, or an explicit polish word,
 * becomes the chain.
 */
export function parsePolishPrompt(text: string): PolishOptions | null {
  const lower = text.toLowerCase().replace(/\s+/g, " ").trim();
  if (!lower) return null;
  const noise = NOISE.test(lower);
  const reverb = REVERB.test(lower);
  const level = LEVEL.test(lower);
  const trigger = POLISH.test(lower);
  const effects = [noise, reverb, level].filter(Boolean).length;
  if (!trigger && effects < 2) return null;

  const space = reverb ? reverbFrom(lower) : { mix: 0, seconds: 0 };
  return {
    denoise: noise || trigger,
    denoiseStrength: strengthFrom(lower),
    reverbMix: reverb ? space.mix : 0,
    reverbSeconds: reverb ? space.seconds : 0,
    normalize: level || trigger,
    targetPeakDb: -1,
  };
}

/** Snake-case params for a LocalPlan, matching the engine's conventions. */
export function polishParams(options: PolishOptions): Record<string, unknown> {
  return {
    denoise: options.denoise,
    denoise_strength: options.denoiseStrength,
    reverb_mix: options.reverbMix,
    reverb_seconds: options.reverbSeconds,
    normalize: options.normalize,
    target_peak_db: options.targetPeakDb,
  };
}

export function describePolish(options: PolishOptions): string {
  const steps: string[] = [];
  if (options.denoise) steps.push(`noise reduction (strength ${options.denoiseStrength.toFixed(1)})`);
  if (options.reverbMix > 0) {
    steps.push(`reverb (${options.reverbSeconds.toFixed(1)} s, ${Math.round(options.reverbMix * 100)}% wet)`);
  }
  if (options.normalize) steps.push(`peak normalized to ${options.targetPeakDb.toFixed(1)} dBFS`);
  return steps.length ? `Polished: ${steps.join(", ")}.` : "Nothing to polish.";
}

/**
 * Run the chain: denoise, reverb, then a final peak stage.
 *
 * Normalization is opt-in. If the user only asked for reverb, the level is left
 * alone unless the result would clip, in which case it is scaled back and the
 * report says so.
 */
export function polishAudio(
  audio: LocalAudio,
  options: PolishOptions,
  onProgress?: (fraction: number) => void,
): PolishResult {
  let out = audio;
  const report: string[] = [];

  if (options.denoise) {
    out = denoiseAudio(out, options.denoiseStrength, onProgress);
    report.push(`Noise reduction (Wiener, strength ${options.denoiseStrength.toFixed(1)})`);
  }

  if (options.reverbMix > 0) {
    out = reverbAudio(out, options.reverbSeconds, options.reverbMix);
    report.push(
      `Reverb (${options.reverbSeconds.toFixed(1)} s tail, ${Math.round(options.reverbMix * 100)}% wet)`,
    );
  }

  let gainDb = 0;
  if (options.normalize) {
    const result = normalize(out, options.targetPeakDb);
    out = result.audio;
    gainDb = result.gainDb;
    report.push(
      `Peak normalized to ${options.targetPeakDb.toFixed(1)} dBFS (${gainDb >= 0 ? "+" : ""}${gainDb.toFixed(1)} dB)`,
    );
  } else {
    let peak = 0;
    for (const channel of out.channels) {
      for (let i = 0; i < channel.length; i++) peak = Math.max(peak, Math.abs(channel[i]));
    }
    if (peak > 0.999) {
      const scale = 0.99 / peak;
      out = {
        sampleRate: out.sampleRate,
        channels: out.channels.map((channel) => {
          const next = new Float32Array(channel.length);
          for (let i = 0; i < channel.length; i++) next[i] = channel[i] * scale;
          return next;
        }),
      };
      gainDb = 20 * Math.log10(scale);
      report.push(`Safety gain ${gainDb.toFixed(1)} dB to stop clipping`);
    }
  }

  return { audio: out, report, gainDb };
}
