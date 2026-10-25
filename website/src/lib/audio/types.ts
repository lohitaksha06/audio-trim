/** Shared types for the in-browser audio engine. */

export interface LocalAudio {
  /** One Float32Array per channel, all the same length, nominally -1..1. */
  channels: Float32Array[];
  sampleRate: number;
}

export type LocalIntent =
  | "trim"
  | "fade"
  | "gain"
  | "normalize"
  | "reverse"
  | "remove"
  | "pitch"
  | "loop"
  | "speed"
  | "denoise"
  | "boost"
  | "tone"
  | "reverb"
  | "silence"
  | "convert"
  | "lab"
  | "info"
  | "unknown";

export interface LocalPlan {
  intent: LocalIntent;
  params: Record<string, unknown>;
  /** Sentence shown in the chat log. Says what actually ran. */
  summary: string;
  /** Set when the browser cannot do this; the UI shows it verbatim. */
  unsupported?: string;
}

export interface LocalResult {
  audio: LocalAudio;
  plan: LocalPlan;
  metadata: Record<string, unknown>;
  /** 16-bit WAV, ready for an object URL. */
  blob: Blob;
}
