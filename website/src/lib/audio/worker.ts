/**
 * Web Worker entry for the browser DSP.
 *
 * Every spectral edit is tens of millions of operations on a few minutes of
 * audio. Running that on the UI thread freezes the tab (measured: 7.7 s for a
 * 3-minute mono denoise), so the work happens here and the page only receives
 * progress messages and the finished bytes.
 *
 * The protocol is deliberately tiny: one request per worker, then terminate.
 * There is no shared state to reset between edits.
 */

import { applyPlan } from "./engine";
import { encodeWavBuffer } from "./wav";
import type { LocalAudio, LocalPlan } from "./types";

export interface WorkerRequest {
  id: number;
  /** Transferable copies of the source channels. */
  channels: ArrayBuffer[];
  sampleRate: number;
  plan: LocalPlan;
}

export type WorkerResponse =
  | { type: "progress"; id: number; fraction: number; label: string }
  | {
      type: "done";
      id: number;
      wav: ArrayBuffer;
      channels: ArrayBuffer[];
      sampleRate: number;
      metadata: Record<string, unknown>;
      summary: string;
    }
  | { type: "error"; id: number; message: string };

// The DOM lib types `self` as Window, whose postMessage signature differs from
// a worker's. A local interface is smaller than pulling in the webworker lib
// next to DOM (which duplicates identifiers).
interface WorkerScope {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage: (message: WorkerResponse, transfer?: Transferable[]) => void;
}

const ctx = self as unknown as WorkerScope;

ctx.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const { id, channels, sampleRate, plan } = event.data;
  try {
    const audio: LocalAudio = {
      sampleRate,
      channels: channels.map((buffer) => new Float32Array(buffer)),
    };

    let lastFraction = -1;
    const output = applyPlan(audio, plan, (fraction, label) => {
      // One state update per percent is plenty; the STFT reports every 256
      // frames, which would be hundreds of messages for a long file.
      if (fraction - lastFraction < 0.02 && fraction < 1) return;
      lastFraction = fraction;
      ctx.postMessage({ type: "progress", id, fraction, label });
    });

    ctx.postMessage({ type: "progress", id, fraction: 1, label: "Encoding WAV" });
    const wav = encodeWavBuffer(output.audio);
    const outChannels = output.audio.channels.map((c) => c.buffer as ArrayBuffer);

    const done: WorkerResponse = {
      type: "done",
      id,
      wav,
      channels: outChannels,
      sampleRate: output.audio.sampleRate,
      metadata: output.metadata,
      summary: output.summary,
    };
    ctx.postMessage(done, [wav, ...outChannels]);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Processing failed";
    ctx.postMessage({ type: "error", id, message });
  }
};
