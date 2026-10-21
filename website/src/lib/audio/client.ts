/**
 * Main-thread client for the DSP worker.
 *
 * Falls back to running in-process when `Worker` is unavailable or the worker
 * cannot start (old browser, strict CSP, a bundler that did not emit it). The
 * fallback is slower and blocks the tab, but it still produces the right audio.
 */

import { runLocalPlan, type LocalProgress } from "./engine";
import type { LocalAudio, LocalPlan, LocalResult } from "./types";
import type { WorkerRequest, WorkerResponse } from "./worker";

let nextJobId = 0;

export function canUseWorker(): boolean {
  return typeof window !== "undefined" && typeof Worker !== "undefined";
}

export async function runLocalPlanAsync(
  source: LocalAudio,
  plan: LocalPlan,
  onProgress?: LocalProgress,
): Promise<LocalResult> {
  if (!canUseWorker()) return runLocalPlan(source, plan, onProgress);

  let worker: Worker;
  try {
    worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  } catch {
    return runLocalPlan(source, plan, onProgress);
  }

  const id = ++nextJobId;
  // Copy, because posting transfers the buffers and the caller still needs its
  // audio for the next edit if this one fails.
  const channels = source.channels.map((c) => c.slice().buffer as ArrayBuffer);

  return await new Promise<LocalResult>((resolve, reject) => {
    let settled = false;

    const stop = () => {
      worker.terminate();
    };

    // If the worker never starts, the same pure function can run here. Re-run
    // from the source rather than trying to salvage a partial job.
    const fallback = () => {
      if (settled) return;
      settled = true;
      stop();
      runLocalPlan(source, plan, onProgress).then(resolve, reject);
    };

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.id !== id) return;

      if (message.type === "progress") {
        onProgress?.(message.fraction, message.label);
        return;
      }
      if (settled) return;
      settled = true;
      stop();

      if (message.type === "done") {
        const audio: LocalAudio = {
          sampleRate: message.sampleRate,
          channels: message.channels.map((buffer) => new Float32Array(buffer)),
        };
        resolve({
          audio,
          plan: { ...plan, summary: message.summary },
          metadata: message.metadata,
          blob: new Blob([message.wav], { type: "audio/wav" }),
        });
      } else {
        reject(new Error(message.message));
      }
    };

    worker.onerror = fallback;

    const request: WorkerRequest = { id, channels, sampleRate: source.sampleRate, plan };
    worker.postMessage(request, channels);
  });
}
