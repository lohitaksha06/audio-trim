/**
 * Main-thread client for the DSP worker.
 *
 * Falls back to running in-process when `Worker` is unavailable or the worker
 * cannot start (old browser, strict CSP, a bundler that did not emit it). The
 * fallback is slower and blocks the tab, but it still produces the right audio.
 */

import { analyzeLocal, runLocalPlan, type LocalAnalysis, type LocalProgress } from "./engine";
import { mixAudio } from "./mix";
import { encodeWav } from "./wav";
import type { LocalAudio, LocalPlan, LocalResult } from "./types";
import type { WorkerRequest, WorkerResponse } from "./worker";

let nextJobId = 0;

declare global {
  interface Window {
    /** Diagnostics: which engine actually ran the last job. */
    __audelleDspMode?: "worker" | "main";
  }
}

export function canUseWorker(): boolean {
  return typeof window !== "undefined" && typeof Worker !== "undefined";
}

function noteMode(mode: "worker" | "main") {
  if (typeof window !== "undefined") window.__audelleDspMode = mode;
}

export async function runLocalPlanAsync(
  source: LocalAudio,
  plan: LocalPlan,
  onProgress?: LocalProgress,
): Promise<LocalResult> {
  if (!canUseWorker()) {
    noteMode("main");
    return runLocalPlan(source, plan, onProgress);
  }

  let worker: Worker;
  try {
    worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    noteMode("worker");
  } catch (error) {
    console.warn("Audelle: DSP worker unavailable, running on the main thread", error);
    noteMode("main");
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
      console.warn("Audelle: DSP worker failed, falling back to the main thread");
      noteMode("main");
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
      } else if (message.type === "error") {
        reject(new Error(message.message));
      }
    };

    worker.onerror = fallback;

    const request: WorkerRequest = { id, channels, sampleRate: source.sampleRate, plan };
    worker.postMessage(request, channels);
  });
}

/**
 * Measure tempo/brightness/rms off the UI thread.
 *
 * Loading a three-minute file used to run the analysis synchronously right
 * after decode. That is the moment the sidebar appeared frozen, so this uses
 * the same worker protocol as the renders.
 */
export async function analyzeLocalAsync(source: LocalAudio): Promise<LocalAnalysis> {
  if (!canUseWorker()) {
    noteMode("main");
    return analyzeLocal(source);
  }

  let worker: Worker;
  try {
    worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    noteMode("worker");
  } catch {
    noteMode("main");
    return analyzeLocal(source);
  }

  const id = ++nextJobId;
  const channels = source.channels.map((c) => c.slice().buffer as ArrayBuffer);

  return await new Promise<LocalAnalysis>((resolve, reject) => {
    let settled = false;
    const stop = () => worker.terminate();
    const fallback = () => {
      if (settled) return;
      settled = true;
      stop();
      noteMode("main");
      resolve(analyzeLocal(source));
    };

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.id !== id || settled) return;
      if (message.type === "analysis") {
        settled = true;
        stop();
        resolve(message.analysis);
      } else if (message.type === "error") {
        settled = true;
        stop();
        reject(new Error(message.message));
      }
    };
    worker.onerror = fallback;

    const request: WorkerRequest = {
      id,
      kind: "analyze",
      channels,
      sampleRate: source.sampleRate,
    };
    worker.postMessage(request, channels);
  });
}

function mixLocal(base: LocalAudio, overlay: LocalAudio, levelDb: number): LocalResult {
  const mixed = mixAudio(base, overlay, levelDb);
  return {
    audio: mixed.audio,
    plan: { intent: "lab", params: { levelDb }, summary: "Local stem mix" },
    metadata: { mixed_seconds: mixed.mixedSeconds, level_db: mixed.appliedDb },
    blob: encodeWav(mixed.audio),
  };
}

/** Sum an imported vocal/stem off the UI thread. */
export async function mixStemAsync(
  base: LocalAudio,
  overlay: LocalAudio,
  levelDb: number,
): Promise<LocalResult> {
  if (!canUseWorker()) {
    noteMode("main");
    return mixLocal(base, overlay, levelDb);
  }

  let worker: Worker;
  try {
    worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    noteMode("worker");
  } catch {
    noteMode("main");
    return mixLocal(base, overlay, levelDb);
  }

  const id = ++nextJobId;
  const channels = base.channels.map((c) => c.slice().buffer as ArrayBuffer);
  const overlayChannels = overlay.channels.map((c) => c.slice().buffer as ArrayBuffer);

  return await new Promise<LocalResult>((resolve, reject) => {
    let settled = false;
    const stop = () => worker.terminate();
    const fallback = () => {
      if (settled) return;
      settled = true;
      stop();
      noteMode("main");
      try {
        resolve(mixLocal(base, overlay, levelDb));
      } catch (error) {
        reject(error);
      }
    };

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;
      if (message.id !== id || settled) return;
      if (message.type === "done") {
        settled = true;
        stop();
        resolve({
          audio: {
            sampleRate: message.sampleRate,
            channels: message.channels.map((buffer) => new Float32Array(buffer)),
          },
          plan: { intent: "lab", params: { levelDb }, summary: message.summary },
          metadata: message.metadata,
          blob: new Blob([message.wav], { type: "audio/wav" }),
        });
      } else if (message.type === "error") {
        settled = true;
        stop();
        reject(new Error(message.message));
      }
    };
    worker.onerror = fallback;

    const request: WorkerRequest = {
      id,
      kind: "mix",
      channels,
      sampleRate: base.sampleRate,
      overlayChannels,
      overlaySampleRate: overlay.sampleRate,
      levelDb,
    };
    worker.postMessage(request, [...channels, ...overlayChannels]);
  });
}
