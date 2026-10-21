/**
 * Decode an uploaded file into channel buffers.
 *
 * `decodeAudioData` handles mp3, m4a/aac, ogg, flac and wav, so there is no
 * format sniffing here. It resamples to the context rate, which is why the
 * context is created at 44.1 kHz: that is the most common source rate and
 * avoids a resample for typical music.
 */

import type { LocalAudio } from "./types";

const PREFERRED_RATE = 44100;

interface DecodeContext {
  decodeAudioData: (data: ArrayBuffer) => Promise<AudioBuffer>;
  close?: () => Promise<void>;
}

function makeContext(): DecodeContext {
  const offline = (window as unknown as {
    OfflineAudioContext?: new (channels: number, length: number, rate: number) => BaseAudioContext;
  }).OfflineAudioContext;
  if (offline) {
    try {
      // Offline contexts need no user gesture and are not subject to autoplay
      // policy, which matters because decoding happens right after a file pick.
      return new offline(1, 1, PREFERRED_RATE) as unknown as DecodeContext;
    } catch {
      // Fall through to a realtime context.
    }
  }
  const Ctor = (window as unknown as { AudioContext: new (opts?: { sampleRate?: number }) => AudioContext })
    .AudioContext;
  return new Ctor({ sampleRate: PREFERRED_RATE }) as unknown as DecodeContext;
}

export async function decodeFile(file: File): Promise<LocalAudio> {
  const bytes = await file.arrayBuffer();
  const ctx = makeContext();
  try {
    const buffer = await ctx.decodeAudioData(bytes);
    const channels: Float32Array[] = [];
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      // `getChannelData` returns a view owned by the buffer; copy so edits do
      // not alias the decoded source.
      channels.push(new Float32Array(buffer.getChannelData(c)));
    }
    return { channels, sampleRate: buffer.sampleRate };
  } finally {
    void ctx.close?.().catch(() => undefined);
  }
}
