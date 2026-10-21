/**
 * 16-bit PCM WAV writer.
 *
 * The browser can decode almost anything but can only encode WAV, so this is
 * the output format for every local edit. 16-bit is what the server's WAV path
 * writes too, and it keeps a 3-minute stereo file around 30 MB instead of 60.
 */

import type { LocalAudio } from "./types";

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
}

export function encodeWav(audio: LocalAudio): Blob {
  const channels = audio.channels.length;
  const frames = audio.channels[0]?.length ?? 0;
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const dataBytes = frames * blockAlign;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, audio.sampleRate, true);
  view.setUint32(28, audio.sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < channels; c++) {
      const sample = Math.max(-1, Math.min(1, audio.channels[c][i] ?? 0));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: "audio/wav" });
}

export function peakOf(audio: LocalAudio): number {
  let peak = 0;
  for (const channel of audio.channels) {
    for (let i = 0; i < channel.length; i++) {
      const v = Math.abs(channel[i]);
      if (v > peak) peak = v;
    }
  }
  return peak;
}

export function durationOf(audio: LocalAudio): number {
  return audio.sampleRate > 0 ? (audio.channels[0]?.length ?? 0) / audio.sampleRate : 0;
}
