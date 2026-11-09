import { describe, expect, test } from "bun:test";
import {
  alignTo,
  applyInstruction,
  matchLevels,
  MAX_COMBINE_TRACKS,
  mixTracks,
  parseCombineInstruction,
  rmsDb,
  suggestRole,
  type CombineTrack,
} from "./combine";
import type { LocalAudio } from "./types";

function burstTrack(times: number[], delay = 0, seconds = 4, sampleRate = 8000): LocalAudio {
  const data = new Float32Array(Math.round(seconds * sampleRate));
  for (const time of times) {
    const start = Math.round((time + delay) * sampleRate);
    for (let i = 0; i < 90; i++) {
      const index = start + i;
      if (index >= data.length) break;
      data[index] += Math.exp(-i / 22) * 0.9;
    }
  }
  return { channels: [data], sampleRate };
}

const BURSTS = [0.18, 0.52, 1.07, 1.41, 2.26, 2.73, 3.19];

function track(id: string, role: CombineTrack["role"], gainDb = 0): CombineTrack {
  return {
    id,
    name: id,
    role,
    audio: burstTrack(BURSTS),
    offset: 0,
    gainDb,
    muted: false,
  };
}

describe("combine engine", () => {
  test("suggests roles from filenames", () => {
    expect(suggestRole("voice-take-3.wav")).toBe("vocals");
    expect(suggestRole("piano.m4a")).toBe("piano");
    expect(suggestRole("karaoke backing.wav")).toBe("backing");
    expect(suggestRole("take-final.wav")).toBe("other");
    expect(MAX_COMBINE_TRACKS).toBe(5);
  });

  test("aligns a delayed take back to the reference", () => {
    const reference = burstTrack(BURSTS);
    const delayed = burstTrack(BURSTS, 1.25);
    const aligned = alignTo(reference, delayed);
    expect(aligned.confidence).toBeGreaterThan(0.2);
    expect(aligned.offset).toBeCloseTo(-1.25, 1);
  });

  test("does not invent an offset for a silent track", () => {
    const reference = burstTrack(BURSTS);
    const silent: LocalAudio = { channels: [new Float32Array(8000)], sampleRate: 8000 };
    const aligned = alignTo(reference, silent);
    expect(aligned.confidence).toBeLessThan(0.2);
  });

  test("mixes offsets on one timeline", () => {
    const sampleRate = 1000;
    const one: LocalAudio = { channels: [new Float32Array(1000).fill(0.2), new Float32Array(1000).fill(0.2)], sampleRate };
    const tracks: CombineTrack[] = [
      { id: "a", name: "a", role: "piano", audio: one, offset: 0, gainDb: 0, muted: false },
      { id: "b", name: "b", role: "vocals", audio: one, offset: 0.5, gainDb: 0, muted: false },
    ];
    const mixed = mixTracks(tracks);
    expect(mixed.duration).toBeCloseTo(1.5, 2);
    expect(mixed.audio.channels[0][100]).toBeCloseTo(0.2, 3);
    expect(mixed.audio.channels[0][750]).toBeCloseTo(0.4, 3);
    expect(mixed.scaledBy).toBe(1);
  });

  test("a muted track is left out of the mix", () => {
    const sampleRate = 1000;
    const one: LocalAudio = { channels: [new Float32Array(1000).fill(0.2)], sampleRate };
    const tracks: CombineTrack[] = [
      { id: "a", name: "a", role: "piano", audio: one, offset: 0, gainDb: 0, muted: false },
      { id: "b", name: "b", role: "vocals", audio: one, offset: 0, gainDb: 0, muted: true },
    ];
    const mixed = mixTracks(tracks);
    expect(mixed.audio.channels[0][500]).toBeCloseTo(0.2, 3);
  });

  test("parses 'highlight the vocals over the piano by 3 dB'", () => {
    const tracks = [track("v", "vocals"), track("p", "piano")];
    const result = parseCombineInstruction("highlight the vocals over the piano by 3 dB", tracks);
    expect(result.gainsDb.v).toBeCloseTo(1.5, 3);
    expect(result.gainsDb.p).toBeCloseTo(-1.5, 3);
    expect(result.summary).toContain("3 dB apart");
  });

  test("parses 'bring the singing up over the backing'", () => {
    const tracks = [track("v", "vocals"), track("b", "backing")];
    const result = parseCombineInstruction("bring the singing up over the backing", tracks);
    expect(result.gainsDb.v).toBeCloseTo(1.5, 3);
    expect(result.gainsDb.b).toBeCloseTo(-1.5, 3);
  });

  test("ducking lowers only the named part", () => {
    const tracks = [track("v", "vocals"), track("p", "piano")];
    const result = parseCombineInstruction("duck the piano under the vocals by 6 dB", tracks);
    expect(result.gainsDb.p).toBeCloseTo(-6, 3);
    expect(result.gainsDb.v).toBeUndefined();
  });

  test("a two-clause balance applies both directions", () => {
    const tracks = [track("v", "vocals"), track("p", "piano")];
    const result = parseCombineInstruction("make the vocals louder and the piano quieter", tracks);
    expect(result.gainsDb.v).toBeCloseTo(3, 3);
    expect(result.gainsDb.p).toBeCloseTo(-3, 3);
  });

  test("accepts an explicit signed dB value", () => {
    const tracks = [track("v", "vocals"), track("p", "piano")];
    const result = parseCombineInstruction("piano -4 dB", tracks);
    expect(result.gainsDb.p).toBeCloseTo(-4, 3);
  });

  test("applies a parsed instruction as a delta", () => {
    const tracks = [track("v", "vocals", 2), track("p", "piano", -1)];
    const result = parseCombineInstruction("highlight the vocals over the piano by 4 dB", tracks);
    const next = applyInstruction(tracks, result);
    expect(next[0].gainDb).toBeCloseTo(4, 3);
    expect(next[1].gainDb).toBeCloseTo(-3, 3);
  });

  test("matches file levels to a common RMS target", () => {
    const loud = { ...track("l", "piano"), audio: { channels: [new Float32Array(1000).fill(0.5)], sampleRate: 1000 } };
    const quiet = { ...track("q", "vocals"), audio: { channels: [new Float32Array(1000).fill(0.05)], sampleRate: 1000 } };
    const next = matchLevels([loud, quiet], -20);
    expect(rmsDb(next[0].audio) + next[0].gainDb).toBeCloseTo(-20, 3);
    expect(rmsDb(next[1].audio) + next[1].gainDb).toBeCloseTo(-20, 3);
  });

  test("says so when the instruction makes no sense", () => {
    const tracks = [track("v", "vocals"), track("p", "piano")];
    const result = parseCombineInstruction("make it sound like summer", tracks);
    expect(result.gainsDb).toEqual({});
    expect(result.summary).toContain("could not");
  });
});
