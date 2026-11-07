import { describe, expect, test } from "bun:test";
import type { LocalAudio } from "../audio";
import { buildEdl, buildFcpxml } from "./exchange";
import { measureLoudness } from "./loudness";
import { buildMidiFile, parseMidi } from "./midi";
import { renderProject, renderStems } from "./render";
import { createTrack, emptyProject, type StudioAsset, type StudioProject } from "./types";

const SR = 44100;

function sine(freq: number, seconds: number, amp: number): Float32Array {
  const out = new Float32Array(Math.round(SR * seconds));
  for (let i = 0; i < out.length; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / SR);
  return out;
}

function asset(id: string, name: string, audio: LocalAudio): StudioAsset {
  return {
    id,
    name,
    audio,
    duration: audio.channels[0].length / audio.sampleRate,
    sampleRate: audio.sampleRate,
    channels: audio.channels.length,
  };
}

function projectWithClips(): { project: StudioProject; assets: Record<string, StudioAsset> } {
  const a = asset("a1", "tone-a.wav", { sampleRate: SR, channels: [sine(440, 4, 0.25)] });
  const b = asset("a2", "tone-b.wav", { sampleRate: SR, channels: [sine(660, 4, 0.2)] });
  const project = emptyProject("Test session");
  const one = createTrack("Tone A");
  one.clips = [{ id: "c1", assetId: "a1", name: "Tone A", start: 0, offset: 0, duration: 4, gainDb: 0, fadeIn: 0.1, fadeOut: 0.1 }];
  const two = createTrack("Tone B");
  two.clips = [{ id: "c2", assetId: "a2", name: "Tone B", start: 1, offset: 0, duration: 2, gainDb: -3, fadeIn: 0.05, fadeOut: 0.05 }];
  project.tracks = [one, two];
  project.loop = { enabled: false, start: 0, end: 4 };
  return { project, assets: { a1: a, a2: b } };
}

describe("studio render", () => {
  test("mixes clips, respects the master ceiling and reports loudness", () => {
    const { project, assets } = projectWithClips();
    const result = renderProject(project, assets);
    expect(result.duration).toBeCloseTo(4, 1);
    expect(result.audio.channels.length).toBe(2);
    expect(result.audio.channels[0].length).toBe(Math.round(4 * SR));
    let peak = 0;
    for (const channel of result.audio.channels) {
      for (let i = 0; i < channel.length; i++) peak = Math.max(peak, Math.abs(channel[i]));
    }
    expect(peak).toBeLessThanOrEqual(10 ** (-1 / 20) + 0.001);
    expect(Number.isFinite(result.report.integratedLufs)).toBe(true);
    expect(result.report.truePeakDbtp).toBeLessThanOrEqual(-0.9);
  });

  test("the lookahead limiter holds the master ceiling", () => {
    const { project, assets } = projectWithClips();
    project.master.gainDb = 12;
    project.master.limiter = true;
    project.master.ceilingDb = -1;
    const result = renderProject(project, assets);
    let samplePeak = 0;
    for (const channel of result.audio.channels) {
      for (let i = 0; i < channel.length; i++) samplePeak = Math.max(samplePeak, Math.abs(channel[i]));
    }
    expect(samplePeak).toBeLessThanOrEqual(10 ** (-1 / 20) + 0.001);
    expect(result.report.truePeakDbtp).toBeLessThanOrEqual(-0.9);
  });

  test("renders a loop range and one stem per track", () => {
    const { project, assets } = projectWithClips();
    const loop = renderProject(project, assets, { start: 1, end: 3 });
    expect(loop.duration).toBeCloseTo(2, 1);
    const stems = renderStems(project, assets);
    expect(stems).toHaveLength(2);
    expect(stems[0].audio.channels[0].length).toBe(Math.round(4 * SR));
    expect(stems[0].name).toBe("Tone A");
  });

  test("volume automation can fade a track in", () => {
    const { project, assets } = projectWithClips();
    const single = { ...project, tracks: [project.tracks[0]] };
    single.tracks[0].automation = [
      { id: "p1", time: 0, gainDb: -60 },
      { id: "p2", time: 4, gainDb: 0 },
    ];
    const result = renderProject(single, assets);
    const channel = result.audio.channels[0];
    const rms = (from: number, to: number) => {
      let sum = 0;
      let count = 0;
      for (let i = from; i < to; i++) {
        sum += channel[i] * channel[i];
        count++;
      }
      return Math.sqrt(sum / Math.max(1, count));
    };
    const early = rms(0, Math.round(0.4 * SR));
    const late = rms(Math.round(3.5 * SR), Math.round(3.9 * SR));
    expect(late).toBeGreaterThan(early * 10);
  });

  test("sidechain ducking lowers the target contribution", () => {
    const { project, assets } = projectWithClips();
    const rms = (audio: ReturnType<typeof renderProject>["audio"]) => {
      let sum = 0;
      let count = 0;
      for (const channel of audio.channels) {
        for (let i = 0; i < channel.length; i++) {
          sum += channel[i] * channel[i];
          count++;
        }
      }
      return Math.sqrt(sum / Math.max(1, count));
    };
    const plain = rms(renderProject(project, assets).audio);
    project.tracks[1].sidechain = {
      enabled: true,
      sourceTrackId: project.tracks[0].id,
      amountDb: 12,
    };
    const ducked = rms(renderProject(project, assets).audio);
    expect(ducked).toBeLessThan(plain);
  });

  test("mute removes a track from the mix", () => {
    const { project, assets } = projectWithClips();
    const loud = renderProject(project, assets).report.integratedLufs;
    project.tracks[0].mute = true;
    const quiet = renderProject(project, assets).report.integratedLufs;
    expect(quiet).toBeLessThan(loud - 2);
  });
});

describe("studio loudness", () => {
  test("a -20 dBFS 1 kHz sine measures near -23 LUFS", () => {
    const report = measureLoudness({ sampleRate: SR, channels: [sine(1000, 5, 0.1)] });
    expect(report.integratedLufs).toBeGreaterThan(-24.5);
    expect(report.integratedLufs).toBeLessThan(-21.5);
    expect(report.truePeakDbtp).toBeGreaterThanOrEqual(-20.1);
  });
});

describe("studio MIDI", () => {
  test("renders a MIDI note track and round-trips its notes", () => {
    const project = emptyProject("MIDI test");
    project.tempo = 120;
    const track = createTrack("Synth");
    track.clips = [
      {
        id: "m1",
        kind: "midi",
        name: "Synth",
        start: 0,
        offset: 0,
        duration: 2,
        gainDb: 0,
        fadeIn: 0,
        fadeOut: 0,
        notes: [
          { midi: 60, start: 0.1, duration: 0.5, velocity: 100 },
          { midi: 64, start: 0.8, duration: 0.5, velocity: 100 },
        ],
      },
    ];
    project.tracks = [track];
    const rendered = renderProject(project, {});
    let peak = 0;
    for (const channel of rendered.audio.channels) {
      for (let i = 0; i < channel.length; i++) peak = Math.max(peak, Math.abs(channel[i]));
    }
    expect(peak).toBeGreaterThan(0.001);
    const bytes = buildMidiFile(project);
    const parsed = parseMidi(bytes);
    expect(parsed.tempo).toBe(120);
    expect(parsed.tracks).toHaveLength(1);
    expect(parsed.tracks[0].notes.map((note) => note.midi)).toEqual([60, 64]);
    expect(parsed.tracks[0].notes[0].start).toBeCloseTo(0.1, 1);
  });
});

describe("studio interchange", () => {
  test("EDL and FCPXML contain every clip", () => {
    const { project, assets } = projectWithClips();
    const edl = buildEdl(project, assets);
    expect(edl).toContain("TITLE: Test session");
    expect(edl).toContain("Tone A");
    expect(edl).toContain("Tone B");
    const fcpxml = buildFcpxml(project, assets);
    expect(fcpxml).toContain('<fcpxml version="1.8">');
    expect(fcpxml).toContain("asset-clip");
    expect(fcpxml).toContain("Tone A");
  });
});
