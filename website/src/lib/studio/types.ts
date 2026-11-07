import type { LocalAudio } from "@/lib/audio";

/** One decoded audio file in the studio's asset pool. */
export interface StudioAsset {
  id: string;
  name: string;
  audio: LocalAudio;
  duration: number;
  sampleRate: number;
  channels: number;
}

/** A region of an asset placed on a track. All times are seconds. */
export interface StudioClip {
  id: string;
  assetId: string;
  name: string;
  /** Timeline position of the clip's first sample. */
  start: number;
  /** Offset into the source asset. */
  offset: number;
  /** Playable length. */
  duration: number;
  gainDb: number;
  fadeIn: number;
  fadeOut: number;
}

export interface StudioEq {
  low: number;
  mid: number;
  high: number;
}

export interface StudioCompressor {
  enabled: boolean;
  thresholdDb: number;
  ratio: number;
  attackMs: number;
  releaseMs: number;
}

export interface StudioAutomationPoint {
  id: string;
  time: number;
  gainDb: number;
}

export interface StudioTrack {
  id: string;
  name: string;
  gainDb: number;
  pan: number;
  mute: boolean;
  solo: boolean;
  eq: StudioEq;
  compressor: StudioCompressor;
  /** Volume automation in dB, linear between points. Added in v1.1. */
  automation?: StudioAutomationPoint[];
  /** Optional ducking from another track. Added in v1.1. */
  sidechain?: {
    enabled: boolean;
    sourceTrackId: string;
    amountDb: number;
  };
  clips: StudioClip[];
}

export interface StudioLoop {
  enabled: boolean;
  start: number;
  end: number;
}

export interface StudioMaster {
  gainDb: number;
  limiter: boolean;
  ceilingDb: number;
}

export interface StudioProject {
  version: 1;
  name: string;
  sampleRate: number;
  tracks: StudioTrack[];
  master: StudioMaster;
  loop: StudioLoop;
}

export interface LoudnessReport {
  integratedLufs: number;
  shortTermLufs: number;
  truePeakDbtp: number;
  samplePeakDbfs: number;
}

export const TRACK_COLORS = [
  "#d7d2c7",
  "#b9c7c2",
  "#c9b9b0",
  "#b7bcc9",
  "#c9c1b2",
  "#b9c9bd",
  "#c9b7c1",
  "#c2c2c2",
];

export function createId(prefix: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}_${crypto.randomUUID().slice(0, 8)}`;
  }
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

export function emptyProject(name = "Untitled session"): StudioProject {
  return {
    version: 1,
    name,
    sampleRate: 44100,
    tracks: [],
    master: { gainDb: 0, limiter: true, ceilingDb: -1 },
    loop: { enabled: false, start: 0, end: 0 },
  };
}

export function createTrack(name: string): StudioTrack {
  return {
    id: createId("trk"),
    name,
    gainDb: 0,
    pan: 0,
    mute: false,
    solo: false,
    eq: { low: 0, mid: 0, high: 0 },
    compressor: { enabled: false, thresholdDb: -18, ratio: 3, attackMs: 10, releaseMs: 120 },
    automation: [],
    clips: [],
  };
}

export function clipEnd(clip: StudioClip): number {
  return clip.start + clip.duration;
}

export function projectDuration(project: StudioProject): number {
  let end = 0;
  for (const track of project.tracks) {
    for (const clip of track.clips) end = Math.max(end, clipEnd(clip));
  }
  return end;
}

export function discoverAssets(project: StudioProject): Set<string> {
  const ids = new Set<string>();
  for (const track of project.tracks) {
    for (const clip of track.clips) ids.add(clip.assetId);
  }
  return ids;
}
