/**
 * Minimal Standard MIDI File reader/writer.
 *
 * The studio does not pretend to be a MIDI sequencer. Import brings note tracks
 * in as playable clips; export writes the tempo, markers, names and note events
 * so the session can be opened in a DAW. No SysEx, MTC, SMPTE division or MIDI
 * 2.0 is implemented.
 */

import type { StudioMidiNote, StudioProject, StudioTrack } from "./types";

export interface ParsedMidiTrack {
  name: string;
  notes: StudioMidiNote[];
  end: number;
}

export interface ParsedMidi {
  tracks: ParsedMidiTrack[];
  tempo: number;
}

function readString(bytes: Uint8Array, offset: number, length: number): string {
  return new TextDecoder().decode(bytes.subarray(offset, offset + length));
}

function readUint32(bytes: Uint8Array, offset: number): number {
  return ((bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3]) >>> 0;
}

function readUint16(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] << 8) | bytes[offset + 1];
}

function readVlq(bytes: Uint8Array, offset: number, end: number): { value: number; next: number } {
  let value = 0;
  let position = offset;
  for (let i = 0; i < 4; i++) {
    if (position >= end) return { value, next: position };
    const byte = bytes[position++];
    value = (value << 7) | (byte & 0x7f);
    if (!(byte & 0x80)) break;
  }
  return { value, next: position };
}

function tickToSeconds(tick: number, division: number, tempo: number): number {
  return (tick * tempo) / (division * 1_000_000);
}

export function parseMidi(bytes: Uint8Array): ParsedMidi {
  if (readString(bytes, 0, 4) !== "MThd") throw new Error("Not a Standard MIDI File");
  const headerLength = readUint32(bytes, 4);
  const division = readUint16(bytes, 12);
  if (division & 0x8000) throw new Error("SMPTE MIDI division is not supported");
  let position = 8 + headerLength;
  const tracks: ParsedMidiTrack[] = [];
  let projectTempo = 500000;
  let tempoSeen = false;

  while (position + 8 <= bytes.length) {
    const id = readString(bytes, position, 4);
    const length = readUint32(bytes, position + 4);
    const start = position + 8;
    const end = Math.min(bytes.length, start + length);
    if (id !== "MTrk") {
      position = start + length;
      continue;
    }

    const notes: StudioMidiNote[] = [];
    const active = new Map<string, { start: number; velocity: number }[]>();
    let name = "MIDI";
    let seconds = 0;
    let tempo = projectTempo;
    let running = 0;
    let cursor = start;

    while (cursor < end) {
      const delta = readVlq(bytes, cursor, end);
      cursor = delta.next;
      seconds += tickToSeconds(delta.value, division, tempo);
      if (cursor >= end) break;
      let status = bytes[cursor];
      if (status < 0x80) {
        if (!running) break;
        status = running;
      } else {
        cursor++;
        if (status < 0xf0) running = status;
      }

      if (status === 0xff) {
        const type = bytes[cursor++];
        const lengthResult = readVlq(bytes, cursor, end);
        cursor = lengthResult.next;
        const data = bytes.subarray(cursor, cursor + lengthResult.value);
        if (type === 0x51 && data.length >= 3) {
          tempo = (data[0] << 16) | (data[1] << 8) | data[2];
          if (!tempoSeen) {
            projectTempo = tempo;
            tempoSeen = true;
          }
        } else if (type === 0x03 && data.length) {
          name = new TextDecoder().decode(data);
        } else if (type === 0x2f) {
          break;
        }
        cursor += lengthResult.value;
        continue;
      }

      if (status === 0xf0 || status === 0xf7) {
        const lengthResult = readVlq(bytes, cursor, end);
        cursor = lengthResult.next + lengthResult.value;
        running = 0;
        continue;
      }

      const command = status & 0xf0;
      const channel = status & 0x0f;
      if (command === 0x90 || command === 0x80) {
        const note = bytes[cursor++] ?? 0;
        const velocity = bytes[cursor++] ?? 0;
        const key = `${channel}:${note}`;
        if (command === 0x90 && velocity > 0) {
          const stack = active.get(key) ?? [];
          stack.push({ start: seconds, velocity });
          active.set(key, stack);
        } else {
          const stack = active.get(key);
          const started = stack?.shift();
          if (started) {
            const duration = Math.max(0.01, seconds - started.start);
            notes.push({
              midi: note,
              start: Number(started.start.toFixed(4)),
              duration: Number(duration.toFixed(4)),
              velocity: started.velocity,
            });
          }
        }
      } else if (command === 0xc0 || command === 0xd0) {
        cursor += 1;
      } else if (command >= 0x80 && command <= 0xe0) {
        cursor += 2;
      } else {
        cursor += 1;
      }
    }

    notes.sort((a, b) => a.start - b.start);
    const endTime = notes.reduce((max, note) => Math.max(max, note.start + note.duration), 0);
    if (notes.length) tracks.push({ name, notes, end: endTime });
    position = start + length;
  }

  return { tracks, tempo: Math.round(60_000_000 / projectTempo) };
}

function vlq(value: number): number[] {
  const bytes = [value & 0x7f];
  let rest = value >> 7;
  while (rest > 0) {
    bytes.unshift((rest & 0x7f) | 0x80);
    rest >>= 7;
  }
  return bytes;
}

function uint32(value: number): number[] {
  return [(value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff];
}

function textEvent(type: number, text: string): number[] {
  const bytes = Array.from(new TextEncoder().encode(text));
  return [0xff, type, ...vlq(bytes.length), ...bytes];
}

interface TimedEvent {
  tick: number;
  order: number;
  bytes: number[];
}

function chunk(events: TimedEvent[]): number[] {
  events.sort((a, b) => a.tick - b.tick || a.order - b.order);
  const body: number[] = [];
  let last = 0;
  for (const event of events) {
    body.push(...vlq(Math.max(0, event.tick - last)), ...event.bytes);
    last = event.tick;
  }
  body.push(0x00, 0xff, 0x2f, 0x00);
  return [...Array.from(new TextEncoder().encode("MTrk")), ...uint32(body.length), ...body];
}

export function buildMidiFile(project: StudioProject): Uint8Array {
  const division = 480;
  const ticksPerSecond = (project.tempo * division) / 60;
  const secondsToTicks = (seconds: number) => Math.max(0, Math.round(seconds * ticksPerSecond));

  const tempoEvents: TimedEvent[] = [
    { tick: 0, order: 0, bytes: textEvent(0x03, project.name || "Audelle") },
    {
      tick: 0,
      order: 1,
      bytes: [
        0xff,
        0x51,
        0x03,
        (Math.round(60_000_000 / Math.max(20, project.tempo)) >> 16) & 0xff,
        (Math.round(60_000_000 / Math.max(20, project.tempo)) >> 8) & 0xff,
        Math.round(60_000_000 / Math.max(20, project.tempo)) & 0xff,
      ],
    },
    { tick: 0, order: 2, bytes: [0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08] },
  ];
  for (const track of project.tracks) {
    for (const clip of track.clips) {
      tempoEvents.push({
        tick: secondsToTicks(clip.start),
        order: 3,
        bytes: textEvent(0x06, `${track.name} / ${clip.name}`),
      });
    }
  }

  const chunks: number[][] = [chunk(tempoEvents)];
  let channel = 0;
  for (const track of project.tracks) {
    const midiClips = track.clips.filter((clip) => clip.kind === "midi" && clip.notes?.length);
    if (!midiClips.length) continue;
    const events: TimedEvent[] = [{ tick: 0, order: 0, bytes: textEvent(0x03, track.name) }];
    let order = 1;
    for (const clip of midiClips) {
      for (const note of clip.notes ?? []) {
        const start = secondsToTicks(clip.start + note.start);
        const end = secondsToTicks(clip.start + note.start + note.duration);
        const ch = channel % 16;
        events.push({ tick: start, order: order++, bytes: [0x90 | ch, note.midi & 0x7f, Math.max(1, Math.min(127, note.velocity))] });
        events.push({ tick: Math.max(start + 1, end), order: order++, bytes: [0x80 | ch, note.midi & 0x7f, 0] });
      }
      if (track.automation?.length) {
        // Volume automation is written as CC7 so a DAW can import the fader.
        for (const point of track.automation) {
          const value = Math.max(0, Math.min(127, Math.round(((point.gainDb + 60) / 72) * 127)));
          events.push({ tick: secondsToTicks(point.time), order: order++, bytes: [0xb0 | (channel % 16), 7, value] });
        }
      }
      channel++;
    }
    chunks.push(chunk(events));
  }

  const header = [
    ...Array.from(new TextEncoder().encode("MThd")),
    ...uint32(6),
    0x00,
    0x01,
    (chunks.length >> 8) & 0xff,
    chunks.length & 0xff,
    (division >> 8) & 0xff,
    division & 0xff,
  ];
  return new Uint8Array([...header, ...chunks.flat()]);
}

export function midiTrackLabel(track: StudioTrack): string {
  return track.name || "MIDI track";
}
