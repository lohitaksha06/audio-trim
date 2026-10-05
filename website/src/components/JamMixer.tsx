"use client";

import { useRef, useState } from "react";
import { processAudio, uploadFile, type ProcessResponse } from "@/services/api";

/**
 * Jam a second file against the track.
 *
 * Both files are analysed, then the incoming file is brought to the track's
 * tempo and key and phase-aligned before mixing. This is the path that makes
 * free text work: attach the second file here, then the composer can say
 * "mix my stem in" and the server receives a real stem_path instead of erroring.
 */

interface JamMixerProps {
  /** The track everything is mixed into. */
  audioPath: string;
  onResult: (res: ProcessResponse, label: string) => void;
  disabled?: boolean;
}

interface JamMeta {
  song_bpm?: number;
  stem_bpm?: number;
  stem_bpm_matched?: number;
  stretch_factor?: number;
  transposed_semitones?: number;
  song_key_pc?: number;
  stem_key_pc?: number;
  beat_offset_sec?: number;
  key_confidence?: number;
  tempo_residual_bpm?: number;
  jam_notes?: string[];
}

const NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

export default function JamMixer({ audioPath, onResult, disabled }: JamMixerProps) {
  const [file, setFile] = useState<File | null>(null);
  const [stemPath, setStemPath] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [level, setLevel] = useState(0.6);
  const [matchKey, setMatchKey] = useState(true);
  const [meta, setMeta] = useState<JamMeta | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const onPick = async (f: File | null) => {
    setError("");
    setMeta(null);
    if (!f) {
      setFile(null);
      setStemPath(null);
      return;
    }
    setFile(f);
    setUploading(true);
    try {
      const up = await uploadFile(f);
      setStemPath(up.audio_path);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setFile(null);
      setStemPath(null);
    } finally {
      setUploading(false);
    }
  };

  const jam = async () => {
    if (!stemPath) return;
    setBusy(true);
    setError("");
    try {
      const prompt = matchKey
        ? "mix my stem in and match its key to the track"
        : "mix my stem in";
      const res = await processAudio(audioPath, prompt, {
        stemPath,
        stemLevel: level,
      });
      const m = (res.metadata ?? {}) as JamMeta;
      setMeta(m);
      const bits: string[] = [];
      if (m.stretch_factor && m.stretch_factor !== 1) {
        bits.push(`tempo ×${m.stretch_factor.toFixed(2)}`);
      }
      if (m.transposed_semitones) {
        bits.push(`${m.transposed_semitones > 0 ? "+" : ""}${m.transposed_semitones} st`);
      }
      if (typeof m.beat_offset_sec === "number" && Math.abs(m.beat_offset_sec) > 0.001) {
        bits.push(`aligned ${m.beat_offset_sec > 0 ? "+" : ""}${m.beat_offset_sec.toFixed(2)}s`);
      }
      onResult(res, bits.length ? `Jammed (${bits.join(", ")})` : "Jammed");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Jam failed");
    } finally {
      setBusy(false);
    }
  };

  const ready = !!stemPath && !uploading && !busy && !disabled;

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-4">
      <div className="mb-1 flex items-center justify-between gap-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-white/78">
          Jam a second file
        </h4>
        <span className="text-xs text-white/52">Tempo, key and phase matched</span>
      </div>
      <p className="mb-3 text-xs leading-relaxed text-white/62">
        Bring in a loop, a stem or another track. It is stretched to your tempo,
        transposed into your key, and lined up on the beat grid before mixing.
      </p>

      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={(e) => void onPick(e.target.files?.[0] ?? null)}
      />

      {file ? (
        <div className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-black/30 px-3 py-2">
          <div className="min-w-0">
            <div className="truncate text-xs text-white/84">{file.name}</div>
            <div className="text-xs text-white/52">
              {uploading ? "Uploading…" : stemPath ? "Ready to jam" : "Upload failed"}
            </div>
          </div>
          <button
            type="button"
            onClick={() => void onPick(null)}
            className="shrink-0 rounded-lg border border-white/10 px-2.5 py-1 text-xs text-white/70 hover:border-white/25 hover:text-white/90"
          >
            Remove
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
          className="w-full rounded-xl border border-dashed border-white/15 px-3 py-3 text-xs text-white/62 transition-colors hover:border-neon-purple/40 hover:text-white/84 disabled:opacity-40"
        >
          Choose a file to jam
        </button>
      )}

      <div className="mt-3 flex items-center gap-3">
        <label className="flex shrink-0 items-center gap-2 text-xs text-white/70">
          <input
            type="checkbox"
            checked={matchKey}
            onChange={(e) => setMatchKey(e.target.checked)}
            className="accent-neon-purple"
          />
          Match key
        </label>
        <div className="flex-1">
          <div className="flex justify-between text-xs text-white/58">
            <span>Stem level</span>
            <span className="font-mono">{Math.round(level * 100)}%</span>
          </div>
          <input
            type="range"
            min={5}
            max={100}
            step={5}
            value={Math.round(level * 100)}
            onChange={(e) => setLevel(Number(e.target.value) / 100)}
            className="mt-1 w-full accent-neon-purple"
          />
        </div>
      </div>

      <button
        type="button"
        onClick={jam}
        disabled={!ready}
        className="mt-3 w-full rounded-lg bg-neon-blue/15 px-3 py-2 text-xs font-medium text-neon-blue transition-colors hover:bg-neon-blue/25 disabled:opacity-40"
      >
        {busy ? "Jamming…" : uploading ? "Uploading…" : "Jam it in"}
      </button>

      {meta && (
        <dl className="mt-3 grid grid-cols-2 gap-1.5 text-xs">
          <div className="rounded-lg border border-white/8 bg-black/25 px-2.5 py-1.5">
            <dt className="text-white/52">Tempo</dt>
            <dd className="font-mono text-white/86">
              {Math.round(meta.song_bpm ?? 0)} ← {Math.round(meta.stem_bpm ?? 0)} BPM
              {meta.stem_bpm_matched &&
              Math.round(meta.stem_bpm_matched) !== Math.round(meta.stem_bpm ?? 0)
                ? ` → ${Math.round(meta.stem_bpm_matched)}`
                : ""}
            </dd>
          </div>
          <div className="rounded-lg border border-white/8 bg-black/25 px-2.5 py-1.5">
            <dt className="text-white/52">Key</dt>
            <dd className="font-mono text-white/86">
              {meta.song_key_pc !== undefined ? NOTES[meta.song_key_pc] : "?"} ←{" "}
              {meta.stem_key_pc !== undefined ? NOTES[meta.stem_key_pc] : "?"}
              {meta.transposed_semitones
                ? ` (${meta.transposed_semitones > 0 ? "+" : ""}${meta.transposed_semitones} st)`
                : ""}
            </dd>
          </div>
        </dl>
      )}

      {meta?.jam_notes && meta.jam_notes.length > 0 && (
        <ul className="mt-2 space-y-0.5 text-xs text-white/58">
          {meta.jam_notes.map((n) => (
            <li key={n}>· {n}</li>
          ))}
        </ul>
      )}

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}