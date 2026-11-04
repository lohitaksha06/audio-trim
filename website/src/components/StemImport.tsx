"use client";

/**
 * Import a vocal/stem and mix it into the current track locally.
 *
 * This is a true audio sum (resampled, padded/truncated), but it does not
 * time-stretch or transpose. The server-side JamMixer does that when a backend
 * is configured.
 */

import { useEffect, useRef, useState } from "react";
import AudioPreview from "@/components/AudioPreview";
import { decodeFile, durationOf, mixStemAsync, type LocalAudio } from "@/lib/audio";

interface StemImportProps {
  base: LocalAudio;
  onApply: (payload: { audio: LocalAudio; blob: Blob; summary: string }) => void;
  onBusyChange?: (busy: boolean) => void;
  disabled?: boolean;
}

export default function StemImport({ base, onApply, onBusyChange, disabled }: StemImportProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [overlay, setOverlay] = useState<LocalAudio | null>(null);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (overlayUrl) URL.revokeObjectURL(overlayUrl);
    };
  }, [overlayUrl]);

  const pick = async (picked: File | null) => {
    setError("");
    if (!picked) {
      setFile(null);
      setOverlay(null);
      if (overlayUrl) URL.revokeObjectURL(overlayUrl);
      setOverlayUrl(null);
      return;
    }
    setFile(picked);
    setBusy(true);
    onBusyChange?.(true);
    try {
      const decoded = await decodeFile(picked);
      setOverlay(decoded);
      if (overlayUrl) URL.revokeObjectURL(overlayUrl);
      setOverlayUrl(URL.createObjectURL(picked));
    } catch (err) {
      setFile(null);
      setOverlay(null);
      setError(err instanceof Error ? err.message : "Could not decode that stem.");
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  };

  const apply = async () => {
    if (!overlay) return;
    setBusy(true);
    onBusyChange?.(true);
    setError("");
    try {
      const result = await mixStemAsync(base, overlay, level);
      const seconds = Number(result.metadata.mixed_seconds ?? 0);
      onApply({
        audio: result.audio,
        blob: result.blob,
        summary: `Mixed “${file?.name ?? "stem"}” at ${level > 0 ? "+" : ""}${level} dB (${seconds.toFixed(1)} s, local sum)`,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Mix failed");
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  };

  return (
    <div className="rounded-none border border-white/10 bg-white/[0.02] p-4">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-white/90">Import a vocal / stem</h3>
          <p className="text-xs text-white/55">
            Put your singer&apos;s take on top of the track. Local mix, no tempo or key matching.
          </p>
        </div>
        <span className="border border-white/15 px-2 py-1 font-mono text-[0.6rem] uppercase tracking-[0.12em] text-white/45">
          local sum
        </span>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.flac,.aac,.m4a,.ogg,audio/*"
        className="hidden"
        onChange={(e) => {
          const picked = e.target.files?.[0] ?? null;
          e.target.value = "";
          void pick(picked);
        }}
      />

      {file ? (
        <div className="rounded-lg border border-white/10 bg-black/30 p-3">
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <p className="truncate text-xs text-white/85">{file.name}</p>
              <p className="text-[0.7rem] text-white/45">
                {overlay ? `${durationOf(overlay).toFixed(1)} s` : busy ? "decoding…" : "ready"}
              </p>
            </div>
            <button
              type="button"
              onClick={() => void pick(null)}
              className="shrink-0 rounded-md border border-white/12 px-2 py-1 text-[0.7rem] text-white/60 hover:text-white"
            >
              Remove
            </button>
          </div>
          {overlayUrl && <AudioPreview src={overlayUrl} height={44} />}
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled || busy}
          className="w-full rounded-lg border border-dashed border-white/15 px-3 py-3 text-xs text-white/60 transition-colors hover:border-white/35 hover:text-white/85 disabled:opacity-40"
        >
          {busy ? "Reading stem…" : "Choose a vocal or stem file"}
        </button>
      )}

      <div className="mt-3 flex items-center gap-3">
        <div className="flex-1">
          <div className="flex justify-between text-[0.7rem] text-white/55">
            <span>Stem level</span>
            <span className="font-mono">
              {level > 0 ? "+" : ""}
              {level} dB
            </span>
          </div>
          <input
            type="range"
            min={-24}
            max={6}
            step={1}
            value={level}
            onChange={(e) => setLevel(Number(e.target.value))}
            className="mt-1 w-full accent-white"
            aria-label="Imported stem level"
          />
        </div>
        <button
          type="button"
          onClick={() => void apply()}
          disabled={!overlay || busy || disabled}
          className="mt-4 shrink-0 rounded-lg bg-[#f2f1ec] px-4 py-1.5 text-xs font-semibold text-black transition-colors hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "Mixing…" : "Mix stem in"}
        </button>
      </div>

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}
