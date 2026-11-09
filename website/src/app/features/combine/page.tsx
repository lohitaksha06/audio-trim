"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type DragEvent, type PointerEvent as ReactPointerEvent } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import { decodeFile, encodeWav } from "@/lib/audio";
import {
  MAX_COMBINE_TRACKS,
  ROLE_OPTIONS,
  applyInstruction,
  autoAlign,
  matchLevels,
  mixTracks,
  parseCombineInstruction,
  rmsDb,
  suggestRole,
  type CombineRole,
  type CombineTrack,
  type MixResult,
} from "@/lib/audio/combine";

const ACCEPT = ".mp3,.wav,.flac,.aac,.m4a,.ogg,.opus,audio/*";

function createId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

function safeName(name: string): string {
  return name.replace(/[^a-z0-9-_]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase() || "audelle";
}

function downloadBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00.00";
  const value = Math.max(0, seconds);
  const minutes = Math.floor(value / 60);
  const rest = value % 60;
  return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
}

export default function CombinePage() {
  const [tracks, setTracks] = useState<CombineTrack[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [instruction, setInstruction] = useState("");
  const [instructionNote, setInstructionNote] = useState("");
  const [mixed, setMixed] = useState<MixResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [playhead, setPlayhead] = useState(0);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timelineDragRef = useRef<{ id: string; startX: number; origin: number; width: number; duration: number } | null>(null);

  const previewUrlRef = useRef<string | null>(null);
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  const total = useMemo(() => {
    if (!tracks.length) return { start: 0, end: 0, duration: 0 };
    const start = Math.min(0, ...tracks.map((track) => track.offset));
    const end = Math.max(
      ...tracks.map((track) => track.offset + (track.audio.channels[0]?.length ?? 0) / track.audio.sampleRate),
    );
    return { start, end, duration: Math.max(0, end - start) };
  }, [tracks]);

  const addFiles = useCallback(
    async (picked: FileList | File[]) => {
      const list = Array.from(picked).filter((file) => file.type.startsWith("audio/") || /\.(mp3|wav|flac|aac|m4a|ogg|opus)$/i.test(file.name));
      if (!list.length) {
        setError("Those files are not audio. Use MP3, WAV, FLAC, M4A, OGG or OPUS.");
        return;
      }
      const room = MAX_COMBINE_TRACKS - tracks.length;
      if (room <= 0) {
        setError(`The limit is ${MAX_COMBINE_TRACKS} files. Remove one first.`);
        return;
      }
      const accepted = list.slice(0, room);
      setBusy(`Decoding ${accepted.length} file${accepted.length === 1 ? "" : "s"}…`);
      setError("");
      try {
        const decoded: CombineTrack[] = [];
        for (const file of accepted) {
          try {
            const audio = await decodeFile(file);
            decoded.push({
              id: createId("track"),
              name: file.name,
              role: suggestRole(file.name),
              audio,
              offset: 0,
              gainDb: 0,
              muted: false,
            });
          } catch {
            setError(`${file.name} could not be decoded. Use MP3, WAV, FLAC, M4A, OGG or OPUS.`);
          }
        }
        if (decoded.length) {
          setTracks((current) => [...current, ...decoded]);
          setStatus(`${decoded.length} file${decoded.length === 1 ? "" : "s"} added. Align them, set an offset, or adjust the faders.`);
          setMixed(null);
        }
        if (list.length > room) {
          setError(`Only the first ${room} file${room === 1 ? "" : "s"} were added. The limit is ${MAX_COMBINE_TRACKS}.`);
        }
      } finally {
        setBusy(null);
      }
    },
    [tracks.length],
  );

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.files.length) void addFiles(event.dataTransfer.files);
  };

  const updateTrack = (id: string, patch: Partial<CombineTrack>) => {
    setTracks((current) => current.map((track) => (track.id === id ? { ...track, ...patch } : track)));
    setMixed(null);
  };

  const removeTrack = (id: string) => {
    setTracks((current) => current.filter((track) => track.id !== id));
    setMixed(null);
    setStatus("File removed.");
  };

  const handleAlign = () => {
    if (tracks.length < 2) {
      setError("Add at least two files before aligning.");
      return;
    }
    setError("");
    const outcome = autoAlign(tracks);
    setTracks(outcome.tracks);
    setMixed(null);
    setStatus(outcome.notes.join(" "));
  };

  const handleInstruction = () => {
    if (!tracks.length) {
      setError("Add the files first, then describe the balance.");
      return;
    }
    if (!instruction.trim()) return;
    const result = parseCombineInstruction(instruction, tracks);
    if (!Object.keys(result.gainsDb).length) {
      setInstructionNote(result.summary);
      return;
    }
    setTracks((current) => applyInstruction(current, result));
    setMixed(null);
    setInstructionNote(result.summary);
    setStatus("Balance updated from your instruction.");
  };

  const resetGains = () => {
    setTracks((current) => current.map((track) => ({ ...track, gainDb: 0 })));
    setMixed(null);
    setInstructionNote("");
    setStatus("All faders reset to 0 dB.");
  };

  const handleMatchLevels = () => {
    setTracks((current) => matchLevels(current));
    setMixed(null);
    setStatus("Faders set so every file sits near the same RMS. Fine-tune by ear.");
  };

  const timelineTimeAt = (event: ReactPointerEvent<HTMLCanvasElement>): { index: number; width: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas || !tracks.length || total.duration <= 0) return null;
    const rect = canvas.getBoundingClientRect();
    const relativeY = Math.max(0, Math.min(0.999, (event.clientY - rect.top) / rect.height));
    const index = Math.floor(relativeY * tracks.length);
    return { index, width: rect.width };
  };

  const startTimelineDrag = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const hit = timelineTimeAt(event);
    if (!hit) return;
    const track = tracks[hit.index];
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    timelineDragRef.current = {
      id: track.id,
      startX: event.clientX,
      origin: track.offset,
      width: hit.width,
      duration: total.duration,
    };
  };

  const moveTimelineDrag = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = timelineDragRef.current;
    if (!drag) return;
    const delta = ((event.clientX - drag.startX) / drag.width) * drag.duration;
    const offset = Math.max(-60, Math.min(60, drag.origin + delta));
    updateTrack(drag.id, { offset: Number(offset.toFixed(3)) });
  };

  const endTimelineDrag = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    timelineDragRef.current = null;
  };

  const renderMix = useCallback(async (forExport: boolean): Promise<MixResult | null> => {
    if (!tracks.length) {
      setError("Add at least one file.");
      return null;
    }
    setError("");
    setBusy(forExport ? "Rendering WAV…" : "Mixing…");
    await new Promise((resolve) => window.setTimeout(resolve, 16));
    try {
      const result = mixTracks(tracks);
      setMixed(result);
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      const url = URL.createObjectURL(encodeWav(result.audio));
      previewUrlRef.current = url;
      setPreviewUrl(url);
      setPlayhead(0);
      if (!forExport) {
        setStatus(
          `Mixed ${tracks.length} files · ${formatTime(result.duration)} · peak ${(20 * Math.log10(Math.max(result.peak, 1e-9))).toFixed(1)} dBFS${result.scaledBy < 1 ? " · safety gain applied" : ""}.`,
        );
      }
      return result;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Mixing failed.");
      return null;
    } finally {
      setBusy(null);
    }
  }, [tracks]);

  const handleExport = async () => {
    const result = mixed ?? (await renderMix(true));
    if (!result) return;
    const target = tracks[0]?.name.replace(/\.[^.]+$/, "") ?? "audelle";
    downloadBlob(encodeWav(result.audio), `${safeName(`${target}-combined`)}.wav`);
    setStatus(`WAV downloaded · ${formatTime(result.duration)}.`);
  };

  // Static waveform overview. The playhead is a separate DOM element so this
  // only redraws when a track, offset or mute state changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const width = 1200;
    const laneHeight = 58;
    const height = Math.max(laneHeight, tracks.length * laneHeight);
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    if (!tracks.length || total.duration <= 0) return;

    const span = Math.max(0.001, total.duration);
    const xOf = (time: number) => ((time - total.start) / span) * width;

    tracks.forEach((track, index) => {
      const top = index * laneHeight + 8;
      const lane = laneHeight - 16;
      const left = xOf(track.offset);
      const frames = track.audio.channels[0]?.length ?? 0;
      const trackWidth = Math.max(1, (frames / track.audio.sampleRate / span) * width);
      context.globalAlpha = track.muted ? 0.22 : 1;

      context.strokeStyle = "rgba(242,241,236,0.08)";
      context.beginPath();
      context.moveTo(0, top + lane / 2);
      context.lineTo(width, top + lane / 2);
      context.stroke();

      const data = track.audio.channels[0];
      const bars = Math.max(1, Math.round(trackWidth));
      const block = Math.max(1, Math.floor(frames / bars));
      context.fillStyle = track.muted ? "rgba(242,241,236,0.35)" : "rgba(242,241,236,0.62)";
      for (let bar = 0; bar < bars; bar++) {
        let peak = 0;
        const start = bar * block;
        for (let i = start; i < Math.min(frames, start + block); i += 9) {
          peak = Math.max(peak, Math.abs(data[i] ?? 0));
        }
        const h = Math.max(1, Math.min(lane, peak * lane * 1.6));
        context.fillRect(left + bar, top + (lane - h) / 2, 1, h);
      }

      context.globalAlpha = 1;
      context.font = "10px monospace";
      context.fillStyle = "rgba(242,241,236,0.42)";
      context.fillText(
        `${index + 1}  ${track.name.slice(0, 34)}${track.name.length > 34 ? "…" : ""} · ${track.offset >= 0 ? "+" : ""}${track.offset.toFixed(2)}s · ${track.gainDb >= 0 ? "+" : ""}${track.gainDb.toFixed(1)} dB`,
        8,
        top + 10,
      );
    });
  }, [tracks, total]);

  const playheadPercent = total.duration > 0
    ? Math.max(0, Math.min(100, ((playhead - total.start) / total.duration) * 100))
    : 0;

  return (
    <>
      <Nav />
      <div className="flex min-h-screen pt-[var(--nav-h)]">
        <Sidebar onPromptSelect={() => {}} />
        <main className="min-w-0 flex-1 overflow-x-hidden">
          <header className="border-b border-white/10 px-4 py-4 sm:px-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-white/40">
                  Workspace · combine
                </p>
                <h1 className="mt-0.5 font-display text-3xl font-bold lowercase tracking-[-0.03em] text-[#f2f1ec]">
                  combine takes.
                </h1>
                <p className="mt-1 max-w-2xl text-[0.9375rem] text-white/55">
                  Add 2–5 recordings of the same performance. Align them automatically or set the timing
                  by hand, then balance the parts with faders or a sentence.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="rounded-none bg-[#f2f1ec] px-4 py-2 text-xs font-semibold text-black hover:bg-white"
                >
                  Add audio
                </button>
                <button
                  onClick={() => void renderMix(false)}
                  disabled={!tracks.length || Boolean(busy)}
                  className="rounded-none border border-white/20 px-3 py-2 text-xs text-white/70 hover:border-white/50 hover:text-white disabled:opacity-30"
                >
                  Mix preview
                </button>
                <button
                  onClick={() => void handleExport()}
                  disabled={!tracks.length || Boolean(busy)}
                  className="rounded-none border border-white/20 px-3 py-2 text-xs text-white/70 hover:border-white/50 hover:text-white disabled:opacity-30"
                >
                  Export WAV
                </button>
              </div>
            </div>
            {status && <p className="mt-2 text-xs text-white/50">{status}</p>}
            {error && <p className="mt-2 text-xs text-red-300">{error}</p>}
            {busy && <p className="mt-2 text-xs text-white/60">{busy}</p>}
          </header>

          <div className="grid gap-4 px-4 py-4 sm:px-6 xl:grid-cols-[minmax(0,1fr)_360px]">
            <section className="min-w-0 space-y-4">
              <div
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                className={`border border-dashed px-4 py-3 transition-colors ${
                  dragging ? "border-white/45 bg-white/[0.06]" : "border-white/15 bg-white/[0.02]"
                }`}
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="text-xs leading-relaxed text-white/50">
                    Drop 2–5 files here. Separate voice, piano, guitar, karaoke or band stems all work.
                    Everything is decoded locally.
                  </p>
                  <span className="font-mono text-[0.6rem] uppercase tracking-[0.14em] text-white/35">
                    {tracks.length}/{MAX_COMBINE_TRACKS} files
                  </span>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept={ACCEPT}
                  className="hidden"
                  onChange={(event) => {
                    if (event.target.files?.length) void addFiles(event.target.files);
                    event.target.value = "";
                  }}
                />
              </div>

              {!tracks.length ? (
                <div className="flex min-h-[300px] flex-col items-center justify-center border border-white/10 bg-white/[0.015] px-6 text-center">
                  <p className="font-display text-2xl lowercase text-[#f2f1ec]">two files, one take.</p>
                  <p className="mt-2 max-w-md text-sm leading-relaxed text-white/50">
                    A voice memo and a piano recording. A karaoke track and your vocal. Up to five parts,
                    lined up on one timeline and mixed to a WAV.
                  </p>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="mt-5 rounded-none border border-white/25 px-5 py-3 text-sm text-white/75 hover:border-white/60 hover:text-white"
                  >
                    Choose audio files
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  {tracks.map((track, index) => {
                    const duration = (track.audio.channels[0]?.length ?? 0) / track.audio.sampleRate;
                    const isReference = index === tracks.findIndex((item) => !item.muted);
                    return (
                      <div key={track.id} className="border border-white/10 bg-white/[0.02] p-3">
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="font-mono text-[0.65rem] text-white/35">{String(index + 1).padStart(2, "0")}</span>
                              <p className="max-w-[22rem] truncate text-sm text-white/85">{track.name}</p>
                              {isReference && (
                                <span className="border border-white/20 px-1.5 py-0.5 font-mono text-[0.55rem] uppercase tracking-[0.12em] text-white/50">
                                  reference
                                </span>
                              )}
                            </div>
                            <p className="mt-0.5 font-mono text-[0.65rem] text-white/40">
                              {formatTime(duration)} · {Number.isFinite(rmsDb(track.audio)) ? `${rmsDb(track.audio).toFixed(1)} dBFS RMS` : "silent"}
                            </p>
                          </div>
                          <div className="flex flex-wrap items-center gap-2">
                            <select
                              aria-label={`Role for ${track.name}`}
                              value={track.role}
                              onChange={(event) => updateTrack(track.id, { role: event.target.value as CombineRole })}
                              className="rounded-none border border-white/15 bg-black px-2 py-1 text-xs text-white/75"
                            >
                              {ROLE_OPTIONS.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                            <button
                              onClick={() => updateTrack(track.id, { muted: !track.muted })}
                              className={`rounded-none border px-2 py-1 text-[0.65rem] ${
                                track.muted ? "border-white/45 bg-white/15 text-white" : "border-white/15 text-white/45"
                              }`}
                            >
                              M
                            </button>
                            <button
                              onClick={() => removeTrack(track.id)}
                              className="rounded-none border border-white/12 px-2 py-1 text-[0.65rem] text-white/40 hover:border-red-300/40 hover:text-red-300"
                            >
                              Remove
                            </button>
                          </div>
                        </div>

                        <div className="mt-3 grid gap-3 sm:grid-cols-2">
                          <div>
                            <div className="flex items-center justify-between gap-2">
                              <label className="text-[0.65rem] uppercase tracking-[0.12em] text-white/40">
                                Timing
                              </label>
                              <span className="font-mono text-[0.7rem] text-white/65">
                                {track.offset >= 0 ? "+" : ""}{track.offset.toFixed(3)} s
                              </span>
                            </div>
                            <div className="mt-1 flex items-center gap-1">
                              {[-0.5, -0.1, -0.01].map((step) => (
                                <button
                                  key={step}
                                  onClick={() => updateTrack(track.id, { offset: Number((track.offset + step).toFixed(3)) })}
                                  className="rounded-none border border-white/12 px-1.5 py-1 font-mono text-[0.6rem] text-white/55 hover:border-white/35 hover:text-white"
                                >
                                  {step > 0 ? "+" : ""}{step}
                                </button>
                              ))}
                              <input
                                type="number"
                                step={0.001}
                                min={-60}
                                max={60}
                                value={Number(track.offset.toFixed(3))}
                                onChange={(event) => updateTrack(track.id, { offset: Math.max(-60, Math.min(60, Number(event.target.value) || 0)) })}
                                aria-label={`Offset in seconds for ${track.name}`}
                                className="w-20 rounded-none border border-white/15 bg-white/[0.03] px-1.5 py-1 text-right font-mono text-xs text-white/75"
                              />
                              {[0.01, 0.1, 0.5].map((step) => (
                                <button
                                  key={step}
                                  onClick={() => updateTrack(track.id, { offset: Number((track.offset + step).toFixed(3)) })}
                                  className="rounded-none border border-white/12 px-1.5 py-1 font-mono text-[0.6rem] text-white/55 hover:border-white/35 hover:text-white"
                                >
                                  +{step}
                                </button>
                              ))}
                            </div>
                          </div>

                          <div>
                            <div className="flex items-center justify-between gap-2">
                              <label className="text-[0.65rem] uppercase tracking-[0.12em] text-white/40">
                                Fader
                              </label>
                              <span className="font-mono text-[0.7rem] text-white/65">
                                {track.gainDb >= 0 ? "+" : ""}{track.gainDb.toFixed(1)} dB
                              </span>
                            </div>
                            <input
                              type="range"
                              min={-24}
                              max={12}
                              step={0.5}
                              value={track.gainDb}
                              onChange={(event) => updateTrack(track.id, { gainDb: Number(event.target.value) })}
                              aria-label={`Gain in dB for ${track.name}`}
                              className="mt-2 w-full"
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {tracks.length > 0 && (
                <div className="border border-white/10 bg-white/[0.015]">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 px-3 py-2">
                    <p className="font-mono text-[0.6rem] uppercase tracking-[0.16em] text-white/40">
                      Timeline · {formatTime(total.duration)}
                    </p>
                    <p className="hidden text-[0.65rem] text-white/35 sm:block">
                      Click or drag a lane to move that file
                    </p>
                    <div className="flex gap-1.5">
                      <button
                        onClick={handleAlign}
                        className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/70 hover:border-white/50 hover:text-white"
                      >
                        Auto-align
                      </button>
                      <button
                        onClick={handleMatchLevels}
                        className="rounded-none border border-white/12 px-2 py-1 text-[0.65rem] text-white/50 hover:border-white/35 hover:text-white"
                      >
                        Match levels
                      </button>
                      <button
                        onClick={resetGains}
                        className="rounded-none border border-white/12 px-2 py-1 text-[0.65rem] text-white/50 hover:border-white/35 hover:text-white"
                      >
                        Reset faders
                      </button>
                    </div>
                  </div>
                  <div className="relative">
                    <canvas
                      ref={canvasRef}
                      className="block h-auto w-full cursor-grab active:cursor-grabbing"
                      style={{ touchAction: "none" }}
                      aria-label="Aligned waveforms. Drag a lane to change its timing."
                      onPointerDown={startTimelineDrag}
                      onPointerMove={moveTimelineDrag}
                      onPointerUp={endTimelineDrag}
                      onPointerCancel={endTimelineDrag}
                    />
                    <div
                      className="pointer-events-none absolute bottom-0 top-0 w-px bg-[#f2f1ec]/70"
                      style={{ left: `${playheadPercent}%` }}
                    />
                  </div>
                </div>
              )}

              {previewUrl && (
                <div className="border border-white/10 bg-white/[0.02] p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="font-mono text-[0.6rem] uppercase tracking-[0.16em] text-white/40">
                      Mixed preview
                    </p>
                    <button
                      onClick={() => void handleExport()}
                      className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white"
                    >
                      Download WAV
                    </button>
                  </div>
                  <audio
                    ref={audioRef}
                    src={previewUrl}
                    controls
                    onTimeUpdate={(event) => setPlayhead(event.currentTarget.currentTime + total.start)}
                    onEnded={() => setPlayhead(total.start)}
                    className="mt-2 w-full"
                  />
                </div>
              )}
            </section>

            <aside className="space-y-4">
              <div className="border border-white/10 bg-white/[0.02] p-3">
                <p className="text-[0.65rem] uppercase tracking-[0.16em] text-white/40">
                  Balance in words
                </p>
                <p className="mt-1 text-xs leading-relaxed text-white/50">
                  Name the parts you labelled above and how far apart they should sit. This changes the
                  faders, not the timing.
                </p>
                <textarea
                  value={instruction}
                  onChange={(event) => setInstruction(event.target.value)}
                  placeholder="highlight the vocals over the piano by 3 dB"
                  rows={3}
                  className="mt-2 w-full resize-none rounded-none border border-white/15 bg-black/40 px-2.5 py-2 text-sm text-white/80 outline-none focus:border-white/35"
                />
                <button
                  onClick={handleInstruction}
                  disabled={!tracks.length || !instruction.trim()}
                  className="mt-2 w-full rounded-none border border-white/25 px-3 py-2 text-xs text-white/75 hover:border-white/60 hover:text-white disabled:opacity-30"
                >
                  Apply balance
                </button>
                {instructionNote && <p className="mt-2 text-xs leading-relaxed text-white/55">{instructionNote}</p>}
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {[
                    "highlight the vocals over the piano by 3 dB",
                    "duck the backing under the vocals by 6 dB",
                    "make the vocals louder and the piano quieter",
                  ].map((example) => (
                    <button
                      key={example}
                      onClick={() => setInstruction(example)}
                      className="rounded-none border border-white/10 px-2 py-1 text-left text-[0.65rem] leading-snug text-white/50 hover:border-white/30 hover:text-white/80"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>

              <div className="border border-white/10 bg-white/[0.02] p-3">
                <p className="text-[0.65rem] uppercase tracking-[0.16em] text-white/40">
                  How alignment works
                </p>
                <p className="mt-1 text-xs leading-relaxed text-white/50">
                  Auto-align compares when notes and words start, not the raw waveform, because two
                  performances of the same take share timing but not samples. The first unmuted file is
                  the reference.
                </p>
                <p className="mt-2 text-xs leading-relaxed text-white/45">
                  A low match is reported instead of being trusted. For a take that starts mid-phrase,
                  set the offset by hand and check by ear.
                </p>
              </div>

              <div className="border border-white/10 bg-white/[0.02] p-3">
                <p className="text-[0.65rem] uppercase tracking-[0.16em] text-white/40">
                  What this does not do
                </p>
                <ul className="mt-1 space-y-1 text-xs leading-relaxed text-white/45">
                  <li>· No tempo stretch or key change; files play at their recorded speed.</li>
                  <li>· No source separation; these are already separate parts.</li>
                  <li>· Exports WAV only. Open the WAV in Studio for EQ, compression or stems.</li>
                  <li>· Auto-align is a timing aid, not a guarantee. Check the result.</li>
                </ul>
              </div>
            </aside>
          </div>
        </main>
      </div>
    </>
  );
}
