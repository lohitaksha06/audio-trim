"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent } from "react";
import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import { decodeFile, encodeWav } from "@/lib/audio";
import { buildEdl, buildFcpxml } from "@/lib/studio/exchange";
import { formatDbtp, formatLufs } from "@/lib/studio/loudness";
import { buildMidiFile, parseMidi } from "@/lib/studio/midi";
import { renderProject, renderStems } from "@/lib/studio/render";
import { loadStudioAsset, loadStudioProject, saveStudioAsset, saveStudioProject } from "@/lib/studio/storage";
import {
  createId,
  createTrack,
  discoverAssets,
  emptyProject,
  projectDuration,
  TRACK_COLORS,
  type LoudnessReport,
  type StudioAsset,
  type StudioClip,
  type StudioProject,
  type StudioTrack,
} from "@/lib/studio/types";

const MIN_ZOOM = 0.45;
const MAX_ZOOM = 3;
const TRACK_HEIGHT = 68;

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
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

function downloadText(text: string, name: string, type = "text/plain"): void {
  downloadBlob(new Blob([text], { type }), name);
}

function safeName(name: string): string {
  return name.replace(/[^a-z0-9-_]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase() || "audelle";
}

function formatTime(seconds: number): string {
  const value = Math.max(0, seconds);
  const minutes = Math.floor(value / 60);
  const rest = value % 60;
  return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
}

function ClipWave({ asset }: { asset: StudioAsset }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const width = 240;
    const height = 24;
    canvas.width = width;
    canvas.height = height;
    const channel = asset.audio.channels[0];
    const bars = 90;
    const block = Math.max(1, Math.floor(channel.length / bars));
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    for (let i = 0; i < bars; i++) {
      let peak = 0;
      const start = i * block;
      for (let j = 0; j < block; j += 12) peak = Math.max(peak, Math.abs(channel[start + j] ?? 0));
      const h = Math.max(1, peak * (height - 4));
      ctx.fillRect((i / bars) * width, (height - h) / 2, Math.max(1, width / bars - 0.6), h);
    }
  }, [asset]);
  return <canvas ref={ref} aria-hidden="true" className="pointer-events-none absolute inset-x-0 bottom-1 h-6 w-full opacity-45" />;
}

export default function StudioPage() {
  const [project, setProject] = useState<StudioProject>(() => emptyProject());
  const [assets, setAssets] = useState<Record<string, StudioAsset>>({});
  const [loaded, setLoaded] = useState(false);
  const [past, setPast] = useState<StudioProject[]>([]);
  const [future, setFuture] = useState<StudioProject[]>([]);
  const [selected, setSelected] = useState<{ trackId: string; clipId: string } | null>(null);
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  const [automationOpenId, setAutomationOpenId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewStart, setPreviewStart] = useState(0);
  const [previewDirty, setPreviewDirty] = useState(true);
  const [status, setStatus] = useState("");
  const [report, setReport] = useState<LoudnessReport | null>(null);
  const [renders, setRenders] = useState<{ id: string; label: string; url: string; name: string }[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const projectInputRef = useRef<HTMLInputElement>(null);
  const dragRef = useRef<{ project: StudioProject; trackId: string; clipId: string; startX: number; originalStart: number } | null>(null);
  const automationDragRef = useRef<{ project: StudioProject; trackId: string; pointId: string; rect: DOMRect } | null>(null);
  const movedRef = useRef(false);

  const contentDuration = projectDuration(project);
  const duration = Math.max(contentDuration, project.loop.end, 10);
  const pps = 64 * zoom;
  const timelineWidth = Math.max(640, (duration + 4) * pps);
  const selectedTrack = project.tracks.find((track) => track.id === selectedTrackId) ?? null;
  const selectedClip = selected
    ? project.tracks.find((track) => track.id === selected.trackId)?.clips.find((clip) => clip.id === selected.clipId) ?? null
    : null;
  const selectedTrackForClip = selected ? project.tracks.find((track) => track.id === selected.trackId) ?? null : null;

  const commit = useCallback(
    (next: StudioProject) => {
      setPast((items) => [...items.slice(-49), project]);
      setFuture([]);
      setProject(next);
      setPreviewDirty(true);
    },
    [project],
  );

  const undo = useCallback(() => {
    setPast((items) => {
      if (!items.length) return items;
      const previous = items[items.length - 1];
      setFuture((next) => [project, ...next].slice(0, 50));
      setProject(previous);
      setPreviewDirty(true);
      return items.slice(0, -1);
    });
  }, [project]);

  const redo = useCallback(() => {
    setFuture((items) => {
      if (!items.length) return items;
      const nextProject = items[0];
      setPast((previous) => [...previous.slice(-49), project]);
      setProject(nextProject);
      setPreviewDirty(true);
      return items.slice(1);
    });
  }, [project]);

  // Restore the last session and its assets.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const stored = await loadStudioProject();
      if (cancelled) return;
      if (stored?.tracks) {
        const loadedAssets: Record<string, StudioAsset> = {};
        for (const id of discoverAssets(stored)) {
          const file = await loadStudioAsset(id);
          if (!file) continue;
          try {
            const audio = await decodeFile(file);
            loadedAssets[id] = {
              id,
              name: file.name,
              audio,
              duration: audio.channels[0].length / audio.sampleRate,
              sampleRate: audio.sampleRate,
              channels: audio.channels.length,
            };
          } catch {
            /* Ignore a corrupt cached asset; the project still opens. */
          }
        }
        if (cancelled) return;
        setAssets(loadedAssets);
        setProject({ ...stored, tempo: stored.tempo || 120 });
        setSelectedTrackId(stored.tracks[0]?.id ?? null);
      }
      setLoaded(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Autosave the document. Assets are written when imported.
  useEffect(() => {
    if (!loaded) return;
    const timer = window.setTimeout(() => void saveStudioProject(project), 400);
    return () => window.clearTimeout(timer);
  }, [loaded, project]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redo();
      }
      if (event.code === "Space" && !(event.target instanceof HTMLInputElement) && !(event.target instanceof HTMLTextAreaElement)) {
        event.preventDefault();
        void togglePlay();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const addFiles = async (files: FileList | File[]) => {
    const list = Array.from(files);
    if (!list.length) return;
    setStatus(`Decoding ${list.length} file${list.length === 1 ? "" : "s"}…`);
    const nextAssets = { ...assets };
    const nextTracks = [...project.tracks];
    let nextTempo = project.tempo || 120;
    for (const file of list) {
      try {
        if (/\.(mid|midi)$/i.test(file.name)) {
          const parsed = parseMidi(new Uint8Array(await file.arrayBuffer()));
          nextTempo = parsed.tempo || nextTempo;
          for (const midiTrack of parsed.tracks) {
            const track = createTrack(midiTrack.name || file.name.replace(/\.[^.]+$/, ""));
            track.clips = [
              {
                id: createId("clip"),
                kind: "midi",
                name: midiTrack.name || file.name.replace(/\.[^.]+$/, ""),
                start: 0,
                offset: 0,
                duration: Math.max(0.1, midiTrack.end),
                gainDb: 0,
                fadeIn: 0,
                fadeOut: 0,
                notes: midiTrack.notes,
              },
            ];
            nextTracks.push(track);
          }
          continue;
        }
        const audio = await decodeFile(file);
        const id = createId("asset");
        const asset: StudioAsset = {
          id,
          name: file.name,
          audio,
          duration: audio.channels[0].length / audio.sampleRate,
          sampleRate: audio.sampleRate,
          channels: audio.channels.length,
        };
        nextAssets[id] = asset;
        await saveStudioAsset(id, file);
        const track = createTrack(file.name.replace(/\.[^.]+$/, "") || "Audio");
        track.clips = [
          {
            id: createId("clip"),
            assetId: id,
            name: file.name.replace(/\.[^.]+$/, "") || file.name,
            start: 0,
            offset: 0,
            duration: asset.duration,
            gainDb: 0,
            fadeIn: 0,
            fadeOut: 0,
          },
        ];
        nextTracks.push(track);
      } catch (error) {
        setStatus(`Could not decode ${file.name}: ${error instanceof Error ? error.message : "unknown error"}`);
      }
    }
    setAssets(nextAssets);
    const next = { ...project, tempo: nextTempo, tracks: nextTracks };
    setPast((items) => [...items.slice(-49), project]);
    setFuture([]);
    setProject(next);
    setSelectedTrackId(nextTracks[nextTracks.length - 1]?.id ?? null);
    setPreviewDirty(true);
    setStatus(`Added ${list.length} track${list.length === 1 ? "" : "s"}.`);
  };

  const updateTrack = (trackId: string, update: (track: StudioTrack) => StudioTrack) => {
    commit({ ...project, tracks: project.tracks.map((track) => (track.id === trackId ? update(track) : track)) });
  };

  const updateClip = (trackId: string, clipId: string, update: (clip: StudioClip) => StudioClip) => {
    commit({
      ...project,
      tracks: project.tracks.map((track) =>
        track.id === trackId
          ? { ...track, clips: track.clips.map((clip) => (clip.id === clipId ? update(clip) : clip)) }
          : track,
      ),
    });
  };

  const splitSelected = () => {
    if (!selected || !selectedClip) return;
    const split = playhead;
    if (split <= selectedClip.start + 0.01 || split >= selectedClip.start + selectedClip.duration - 0.01) {
      setStatus("Put the playhead inside the selected clip first.");
      return;
    }
    const left = split - selectedClip.start;
    const right = selectedClip.duration - left;
    const isMidi = selectedClip.kind === "midi";
    const notes = selectedClip.notes ?? [];
    const leftNotes = isMidi
      ? notes
          .filter((note) => note.start < left)
          .map((note) => ({ ...note, duration: Math.min(note.duration, left - note.start) }))
      : selectedClip.notes;
    const rightNotes = isMidi
      ? notes
          .filter((note) => note.start + note.duration > left)
          .map((note) => {
            const start = Math.max(0, note.start - left);
            const end = note.start + note.duration - Math.max(note.start, left);
            return { ...note, start, duration: Math.max(0.01, end) };
          })
      : selectedClip.notes;
    const leftClip: StudioClip = {
      ...selectedClip,
      id: createId("clip"),
      duration: left,
      notes: leftNotes,
      fadeOut: Math.max(selectedClip.fadeOut, 0.01),
    };
    const rightClip: StudioClip = {
      ...selectedClip,
      id: createId("clip"),
      start: split,
      offset: isMidi ? 0 : selectedClip.offset + left,
      duration: right,
      notes: rightNotes,
      fadeIn: Math.max(selectedClip.fadeIn, 0.01),
    };
    const track = project.tracks.find((item) => item.id === selected.trackId);
    if (!track) return;
    const next = {
      ...project,
      tracks: project.tracks.map((item) =>
        item.id === track.id
          ? { ...item, clips: item.clips.flatMap((clip) => (clip.id === selectedClip.id ? [leftClip, rightClip] : [clip])) }
          : item,
      ),
    };
    commit(next);
    setSelected({ trackId: selected.trackId, clipId: rightClip.id });
    setStatus("Split at the playhead. Fade handles keep the seam clean.");
  };

  const duplicateSelected = () => {
    if (!selected || !selectedClip) return;
    const copy: StudioClip = {
      ...selectedClip,
      id: createId("clip"),
      start: selectedClip.start + selectedClip.duration,
    };
    updateTrack(selected.trackId, (track) => ({ ...track, clips: [...track.clips, copy] }));
    setSelected({ trackId: selected.trackId, clipId: copy.id });
  };

  const deleteSelected = () => {
    if (!selected) return;
    const track = project.tracks.find((item) => item.id === selected.trackId);
    if (!track) return;
    const next = {
      ...project,
      tracks: project.tracks.map((item) =>
        item.id === track.id ? { ...item, clips: item.clips.filter((clip) => clip.id !== selected.clipId) } : item,
      ),
    };
    commit(next);
    setSelected(null);
  };

  const crossfadeSelected = () => {
    if (!selected || !selectedClip || !selectedTrackForClip) return;
    const sorted = [...selectedTrackForClip.clips].sort((a, b) => a.start - b.start);
    const index = sorted.findIndex((clip) => clip.id === selectedClip.id);
    const next = sorted[index + 1];
    if (!next || Math.abs(next.start - (selectedClip.start + selectedClip.duration)) > 0.5) {
      setStatus("Put the next clip within 0.5 s, then crossfade.");
      return;
    }
    const fade = 0.05;
    const updated = {
      ...project,
      tracks: project.tracks.map((track) =>
        track.id === selectedTrackForClip.id
          ? {
              ...track,
              clips: track.clips.map((clip) => {
                if (clip.id === selectedClip.id) return { ...clip, fadeOut: fade };
                if (clip.id === next.id) return { ...clip, start: clip.start - fade, offset: clip.offset + fade, fadeIn: fade };
                return clip;
              }),
            }
          : track,
      ),
    };
    commit(updated);
    setStatus("50 ms crossfade applied.");
  };

  const removeTrack = (trackId: string) => {
    commit({ ...project, tracks: project.tracks.filter((track) => track.id !== trackId) });
    if (selectedTrackId === trackId) setSelectedTrackId(null);
    if (selected?.trackId === trackId) setSelected(null);
  };

  const startClipDrag = (event: ReactPointerEvent, track: StudioTrack, clip: StudioClip) => {
    event.stopPropagation();
    setSelected({ trackId: track.id, clipId: clip.id });
    setSelectedTrackId(track.id);
    dragRef.current = {
      project,
      trackId: track.id,
      clipId: clip.id,
      startX: event.clientX,
      originalStart: clip.start,
    };
    movedRef.current = false;
    const move = (moveEvent: PointerEvent) => {
      const drag = dragRef.current;
      if (!drag) return;
      const delta = (moveEvent.clientX - drag.startX) / pps;
      if (Math.abs(delta) < 0.005) return;
      movedRef.current = true;
      const snapped = Math.max(0, Math.round((drag.originalStart + delta) * 20) / 20);
      setProject((current) => ({
        ...current,
        tracks: current.tracks.map((item) =>
          item.id === drag.trackId
            ? {
                ...item,
                clips: item.clips.map((itemClip) => (itemClip.id === drag.clipId ? { ...itemClip, start: snapped } : itemClip)),
              }
            : item,
        ),
      }));
      setPreviewDirty(true);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      const drag = dragRef.current;
      dragRef.current = null;
      if (drag && movedRef.current) {
        setPast((items) => [...items.slice(-49), drag.project]);
        setFuture([]);
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const automationY = (gainDb: number): number => 6 + (1 - (clamp(gainDb, -60, 12) + 60) / 72) * 40;
  const automationGain = (y: number): number => clamp(12 - ((y - 6) / 40) * 72, -60, 12);

  const addAutomationPoint = (event: ReactMouseEvent<HTMLDivElement>, track: StudioTrack) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const time = clamp((event.clientX - rect.left) / pps, 0, duration);
    const gainDb = automationGain(event.clientY - rect.top);
    updateTrack(track.id, (item) => ({
      ...item,
      automation: [
        ...(item.automation ?? []).filter((point) => Math.abs(point.time - time) > 0.04),
        { id: createId("auto"), time: Number(time.toFixed(3)), gainDb: Number(gainDb.toFixed(1)) },
      ].sort((a, b) => a.time - b.time),
    }));
  };

  const startAutomationDrag = (event: ReactPointerEvent<SVGCircleElement>, trackId: string, pointId: string) => {
    event.stopPropagation();
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg) return;
    automationDragRef.current = {
      project,
      trackId,
      pointId,
      rect: svg.getBoundingClientRect(),
    };
    let moved = false;
    const move = (moveEvent: PointerEvent) => {
      const drag = automationDragRef.current;
      if (!drag) return;
      const time = clamp((moveEvent.clientX - drag.rect.left) / pps, 0, duration);
      const gainDb = automationGain(moveEvent.clientY - drag.rect.top);
      if (Math.abs(time - (drag.project.tracks.find((track) => track.id === drag.trackId)?.automation?.find((point) => point.id === drag.pointId)?.time ?? time)) > 0.01) moved = true;
      setProject((current) => ({
        ...current,
        tracks: current.tracks.map((track) =>
          track.id === drag.trackId
            ? {
                ...track,
                automation: (track.automation ?? []).map((point) =>
                  point.id === drag.pointId ? { ...point, time: Number(time.toFixed(3)), gainDb: Number(gainDb.toFixed(1)) } : point,
                ),
              }
            : track,
        ),
      }));
      setPreviewDirty(true);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      if (automationDragRef.current && moved) {
        setPast((items) => [...items.slice(-49), automationDragRef.current!.project]);
        setFuture([]);
      }
      automationDragRef.current = null;
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const ensurePreview = async (): Promise<string | null> => {
    if (!project.tracks.length) {
      setStatus("Drop a file onto the timeline first.");
      return null;
    }
    setStatus("Rendering preview…");
    const start = project.loop.enabled ? project.loop.start : 0;
    const end = project.loop.enabled ? project.loop.end : undefined;
    const result = renderProject(project, assets, { start, end });
    const blob = encodeWav(result.audio);
    const url = URL.createObjectURL(blob);
    setPreviewUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return url;
    });
    setPreviewStart(start);
    setReport(result.report);
    setPreviewDirty(false);
    setStatus(project.loop.enabled ? "Loop rendered." : "Mix rendered.");
    return url;
  };

  async function togglePlay() {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) {
      audio.pause();
      setPlaying(false);
      return;
    }
    let url = previewUrl;
    if (!url || previewDirty) url = await ensurePreview();
    if (!url) return;
    if (audio.src !== url) audio.src = url;
    audio.loop = project.loop.enabled;
    try {
      await audio.play();
      setPlaying(true);
    } catch {
      setStatus("The browser blocked playback. Press play again.");
    }
  }

  const seek = (seconds: number) => {
    const target = clamp(seconds, 0, duration);
    setPlayhead(target);
    const audio = audioRef.current;
    if (audio && previewUrl && !previewDirty) {
      audio.currentTime = clamp(target - previewStart, 0, audio.duration || 0);
    }
  };

  const renderQueue = async (kind: "mix" | "loop" | "stems") => {
    if (!project.tracks.length) {
      setStatus("Nothing to render.");
      return;
    }
    try {
      if (kind === "stems") {
        setStatus("Rendering stems…");
        const stems = renderStems(project, assets);
        const entries: Record<string, Uint8Array> = {};
        for (const stem of stems) {
          const blob = encodeWav(stem.audio);
          entries[`${safeName(stem.name)}.wav`] = new Uint8Array(await blob.arrayBuffer());
        }
        const zip = zipSync(entries);
        const blob = new Blob([zip], { type: "application/zip" });
        downloadBlob(blob, `${safeName(project.name)}-stems.zip`);
        const url = URL.createObjectURL(blob);
        setRenders((items) => [{ id: createId("render"), label: "Stems ZIP", url, name: `${safeName(project.name)}-stems.zip` }, ...items]);
        setStatus(`Rendered ${stems.length} stems.`);
        return;
      }
      const start = kind === "loop" ? project.loop.start : 0;
      const end = kind === "loop" ? project.loop.end : undefined;
      setStatus(kind === "loop" ? "Rendering loop…" : "Rendering mix…");
      const result = renderProject(project, assets, { start, end });
      const blob = encodeWav(result.audio);
      const url = URL.createObjectURL(blob);
      setReport(result.report);
      setRenders((items) => [
        {
          id: createId("render"),
          label: kind === "loop" ? "Loop" : "Mix",
          url,
          name: `${safeName(project.name)}-${kind}.wav`,
        },
        ...items,
      ]);
      setStatus(`${kind === "loop" ? "Loop" : "Mix"} rendered: ${result.report.integratedLufs.toFixed(1)} LUFS.`);
    } catch (error) {
      setStatus(`Render failed: ${error instanceof Error ? error.message : "unknown error"}`);
    }
  };

  const exportProject = async () => {
    setStatus("Packing project…");
    const entries: Record<string, Uint8Array> = {
      "project.json": strToU8(JSON.stringify(project, null, 2)),
    };
    for (const [id, asset] of Object.entries(assets)) {
      const blob = encodeWav(asset.audio);
      entries[`assets/${id}.wav`] = new Uint8Array(await blob.arrayBuffer());
    }
    const zip = zipSync(entries);
    downloadBlob(new Blob([zip], { type: "application/zip" }), `${safeName(project.name)}.audelle.zip`);
    setStatus("Project bundle downloaded.");
  };

  const importProject = async (file: File) => {
    setStatus("Opening project…");
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const entries = unzipSync(bytes);
      const projectBytes = entries["project.json"];
      if (!projectBytes) throw new Error("project.json is missing");
      const nextProject = JSON.parse(strFromU8(projectBytes)) as StudioProject;
      const nextAssets: Record<string, StudioAsset> = {};
      for (const id of discoverAssets(nextProject)) {
        const bytesForAsset = entries[`assets/${id}.wav`];
        if (!bytesForAsset) continue;
        const assetFile = new File([bytesForAsset], `${id}.wav`, { type: "audio/wav" });
        const audio = await decodeFile(assetFile);
        nextAssets[id] = {
          id,
          name: assetFile.name,
          audio,
          duration: audio.channels[0].length / audio.sampleRate,
          sampleRate: audio.sampleRate,
          channels: audio.channels.length,
        };
        await saveStudioAsset(id, assetFile);
      }
      setProject({ ...nextProject, tempo: nextProject.tempo || 120 });
      setAssets(nextAssets);
      setPast([]);
      setFuture([]);
      setSelected(null);
      setSelectedTrackId(nextProject.tracks[0]?.id ?? null);
      setPreviewDirty(true);
      setStatus("Project opened.");
    } catch (error) {
      setStatus(`Could not open project: ${error instanceof Error ? error.message : "unknown error"}`);
    }
  };

  const newProject = () => {
    setProject(emptyProject());
    setAssets({});
    setPast([]);
    setFuture([]);
    setSelected(null);
    setSelectedTrackId(null);
    setPreviewDirty(true);
    setReport(null);
    setStatus("New empty session.");
  };

  const ticks = useMemo(() => {
    const step = zoom >= 1.8 ? 1 : zoom >= 0.9 ? 2 : 5;
    const values: number[] = [];
    for (let time = 0; time <= duration + 4; time += step) values.push(time);
    return values;
  }, [duration, zoom]);

  return (
    <div className="min-h-screen bg-black text-white">
      <Nav />
      <div className="flex min-h-screen pt-[var(--nav-h)]">
        <Sidebar onPromptSelect={() => {}} />
        <main className="min-w-0 flex-1 overflow-x-hidden">
          <header className="border-b border-white/10 px-4 py-4 sm:px-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="border-l-2 border-white/40 pl-3">
                <p className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-white/40">Studio · multitrack</p>
                <input
                  value={project.name}
                  onChange={(event) => setProject({ ...project, name: event.target.value })}
                  className="mt-1 w-full max-w-xl border-none bg-transparent font-display text-3xl font-bold lowercase leading-none tracking-[-0.04em] text-[#f2f1ec] outline-none sm:text-4xl"
                  aria-label="Project name"
                />
                <p className="mt-1 text-sm text-white/50">
                  {project.tracks.length} track{project.tracks.length === 1 ? "" : "s"} · {formatTime(contentDuration)} · preview
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <button onClick={() => fileInputRef.current?.click()} className="rounded-none bg-[#f2f1ec] px-4 py-2 text-xs font-semibold text-black hover:bg-white">
                  Import audio / video / MIDI
                </button>
                <button onClick={() => projectInputRef.current?.click()} className="rounded-none border border-white/20 px-3 py-2 text-xs text-white/70 hover:border-white/50 hover:text-white">
                  Open project
                </button>
                <button onClick={() => void exportProject()} className="rounded-none border border-white/20 px-3 py-2 text-xs text-white/70 hover:border-white/50 hover:text-white">
                  Save bundle
                </button>
                <button onClick={newProject} className="rounded-none border border-white/20 px-3 py-2 text-xs text-white/60 hover:border-white/50 hover:text-white">
                  New
                </button>
              </div>
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <button onClick={() => void togglePlay()} className="rounded-none border border-white/25 px-4 py-2 text-xs text-white/80 hover:border-white/60 hover:text-white">
                {playing ? "Pause" : "Play"}
              </button>
              <button onClick={() => seek(0)} className="rounded-none border border-white/15 px-3 py-2 text-xs text-white/55 hover:text-white">
                Return
              </button>
              <span className="font-mono text-xs text-white/55">{formatTime(playhead)} / {formatTime(contentDuration)}</span>
              <label className="ml-2 flex items-center gap-2 text-xs text-white/55">
                <input
                  type="checkbox"
                  checked={project.loop.enabled}
                  onChange={(event) => {
                    const enabled = event.target.checked;
                    commit({
                      ...project,
                      loop: {
                        enabled,
                        start: enabled ? project.loop.start || 0 : project.loop.start,
                        end: enabled ? project.loop.end || duration : project.loop.end,
                      },
                    });
                  }}
                  className="accent-white"
                />
                Loop
              </label>
              {project.loop.enabled && (
                <>
                  <input
                    type="number"
                    min={0}
                    step={0.1}
                    value={project.loop.start}
                    onChange={(event) => commit({ ...project, loop: { ...project.loop, start: Number(event.target.value) } })}
                    className="w-20 rounded-none border border-white/15 bg-white/[0.03] px-2 py-1 font-mono text-xs"
                    aria-label="Loop start"
                  />
                  <span className="text-xs text-white/35">→</span>
                  <input
                    type="number"
                    min={0}
                    step={0.1}
                    value={project.loop.end}
                    onChange={(event) => commit({ ...project, loop: { ...project.loop, end: Number(event.target.value) } })}
                    className="w-20 rounded-none border border-white/15 bg-white/[0.03] px-2 py-1 font-mono text-xs"
                    aria-label="Loop end"
                  />
                </>
              )}
              <div className="ml-auto flex items-center gap-1">
                <button onClick={() => setZoom((value) => clamp(value - 0.15, MIN_ZOOM, MAX_ZOOM))} className="rounded-none border border-white/15 px-2 py-1 text-xs text-white/60 hover:text-white">−</button>
                <span className="font-mono text-xs text-white/40">{Math.round(zoom * 100)}%</span>
                <button onClick={() => setZoom((value) => clamp(value + 0.15, MIN_ZOOM, MAX_ZOOM))} className="rounded-none border border-white/15 px-2 py-1 text-xs text-white/60 hover:text-white">+</button>
                <button onClick={undo} disabled={!past.length} className="ml-2 rounded-none border border-white/15 px-2 py-1 text-xs text-white/60 hover:text-white disabled:opacity-30">Undo</button>
                <button onClick={redo} disabled={!future.length} className="rounded-none border border-white/15 px-2 py-1 text-xs text-white/60 hover:text-white disabled:opacity-30">Redo</button>
              </div>
            </div>
          </header>

          <div className="grid min-h-[calc(100vh-var(--nav-h)-180px)] gap-0 xl:grid-cols-[minmax(0,1fr)_360px]">
            <section className="min-w-0 border-r border-white/10">
              <div className="flex items-center justify-between border-b border-white/10 px-4 py-2">
                <p className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-white/40">Timeline</p>
                <p className="text-xs text-white/40">Drag clips · click ruler to seek · split at playhead</p>
              </div>
              {!project.tracks.length ? (
                <div className="flex min-h-[420px] flex-col items-center justify-center p-8 text-center">
                  <p className="font-display text-3xl font-bold text-[#f2f1ec]">build a session.</p>
                  <p className="mt-2 max-w-md text-sm leading-relaxed text-white/50">
                    Import audio, video or MIDI files. Each becomes a track with a clip on the timeline. Video is decoded to its audio track; MIDI becomes a playable note track.
                  </p>
                  <button onClick={() => fileInputRef.current?.click()} className="mt-6 rounded-none border border-white/25 px-5 py-3 text-sm text-white/75 hover:border-white/60 hover:text-white">
                    Import audio, video or MIDI
                  </button>
                </div>
              ) : (
                <div className="overflow-auto">
                  <div className="relative" style={{ width: timelineWidth + 180 }}>
                    <div className="sticky top-0 z-20 flex border-b border-white/10 bg-black/95">
                      <div className="sticky left-0 z-30 w-44 shrink-0 border-r border-white/10 bg-black px-3 py-2 font-mono text-[0.6rem] uppercase tracking-[0.16em] text-white/35">
                        Tracks
                      </div>
                      <div className="relative h-8" style={{ width: timelineWidth }}>
                        {ticks.map((time) => (
                          <button
                            key={time}
                            onClick={() => seek(time)}
                            className="absolute top-0 h-8 border-l border-white/10 pl-1 text-left font-mono text-[0.6rem] text-white/35 hover:text-white/70"
                            style={{ left: time * pps }}
                          >
                            {formatTime(time).slice(0, -3)}
                          </button>
                        ))}
                      </div>
                    </div>
                    {project.tracks.map((track, trackIndex) => (
                      <div
                        key={track.id}
                        className="flex border-b border-white/8"
                        style={{ height: TRACK_HEIGHT + (automationOpenId === track.id ? 54 : 0) }}
                      >
                        <div className="sticky left-0 z-10 w-44 shrink-0 border-r border-white/10 bg-black px-3 py-2">
                          <div className="flex items-center gap-1.5">
                            <span className="h-2 w-2" style={{ backgroundColor: TRACK_COLORS[trackIndex % TRACK_COLORS.length] }} />
                            <input
                              value={track.name}
                              onChange={(event) => updateTrack(track.id, (item) => ({ ...item, name: event.target.value }))}
                              className="min-w-0 flex-1 bg-transparent text-xs font-medium text-white/80 outline-none"
                              aria-label="Track name"
                            />
                          </div>
                          <div className="mt-1 flex items-center gap-1">
                            <button
                              onClick={() => updateTrack(track.id, (item) => ({ ...item, mute: !item.mute }))}
                              className={`rounded-none border px-1.5 py-0.5 text-[0.6rem] ${track.mute ? "border-white/50 bg-white/15 text-white" : "border-white/15 text-white/45"}`}
                            >
                              M
                            </button>
                            <button
                              onClick={() => updateTrack(track.id, (item) => ({ ...item, solo: !item.solo }))}
                              className={`rounded-none border px-1.5 py-0.5 text-[0.6rem] ${track.solo ? "border-white/50 bg-white/15 text-white" : "border-white/15 text-white/45"}`}
                            >
                              S
                            </button>
                            <button
                              onClick={() => setAutomationOpenId((current) => (current === track.id ? null : track.id))}
                              className={`rounded-none border px-1.5 py-0.5 text-[0.6rem] ${automationOpenId === track.id ? "border-white/50 bg-white/15 text-white" : "border-white/15 text-white/45"}`}
                              title="Volume automation lane"
                            >
                              A
                            </button>
                            <button onClick={() => removeTrack(track.id)} className="ml-auto text-[0.6rem] text-white/30 hover:text-red-300">×</button>
                          </div>
                        </div>
                        <div
                          className="relative"
                          style={{ width: timelineWidth, backgroundImage: "repeating-linear-gradient(to right, rgba(255,255,255,0.035) 0, rgba(255,255,255,0.035) 1px, transparent 1px, transparent 64px)" }}
                          onClick={(event) => {
                            const rect = event.currentTarget.getBoundingClientRect();
                            seek((event.clientX - rect.left) / pps);
                          }}
                        >
                          {track.clips.map((clip) => {
                            const asset = clip.assetId ? assets[clip.assetId] : undefined;
                            const active = selected?.clipId === clip.id;
                            return (
                              <div
                                key={clip.id}
                                onPointerDown={(event) => startClipDrag(event, track, clip)}
                                onClick={(event) => event.stopPropagation()}
                                className={`absolute top-2 h-[52px] cursor-grab overflow-hidden border ${active ? "border-white/80" : "border-white/25"}`}
                                style={{
                                  left: clip.start * pps,
                                  width: Math.max(8, clip.duration * pps),
                                  backgroundColor: `${TRACK_COLORS[trackIndex % TRACK_COLORS.length]}22`,
                                }}
                              >
                                {asset && <ClipWave asset={asset} />}
                                {clip.kind === "midi" && (
                                  <div className="pointer-events-none absolute inset-x-0 bottom-1 h-6 overflow-hidden">
                                    {(clip.notes ?? []).map((note, noteIndex) => (
                                      <span
                                        key={noteIndex}
                                        className="absolute rounded-[1px] bg-white/40"
                                        style={{
                                          left: `${(note.start / Math.max(0.01, clip.duration)) * 100}%`,
                                          width: `${Math.max(0.6, (note.duration / Math.max(0.01, clip.duration)) * 100)}%`,
                                          top: `${((84 - Math.max(36, Math.min(84, note.midi))) / 48) * 100}%`,
                                          height: "12%",
                                        }}
                                      />
                                    ))}
                                  </div>
                                )}
                                <div className="relative z-10 px-2 pt-1">
                                  <p className="truncate text-[0.7rem] font-medium text-white/85">{clip.name}</p>
                                  <p className="font-mono text-[0.6rem] text-white/45">{clip.duration.toFixed(2)}s</p>
                                </div>
                                {clip.fadeIn > 0 && <div className="pointer-events-none absolute inset-y-0 left-0" style={{ width: `${Math.min(40, clip.fadeIn * pps)}px`, background: "linear-gradient(to right, rgba(0,0,0,0.55), transparent)" }} />}
                                {clip.fadeOut > 0 && <div className="pointer-events-none absolute inset-y-0 right-0" style={{ width: `${Math.min(40, clip.fadeOut * pps)}px`, background: "linear-gradient(to left, rgba(0,0,0,0.55), transparent)" }} />}
                              </div>
                            );
                          })}
                          <div className="pointer-events-none absolute bottom-0 top-0 z-10 w-px bg-[#f2f1ec]/80" style={{ left: playhead * pps }} />
                          {automationOpenId === track.id && (
                            <div
                              className="absolute inset-x-0 bottom-0 z-20 h-[54px] cursor-crosshair border-t border-white/12 bg-black/70"
                              onClick={(event) => {
                                event.stopPropagation();
                                addAutomationPoint(event, track);
                              }}
                              title="Click to add a volume point; drag a point to move it"
                            >
                              <svg width={timelineWidth} height={52} className="block">
                                {(() => {
                                  const points = [...(track.automation ?? [])].sort((a, b) => a.time - b.time);
                                  const path = points.length
                                    ? points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.time * pps} ${automationY(point.gainDb)}`).join(" ")
                                    : `M 0 ${automationY(track.gainDb)} L ${timelineWidth} ${automationY(track.gainDb)}`;
                                  return (
                                    <>
                                      <path d={path} fill="none" stroke="rgba(242,241,236,0.65)" strokeWidth={1.25} />
                                      <line x1={playhead * pps} y1={0} x2={playhead * pps} y2={52} stroke="rgba(242,241,236,0.25)" />
                                      {points.map((point) => (
                                        <circle
                                          key={point.id}
                                          cx={point.time * pps}
                                          cy={automationY(point.gainDb)}
                                          r={4}
                                          fill="#f2f1ec"
                                          stroke="#000"
                                          strokeWidth={1}
                                          className="cursor-grab"
                                          onPointerDown={(event) => startAutomationDrag(event, track.id, point.id)}
                                          onClick={(event) => event.stopPropagation()}
                                        />
                                      ))}
                                    </>
                                  );
                                })()}
                              </svg>
                              <span className="pointer-events-none absolute left-2 top-1 font-mono text-[0.55rem] uppercase tracking-[0.14em] text-white/35">
                                Volume automation
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            <aside className="space-y-4 p-4">
              <div className="border border-white/10 bg-white/[0.02] p-3">
                <p className="eyebrow mb-2">Clip</p>
                {selectedClip && selected ? (
                  <div className="space-y-2.5">
                    <p className="truncate text-sm text-white/80">{selectedClip.name}</p>
                    <label className="block text-[0.65rem] text-white/45">Start
                      <input type="number" step={0.05} value={Number(selectedClip.start.toFixed(2))} onChange={(event) => updateClip(selected.trackId, selected.clipId, (clip) => ({ ...clip, start: Math.max(0, Number(event.target.value)) }))} className="mt-1 w-full rounded-none border border-white/15 bg-white/[0.03] px-2 py-1 font-mono text-xs text-white/75" />
                    </label>
                    <label className="block text-[0.65rem] text-white/45">Duration
                      <input type="number" step={0.05} value={Number(selectedClip.duration.toFixed(2))} onChange={(event) => updateClip(selected.trackId, selected.clipId, (clip) => ({ ...clip, duration: Math.max(0.02, Number(event.target.value)) }))} className="mt-1 w-full rounded-none border border-white/15 bg-white/[0.03] px-2 py-1 font-mono text-xs text-white/75" />
                    </label>
                    <label className="block text-[0.65rem] text-white/45">Source offset
                      <input type="number" step={0.05} value={Number(selectedClip.offset.toFixed(2))} onChange={(event) => updateClip(selected.trackId, selected.clipId, (clip) => ({ ...clip, offset: Math.max(0, Number(event.target.value)) }))} className="mt-1 w-full rounded-none border border-white/15 bg-white/[0.03] px-2 py-1 font-mono text-xs text-white/75" />
                    </label>
                    <label className="block text-[0.65rem] text-white/45">Clip gain · {selectedClip.gainDb.toFixed(1)} dB
                      <input type="range" min={-24} max={12} step={0.5} value={selectedClip.gainDb} onChange={(event) => updateClip(selected.trackId, selected.clipId, (clip) => ({ ...clip, gainDb: Number(event.target.value) }))} className="mt-1 w-full accent-white" />
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      <label className="block text-[0.65rem] text-white/45">Fade in
                        <input type="number" min={0} step={0.01} value={selectedClip.fadeIn} onChange={(event) => updateClip(selected.trackId, selected.clipId, (clip) => ({ ...clip, fadeIn: Math.max(0, Number(event.target.value)) }))} className="mt-1 w-full rounded-none border border-white/15 bg-white/[0.03] px-2 py-1 font-mono text-xs text-white/75" />
                      </label>
                      <label className="block text-[0.65rem] text-white/45">Fade out
                        <input type="number" min={0} step={0.01} value={selectedClip.fadeOut} onChange={(event) => updateClip(selected.trackId, selected.clipId, (clip) => ({ ...clip, fadeOut: Math.max(0, Number(event.target.value)) }))} className="mt-1 w-full rounded-none border border-white/15 bg-white/[0.03] px-2 py-1 font-mono text-xs text-white/75" />
                      </label>
                    </div>
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      <button onClick={splitSelected} className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white">Split at playhead</button>
                      <button onClick={crossfadeSelected} className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white">Crossfade 50 ms</button>
                      <button onClick={duplicateSelected} className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white">Duplicate</button>
                      <button onClick={deleteSelected} className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/50 hover:border-red-300/50 hover:text-red-200">Delete</button>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs leading-relaxed text-white/45">Select a clip to trim its edges, set fades, split it at the playhead or duplicate it.</p>
                )}
              </div>

              <div className="border border-white/10 bg-white/[0.02] p-3">
                <p className="eyebrow mb-2">Channel strip</p>
                {selectedTrack ? (
                  <div className="space-y-2.5">
                    <p className="truncate text-sm text-white/80">{selectedTrack.name}</p>
                    <label className="block text-[0.65rem] text-white/45">Volume · {selectedTrack.gainDb.toFixed(1)} dB
                      <input type="range" min={-60} max={12} step={0.5} value={selectedTrack.gainDb} onChange={(event) => updateTrack(selectedTrack.id, (track) => ({ ...track, gainDb: Number(event.target.value) }))} className="mt-1 w-full accent-white" />
                    </label>
                    <label className="block text-[0.65rem] text-white/45">Pan · {selectedTrack.pan.toFixed(2)}
                      <input type="range" min={-1} max={1} step={0.01} value={selectedTrack.pan} onChange={(event) => updateTrack(selectedTrack.id, (track) => ({ ...track, pan: Number(event.target.value) }))} className="mt-1 w-full accent-white" />
                    </label>
                    <div className="grid grid-cols-3 gap-1.5">
                      {(["low", "mid", "high"] as const).map((band) => (
                        <label key={band} className="block text-[0.6rem] uppercase tracking-[0.1em] text-white/40">{band}
                          <input type="range" min={-12} max={12} step={0.5} value={selectedTrack.eq[band]} onChange={(event) => updateTrack(selectedTrack.id, (track) => ({ ...track, eq: { ...track.eq, [band]: Number(event.target.value) } }))} className="mt-1 w-full accent-white" />
                        </label>
                      ))}
                    </div>
                    <label className="flex items-center gap-2 text-[0.7rem] text-white/55">
                      <input type="checkbox" checked={selectedTrack.compressor.enabled} onChange={(event) => updateTrack(selectedTrack.id, (track) => ({ ...track, compressor: { ...track.compressor, enabled: event.target.checked } }))} className="accent-white" />
                      Compressor
                    </label>
                    {selectedTrack.compressor.enabled && (
                      <div className="grid grid-cols-2 gap-2">
                        <label className="block text-[0.6rem] text-white/40">Threshold
                          <input type="number" value={selectedTrack.compressor.thresholdDb} onChange={(event) => updateTrack(selectedTrack.id, (track) => ({ ...track, compressor: { ...track.compressor, thresholdDb: Number(event.target.value) } }))} className="mt-1 w-full rounded-none border border-white/15 bg-white/[0.03] px-2 py-1 font-mono text-xs" />
                        </label>
                        <label className="block text-[0.6rem] text-white/40">Ratio
                          <input type="number" min={1} max={20} step={0.5} value={selectedTrack.compressor.ratio} onChange={(event) => updateTrack(selectedTrack.id, (track) => ({ ...track, compressor: { ...track.compressor, ratio: Number(event.target.value) } }))} className="mt-1 w-full rounded-none border border-white/15 bg-white/[0.03] px-2 py-1 font-mono text-xs" />
                        </label>
                      </div>
                    )}
                    <div className="border-t border-white/10 pt-2">
                      <div className="flex items-center justify-between">
                        <p className="text-[0.6rem] uppercase tracking-[0.1em] text-white/40">Volume automation</p>
                        <button
                          onClick={() =>
                            updateTrack(selectedTrack.id, (track) => ({
                              ...track,
                              automation: [
                                ...(track.automation ?? []).filter((point) => Math.abs(point.time - playhead) > 0.02),
                                { id: createId("auto"), time: Number(playhead.toFixed(3)), gainDb: track.gainDb },
                              ].sort((a, b) => a.time - b.time),
                            }))
                          }
                          className="rounded-none border border-white/15 px-1.5 py-0.5 text-[0.6rem] text-white/55 hover:text-white"
                        >
                          + at playhead
                        </button>
                      </div>
                      {(selectedTrack.automation ?? []).length === 0 ? (
                        <p className="mt-1 text-[0.65rem] text-white/30">No points. The fader is static.</p>
                      ) : (
                        <ul className="mt-1 space-y-1">
                          {(selectedTrack.automation ?? []).map((point) => (
                            <li key={point.id} className="flex items-center gap-1.5 font-mono text-[0.65rem]">
                              <span className="w-14 text-white/40">{formatTime(point.time)}</span>
                              <input
                                type="number"
                                value={point.gainDb}
                                onChange={(event) =>
                                  updateTrack(selectedTrack.id, (track) => ({
                                    ...track,
                                    automation: (track.automation ?? []).map((item) =>
                                      item.id === point.id ? { ...item, gainDb: Number(event.target.value) } : item,
                                    ),
                                  }))
                                }
                                className="w-16 rounded-none border border-white/15 bg-white/[0.03] px-1.5 py-0.5 text-right"
                                aria-label="Automation gain"
                              />
                              <span className="text-white/35">dB</span>
                              <button
                                onClick={() =>
                                  updateTrack(selectedTrack.id, (track) => ({
                                    ...track,
                                    automation: (track.automation ?? []).filter((item) => item.id !== point.id),
                                  }))
                                }
                                className="ml-auto text-white/30 hover:text-red-300"
                                aria-label="Remove automation point"
                              >
                                ×
                              </button>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                    <div className="border-t border-white/10 pt-2">
                      <label className="flex items-center gap-2 text-[0.7rem] text-white/55">
                        <input
                          type="checkbox"
                          checked={Boolean(selectedTrack.sidechain?.enabled)}
                          onChange={(event) =>
                            updateTrack(selectedTrack.id, (track) => ({
                              ...track,
                              sidechain: {
                                enabled: event.target.checked,
                                sourceTrackId:
                                  track.sidechain?.sourceTrackId ??
                                  project.tracks.find((item) => item.id !== track.id)?.id ??
                                  "",
                                amountDb: track.sidechain?.amountDb ?? 6,
                              },
                            }))
                          }
                          className="accent-white"
                        />
                        Sidechain duck
                      </label>
                      {selectedTrack.sidechain?.enabled && (
                        <div className="mt-1.5 grid grid-cols-2 gap-2">
                          <label className="block text-[0.6rem] text-white/40">From
                            <select
                              value={selectedTrack.sidechain.sourceTrackId}
                              onChange={(event) => updateTrack(selectedTrack.id, (track) => ({ ...track, sidechain: { ...(track.sidechain ?? { enabled: true, amountDb: 6 }), sourceTrackId: event.target.value } }))}
                              className="mt-1 w-full rounded-none border border-white/15 bg-white/[0.03] px-1.5 py-1 text-xs text-white"
                            >
                              <option value="">Choose…</option>
                              {project.tracks.filter((item) => item.id !== selectedTrack.id).map((item) => (
                                <option key={item.id} value={item.id}>{item.name}</option>
                              ))}
                            </select>
                          </label>
                          <label className="block text-[0.6rem] text-white/40">Duck · {selectedTrack.sidechain.amountDb} dB
                            <input type="range" min={0} max={24} step={1} value={selectedTrack.sidechain.amountDb} onChange={(event) => updateTrack(selectedTrack.id, (track) => ({ ...track, sidechain: { ...(track.sidechain ?? { enabled: true, sourceTrackId: "" }), amountDb: Number(event.target.value) } }))} className="mt-1 w-full accent-white" />
                          </label>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-white/45">Select a track to open its channel strip.</p>
                )}
              </div>

              <div className="border border-white/10 bg-white/[0.02] p-3">
                <div className="flex items-center justify-between">
                  <p className="eyebrow">Master</p>
                  <button onClick={() => void renderQueue("mix")} className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white">Render mix</button>
                </div>
                <label className="mt-3 block text-[0.65rem] text-white/45">Master gain · {project.master.gainDb.toFixed(1)} dB
                  <input type="range" min={-24} max={12} step={0.5} value={project.master.gainDb} onChange={(event) => commit({ ...project, master: { ...project.master, gainDb: Number(event.target.value) } })} className="mt-1 w-full accent-white" />
                </label>
                <label className="mt-2 flex items-center gap-2 text-[0.7rem] text-white/55">
                  <input type="checkbox" checked={project.master.limiter} onChange={(event) => commit({ ...project, master: { ...project.master, limiter: event.target.checked } })} className="accent-white" />
                  Lookahead limiter
                </label>
                {report && (
                  <div className="mt-3 border-t border-white/10 pt-3">
                    <div className="grid grid-cols-2 gap-2 font-mono text-[0.7rem]">
                      <span className="text-white/45">Integrated</span><span className="text-right text-white/80">{formatLufs(report.integratedLufs)}</span>
                      <span className="text-white/45">Short-term</span><span className="text-right text-white/80">{formatLufs(report.shortTermLufs)}</span>
                      <span className="text-white/45">True peak</span><span className="text-right text-white/80">{formatDbtp(report.truePeakDbtp)}</span>
                      <span className="text-white/45">Crest</span><span className="text-right text-white/80">{Number.isFinite(report.crestDb) ? `${report.crestDb.toFixed(1)} dB` : "—"}</span>
                      <span className="text-white/45">DC offset</span><span className="text-right text-white/80">{report.dcOffset.toFixed(4)}</span>
                      <span className="text-white/45">Stereo</span><span className="text-right text-white/80">{report.stereoCorrelation.toFixed(2)}</span>
                    </div>
                    <div className="mt-2 space-y-0.5 border-t border-white/8 pt-2 text-[0.65rem]">
                      {[
                        ["Headroom", report.truePeakDbtp <= -1],
                        ["Not silent", report.integratedLufs > -40],
                        ["DC safe", report.dcOffset < 0.01],
                        ["Mono safe", report.stereoCorrelation > -0.2],
                      ].map(([label, pass]) => (
                        <p key={String(label)} className="flex justify-between">
                          <span className="text-white/40">{label}</span>
                          <span className={pass ? "text-white/75" : "text-red-300"}>{pass ? "pass" : "check"}</span>
                        </p>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="border border-white/10 bg-white/[0.02] p-3">
                <p className="eyebrow mb-2">Export & interchange</p>
                <label className="mb-2 flex items-center justify-between gap-2 text-[0.65rem] text-white/45">
                  MIDI tempo
                  <span className="flex items-center gap-1">
                    <input
                      type="number"
                      min={20}
                      max={300}
                      value={project.tempo || 120}
                      onChange={(event) => commit({ ...project, tempo: clamp(Number(event.target.value) || 120, 20, 300) })}
                      className="w-16 rounded-none border border-white/15 bg-white/[0.03] px-1.5 py-0.5 text-right font-mono text-xs text-white/75"
                      aria-label="MIDI tempo"
                    />
                    <span>BPM</span>
                  </span>
                </label>
                <div className="flex flex-wrap gap-1.5">
                  <button onClick={() => void renderQueue("loop")} disabled={!project.loop.enabled} className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white disabled:opacity-30">Render loop</button>
                  <button onClick={() => void renderQueue("stems")} className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white">Stems ZIP</button>
                  <button onClick={() => downloadText(buildEdl(project, assets), `${safeName(project.name)}.edl`)} className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white">EDL</button>
                  <button onClick={() => downloadText(buildFcpxml(project, assets), `${safeName(project.name)}.fcpxml`, "application/xml")} className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white">FCPXML</button>
                  <button
                    onClick={() => {
                      const midi = buildMidiFile(project);
                      const bytes = new Uint8Array(midi);
                      downloadBlob(new Blob([bytes.buffer as ArrayBuffer], { type: "audio/midi" }), `${safeName(project.name)}.mid`);
                    }}
                    className="rounded-none border border-white/20 px-2 py-1 text-[0.65rem] text-white/65 hover:border-white/50 hover:text-white"
                  >
                    MIDI
                  </button>
                </div>
                <p className="mt-2 text-[0.65rem] leading-relaxed text-white/35">
                  EDL opens in Reaper, Premiere and Resolve. FCPXML opens in Final Cut. Project bundles include the WAV assets.
                </p>
                {renders.length > 0 && (
                  <ul className="mt-3 space-y-1 border-t border-white/10 pt-2">
                    {renders.slice(0, 5).map((render) => (
                      <li key={render.id} className="flex items-center justify-between gap-2 text-[0.7rem]">
                        <span className="text-white/55">{render.label}</span>
                        <a href={render.url} download={render.name} className="text-white/80 underline underline-offset-2 hover:text-white">Download</a>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </aside>
          </div>

          <section className="border-t border-white/10 px-4 py-4 sm:px-6">
            <div className="flex items-center justify-between">
              <p className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-white/40">Mixer</p>
              <p className="text-xs text-white/35">{project.tracks.length} channels · master ceiling {project.master.ceilingDb} dB</p>
            </div>
            <div className="mt-3 flex gap-2 overflow-x-auto pb-2">
              {project.tracks.map((track, index) => (
                <div key={track.id} className={`min-w-[158px] border p-2.5 ${selectedTrackId === track.id ? "border-white/50" : "border-white/10"}`}>
                  <div className="flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5" style={{ backgroundColor: TRACK_COLORS[index % TRACK_COLORS.length] }} />
                    <button onClick={() => setSelectedTrackId(track.id)} className="min-w-0 flex-1 truncate text-left text-xs text-white/75">{track.name}</button>
                  </div>
                  <input type="range" min={-60} max={12} step={0.5} value={track.gainDb} onChange={(event) => updateTrack(track.id, (item) => ({ ...item, gainDb: Number(event.target.value) }))} className="mt-2 w-full accent-white" aria-label={`${track.name} volume`} />
                  <div className="flex justify-between font-mono text-[0.6rem] text-white/40"><span>VOL</span><span>{track.gainDb.toFixed(1)}</span></div>
                  <input type="range" min={-1} max={1} step={0.01} value={track.pan} onChange={(event) => updateTrack(track.id, (item) => ({ ...item, pan: Number(event.target.value) }))} className="mt-1 w-full accent-white" aria-label={`${track.name} pan`} />
                  <div className="flex justify-between font-mono text-[0.6rem] text-white/40"><span>PAN</span><span>{track.pan.toFixed(2)}</span></div>
                </div>
              ))}
              <div className="min-w-[158px] border border-white/20 bg-white/[0.03] p-2.5">
                <p className="text-xs text-white/75">Master</p>
                <input type="range" min={-24} max={12} step={0.5} value={project.master.gainDb} onChange={(event) => commit({ ...project, master: { ...project.master, gainDb: Number(event.target.value) } })} className="mt-2 w-full accent-white" aria-label="Master volume" />
                <div className="flex justify-between font-mono text-[0.6rem] text-white/40"><span>GAIN</span><span>{project.master.gainDb.toFixed(1)}</span></div>
                <p className="mt-2 font-mono text-[0.6rem] text-white/35">{report ? formatLufs(report.integratedLufs) : "LUFS after render"}</p>
              </div>
            </div>
          </section>

          {status && <p className="border-t border-white/10 px-4 py-2 font-mono text-[0.65rem] text-white/45">{status}</p>}
        </main>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".mp3,.wav,.flac,.aac,.m4a,.ogg,.mp4,.mov,.mkv,.avi,.mid,.midi,audio/*,video/mp4,video/quicktime,video/x-matroska,video/x-msvideo"
        multiple
        className="hidden"
        onChange={(event) => {
          if (event.target.files) void addFiles(event.target.files);
          event.target.value = "";
        }}
      />
      <input
        ref={projectInputRef}
        type="file"
        accept=".zip,.audelle"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void importProject(file);
          event.target.value = "";
        }}
      />
      <audio
        ref={audioRef}
        onTimeUpdate={(event) => setPlayhead(previewStart + event.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        className="hidden"
      />
    </div>
  );
}
