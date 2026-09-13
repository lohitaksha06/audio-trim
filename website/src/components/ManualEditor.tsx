"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { processAudio, type ProcessResponse } from "@/services/api";

interface ManualEditorProps {
  /** Playable URL (object URL or /api/export/download URL). */
  src: string;
  /** Server audio_path / storage key that edits chain onto. */
  audioPath: string;
  /** Page updates its chaining state + output panel from the result. */
  onResult: (res: ProcessResponse, label: string) => void;
}

const PEAKS = 900;

const fmt = (s: number) => {
  if (!isFinite(s) || s < 0) return "0:00.0";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
};

export default function ManualEditor({ src, audioPath, onResult }: ManualEditorProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const peaksRef = useRef<Float32Array | null>(null);
  const dragRef = useRef<"start" | "end" | "select" | null>(null);
  const dragAnchorRef = useRef(0);

  const [duration, setDuration] = useState(0);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const [current, setCurrent] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loadingWave, setLoadingWave] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState("");

  // adjust controls
  const [fadeIn, setFadeIn] = useState(0);
  const [fadeOut, setFadeOut] = useState(0);
  const [gainDb, setGainDb] = useState(0);
  const [speed, setSpeed] = useState(1);

  // decode peaks whenever the source changes
  useEffect(() => {
    let cancelled = false;
    setLoadingWave(true);
    peaksRef.current = null;
    setStart(0);
    setEnd(0);
    setCurrent(0);
    (async () => {
      try {
        const res = await fetch(src);
        const buf = await res.arrayBuffer();
        const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const ctx = new Ctx();
        const decoded = await ctx.decodeAudioData(buf);
        if (cancelled) return;
        const ch = decoded.getChannelData(0);
        const peaks = new Float32Array(PEAKS);
        const block = Math.max(1, Math.floor(ch.length / PEAKS));
        for (let i = 0; i < PEAKS; i++) {
          let max = 0;
          const off = i * block;
          for (let j = off; j < Math.min(off + block, ch.length); j += 7) {
            const v = Math.abs(ch[j]);
            if (v > max) max = v;
          }
          peaks[i] = max;
        }
        peaksRef.current = peaks;
        setDuration(decoded.duration);
        setEnd(decoded.duration);
        void ctx.close();
      } catch {
        if (!cancelled) setError("Could not draw waveform — transport still works.");
      } finally {
        if (!cancelled) setLoadingWave(false);
      }
    })();
    return () => { cancelled = true; };
  }, [src]);

  // draw waveform + selection + handles + playhead
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const dur = duration || 1;
    const xOf = (t: number) => (t / dur) * w;

    // peaks
    const peaks = peaksRef.current;
    if (peaks) {
      const grad = ctx.createLinearGradient(0, h, 0, 0);
      grad.addColorStop(0, "#0affc9");
      grad.addColorStop(1, "#7c3aed");
      ctx.fillStyle = grad;
      const bw = w / peaks.length;
      for (let i = 0; i < peaks.length; i++) {
        const ph = Math.max(1.5, peaks[i] * (h - 8));
        ctx.fillRect(i * bw, (h - ph) / 2, Math.max(1, bw - 0.4), ph);
      }
    }

    // dim outside selection
    if (end > start) {
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(0, 0, xOf(start), h);
      ctx.fillRect(xOf(end), 0, w - xOf(end), h);
      // selection tint
      ctx.fillStyle = "rgba(0,212,255,0.08)";
      ctx.fillRect(xOf(start), 0, xOf(end) - xOf(start), h);
      // handles
      for (const t of [start, end]) {
        const x = xOf(t);
        ctx.fillStyle = "#00d4ff";
        ctx.fillRect(x - 2, 0, 4, h);
        ctx.beginPath();
        ctx.arc(x, 10, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // playhead
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillRect(xOf(current) - 1, 0, 2, h);
  }, [duration, start, end, current]);

  useEffect(() => { draw(); }, [draw]);
  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  // audio element events
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onTime = () => {
      setCurrent(el.currentTime);
      if (playing && end > start && el.currentTime >= end) el.pause();
    };
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    return () => {
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
    };
  }, [playing, start, end]);

  const seekFromEvent = (e: React.PointerEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || !duration) return 0;
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    return ratio * duration;
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!duration) return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const t = seekFromEvent(e);
    const canvas = canvasRef.current!;
    const w = canvas.clientWidth;
    const xOf = (tt: number) => (tt / duration) * w;
    const x = ((e.clientX - canvas.getBoundingClientRect().left));
    if (Math.abs(x - xOf(start)) < 12) dragRef.current = "start";
    else if (Math.abs(x - xOf(end)) < 12) dragRef.current = "end";
    else {
      dragRef.current = "select";
      dragAnchorRef.current = t;
      setStart(t);
      setEnd(t);
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const mode = dragRef.current;
    if (!mode || !duration) return;
    const t = seekFromEvent(e);
    if (mode === "start") setStart(Math.min(t, end));
    else if (mode === "end") setEnd(Math.max(t, start));
    else {
      const a = dragAnchorRef.current;
      setStart(Math.min(a, t));
      setEnd(Math.max(a, t));
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (dragRef.current === "select") {
      // plain click (no drag) = seek
      if (Math.abs(end - start) < 0.05 && audioRef.current) {
        audioRef.current.currentTime = start;
        setCurrent(start);
      }
    }
    dragRef.current = null;
  };

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) {
      if (end > start && (el.currentTime < start || el.currentTime >= end)) el.currentTime = start;
      void el.play();
    } else el.pause();
  };

  const playSelection = () => {
    const el = audioRef.current;
    if (!el || !(end > start)) return;
    el.currentTime = start;
    void el.play();
  };

  const runOp = async (label: string, prompt: string) => {
    setBusy(label);
    setError("");
    try {
      const res = await processAudio(audioPath, prompt);
      onResult(res, label);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : `${label} failed`);
    } finally {
      setBusy(null);
    }
  };

  const selLen = end - start;
  const hasSel = selLen > 0.05 && duration > 0;

  return (
    <div className="space-y-4">
      <audio ref={audioRef} src={src} preload="metadata" className="hidden" />

      <div className="rounded-2xl border border-white/10 bg-black/40 p-3">
        <div className="mb-2 flex items-center justify-between text-[11px] text-white/40">
          <span>{loadingWave ? "Drawing waveform…" : "Drag on the waveform to select · drag the blue handles to trim · click to seek"}</span>
          <span className="font-mono">{fmt(current)} / {fmt(duration)}</span>
        </div>
        <canvas
          ref={canvasRef}
          className="h-36 w-full cursor-crosshair touch-none rounded-xl"
          style={{ height: 144 }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        />
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button type="button" onClick={toggle} className="rounded-lg bg-neon-blue/15 px-3 py-1.5 text-xs text-neon-blue hover:bg-neon-blue/25 transition-colors">
            {playing ? "Pause" : "Play"}
          </button>
          <button type="button" onClick={playSelection} disabled={!hasSel} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/50 hover:border-white/20 transition-colors disabled:opacity-40">
            Play selection
          </button>
          <button type="button" onClick={() => { if (audioRef.current) { const t = audioRef.current.currentTime; setStart(Math.min(t, end || duration)); } }} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/50 hover:border-white/20 transition-colors">
            Set Start {fmt(start)}
          </button>
          <button type="button" onClick={() => { if (audioRef.current) { const t = audioRef.current.currentTime; setEnd(Math.max(t, start)); } }} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/50 hover:border-white/20 transition-colors">
            Set End {fmt(end)}
          </button>
          <button type="button" onClick={() => { setStart(0); setEnd(duration); }} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/30 hover:text-white/60 transition-colors">
            Select all
          </button>
          {hasSel && <span className="text-[11px] text-neon-blue/70">Selected {selLen.toFixed(1)}s</span>}
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/50">Region</h4>
          <div className="flex flex-col gap-2">
            <button type="button" onClick={() => runOp("Keep selection", `trim from ${start.toFixed(2)} to ${end.toFixed(2)}`)} disabled={!hasSel || !!busy} className="rounded-lg bg-gradient-to-r from-neon-blue to-neon-purple px-3 py-2 text-xs font-semibold text-black hover:scale-[1.01] transition-transform disabled:opacity-40">
              {busy === "Keep selection" ? "Working…" : "Keep selection (trim)"}
            </button>
            <button type="button" onClick={() => runOp("Cut selection", `remove the section from ${start.toFixed(2)} to ${end.toFixed(2)}`)} disabled={!hasSel || !!busy} className="rounded-lg border border-red-400/20 bg-red-500/10 px-3 py-2 text-xs text-red-300 hover:bg-red-500/20 transition-colors disabled:opacity-40">
              {busy === "Cut selection" ? "Working…" : "Cut selection out"}
            </button>
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/50">Fades</h4>
          <label className="mb-2 flex items-center gap-2 text-xs text-white/50">
            <span className="w-14">Fade in</span>
            <input type="range" min={0} max={10} step={0.5} value={fadeIn} onChange={(e) => setFadeIn(Number(e.target.value))} className="flex-1 accent-neon-blue" />
            <span className="w-10 text-right font-mono text-white/70">{fadeIn.toFixed(1)}s</span>
          </label>
          <label className="mb-2 flex items-center gap-2 text-xs text-white/50">
            <span className="w-14">Fade out</span>
            <input type="range" min={0} max={10} step={0.5} value={fadeOut} onChange={(e) => setFadeOut(Number(e.target.value))} className="flex-1 accent-neon-blue" />
            <span className="w-10 text-right font-mono text-white/70">{fadeOut.toFixed(1)}s</span>
          </label>
          <button type="button" onClick={() => runOp("Fades", `fade in ${fadeIn.toFixed(1)}s and fade out ${fadeOut.toFixed(1)}s`)} disabled={(fadeIn <= 0 && fadeOut <= 0) || !!busy} className="w-full rounded-lg bg-neon-blue/15 px-3 py-2 text-xs font-medium text-neon-blue hover:bg-neon-blue/25 transition-colors disabled:opacity-40">
            {busy === "Fades" ? "Working…" : "Apply fades"}
          </button>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/50">Volume</h4>
          <label className="mb-2 flex items-center gap-2 text-xs text-white/50">
            <span className="w-14">Gain</span>
            <input type="range" min={-12} max={12} step={0.5} value={gainDb} onChange={(e) => setGainDb(Number(e.target.value))} className="flex-1 accent-neon-purple" />
            <span className="w-14 text-right font-mono text-white/70">{gainDb > 0 ? `+${gainDb.toFixed(1)}` : gainDb.toFixed(1)} dB</span>
          </label>
          <div className="flex gap-2">
            <button type="button" onClick={() => runOp("Gain", `set gain to ${gainDb >= 0 ? "+" : ""}${gainDb.toFixed(1)}dB`)} disabled={gainDb === 0 || !!busy} className="flex-1 rounded-lg bg-neon-purple/15 px-3 py-2 text-xs font-medium text-neon-purple hover:bg-neon-purple/25 transition-colors disabled:opacity-40">
              {busy === "Gain" ? "Working…" : "Apply gain"}
            </button>
            <button type="button" onClick={() => runOp("Normalize", "normalize the volume")} disabled={!!busy} className="flex-1 rounded-lg border border-white/10 px-3 py-2 text-xs text-white/50 hover:border-white/20 transition-colors disabled:opacity-40">
              {busy === "Normalize" ? "Working…" : "Normalize"}
            </button>
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/50">Speed</h4>
          <label className="mb-2 flex items-center gap-2 text-xs text-white/50">
            <span className="w-14">Rate</span>
            <input type="range" min={0.5} max={2} step={0.05} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="flex-1 accent-neon-blue" />
            <span className="w-14 text-right font-mono text-white/70">{speed.toFixed(2)}x</span>
          </label>
          <button
            type="button"
            onClick={() => runOp("Speed", speed >= 1 ? `make it ${speed.toFixed(2)}x faster` : `make it ${(1 / speed).toFixed(2)}x slower`)}
            disabled={Math.abs(speed - 1) < 0.01 || !!busy}
            className="w-full rounded-lg bg-neon-blue/15 px-3 py-2 text-xs font-medium text-neon-blue hover:bg-neon-blue/25 transition-colors disabled:opacity-40"
          >
            {busy === "Speed" ? "Working…" : "Apply speed"}
          </button>
        </div>
      </div>

      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}
