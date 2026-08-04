"use client";

import { useEffect, useRef, useState } from "react";

interface ManualEditorProps {
  src: string;
  onApply: (start: number, end: number) => void;
}

export default function ManualEditor({ src, onApply }: ManualEditorProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [duration, setDuration] = useState(0);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const [current, setCurrent] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onLoaded = () => {
      setDuration(el.duration || 0);
      setEnd(el.duration || 0);
    };
    const onTime = () => setCurrent(el.currentTime);
    el.addEventListener("loadedmetadata", onLoaded);
    el.addEventListener("timeupdate", onTime);
    return () => {
      el.removeEventListener("loadedmetadata", onLoaded);
      el.removeEventListener("timeupdate", onTime);
    };
  }, []);

  const fmt = (s: number) => {
    if (!isFinite(s)) return "0:00";
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const setMark = (which: "start" | "end") => {
    const el = audioRef.current;
    if (!el) return;
    const t = el.currentTime;
    if (which === "start") {
      const s = Math.min(t, end);
      setStart(s);
      el.currentTime = s;
    } else {
      const e = Math.max(t, start);
      setEnd(e);
      el.currentTime = e;
    }
  };

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) {
      if (el.currentTime >= end) el.currentTime = start;
      void el.play();
    } else {
      el.pause();
    }
  };

  return (
    <div className="space-y-3">
      <audio ref={audioRef} src={src} preload="metadata" className="hidden" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} />
      <div className="relative h-2 rounded-full bg-white/10">
        <div
          className="absolute top-0 h-full rounded-full bg-gradient-to-r from-neon-blue to-neon-purple"
          style={{ left: `${(start / (duration || 1)) * 100}%`, width: `${((end - start) / (duration || 1)) * 100}%` }}
        />
      </div>
      <div className="flex justify-between text-[11px] text-white/40">
        <span>Current: {fmt(current)}</span>
        <span>Duration: {fmt(duration)}</span>
      </div>
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={toggle} className="rounded-lg bg-neon-blue/15 px-3 py-1.5 text-xs text-neon-blue hover:bg-neon-blue/25 transition-colors">
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={() => setMark("start")} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/50 hover:border-white/20 transition-colors">
          Set Start
        </button>
        <button type="button" onClick={() => setMark("end")} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/50 hover:border-white/20 transition-colors">
          Set End
        </button>
        <button
          type="button"
          onClick={() => onApply(start, end)}
          disabled={!(end > start)}
          className="rounded-lg bg-gradient-to-r from-neon-blue to-neon-purple px-3 py-1.5 text-xs font-semibold text-black hover:scale-105 transition-all disabled:opacity-40"
        >
          Apply Trim
        </button>
      </div>
    </div>
  );
}