"use client";

import { useEffect, useRef, useState } from "react";
import type { CurvePoint } from "@/services/api";

interface AudioPreviewProps {
  src: string;
  curve?: CurvePoint[];
  height?: number;
}

export default function AudioPreview({ src, curve, height = 64 }: AudioPreviewProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onTime = () => {
      if (el.duration) setProgress(el.currentTime / el.duration);
    };
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("ended", onPause);
    return () => {
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("ended", onPause);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, canvas.clientWidth, height);

    const max = Math.max(...(curve?.map((p) => p.energy) || [0.1]), 0.1);
    const points = curve || [];
    const n = points.length || 1;
    const barW = canvas.clientWidth / Math.min(n, 200);
    points.forEach((p, i) => {
      const h = Math.max(2, (p.energy / max) * (height - 4));
      const x = (i / n) * canvas.clientWidth;
      const grad = ctx.createLinearGradient(0, height, 0, 0);
      grad.addColorStop(0, "#0affc9");
      grad.addColorStop(1, "#7c3aed");
      ctx.fillStyle = grad;
      ctx.fillRect(x, height - h, barW - 1, h);
    });

    // playhead overlay
    const px = progress * canvas.clientWidth;
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.fillRect(px, 0, 2, height);
  }, [curve, progress, height]);

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  };

  return (
    <div className="w-full">
      <canvas
        ref={canvasRef}
        className="w-full cursor-pointer rounded-lg bg-white/[0.02]"
        style={{ height: `${height}px` }}
        onClick={toggle}
        title="Toggle preview"
      />
      <audio ref={audioRef} src={src} preload="metadata" className="hidden" />
      <button
        type="button"
        onClick={toggle}
        className="mt-2 inline-flex items-center gap-2 rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/50 hover:border-white/20 hover:text-white/80 transition-colors"
      >
        <svg className="h-3.5 w-3.5" fill="currentColor" viewBox="0 0 24 24">
          {playing ? (
            <path d="M8 5v14l11-7z" />
          ) : (
            <path d="M6 4h4v16H6zM14 4h4v16h-4z" />
          )}
        </svg>
        {playing ? "Pause" : "Preview"}
      </button>
    </div>
  );
}