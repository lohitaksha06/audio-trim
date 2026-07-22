"use client";

import { useEffect, useRef } from "react";

export default function AudioWave() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };
    resize();
    window.addEventListener("resize", resize);

    let time = 0;

    const animate = () => {
      time += 0.008;
      const w = canvas.width / window.devicePixelRatio;
      const h = canvas.height / window.devicePixelRatio;

      ctx.clearRect(0, 0, w, h);

      const waves = [
        { color: "rgba(0, 212, 255,", amp: 30, freq: 0.02, phase: 0, alpha: 0.6 },
        { color: "rgba(155, 89, 182,", amp: 25, freq: 0.025, phase: 1.5, alpha: 0.5 },
        { color: "rgba(0, 212, 255,", amp: 20, freq: 0.018, phase: 3, alpha: 0.4 },
        { color: "rgba(192, 132, 252,", amp: 18, freq: 0.03, phase: 4.5, alpha: 0.35 },
        { color: "rgba(0, 212, 255,", amp: 15, freq: 0.022, phase: 6, alpha: 0.25 },
        { color: "rgba(155, 89, 182,", amp: 12, freq: 0.035, phase: 7.5, alpha: 0.2 },
      ];

      for (let i = 0; i < waves.length; i++) {
        const { color, amp, freq, phase, alpha } = waves[i];
        ctx.beginPath();
        ctx.moveTo(0, h / 2);

        for (let x = 0; x <= w; x += 2) {
          const y =
            h / 2 +
            Math.sin(x * freq + time + phase) * amp +
            Math.sin(x * 0.01 + time * 1.3 + phase) * amp * 0.5 +
            Math.sin(x * 0.04 + time * 0.7 + phase) * amp * 0.3;
          ctx.lineTo(x, y);
        }

        ctx.strokeStyle = `${color} ${alpha})`;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = color.replace("rgba(", "").replace(",", "") + " 1)";
        ctx.shadowBlur = 15 + i * 3;
        ctx.stroke();
      }

      ctx.shadowBlur = 0;

      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none opacity-70"
    />
  );
}
