"use client";

import { useEffect, useRef } from "react";

/**
 * Decorative segmented equalizer, blue -> pink -> purple.
 *
 * Each column is a stack of rounded cells that pumps like a DJ meter, with a
 * fading mirrored reflection under the baseline. Drawn on one canvas rather
 * than with DOM nodes: a full-width band is ~60 columns x ~30 cells on a wide
 * screen, which is far cheaper to paint than 1,800 elements.
 */

type Stop = { at: number; rgb: [number, number, number] };

const STOPS: Stop[] = [
  { at: 0.0, rgb: [43, 75, 255] },
  { at: 0.3, rgb: [77, 124, 255] },
  { at: 0.55, rgb: [255, 79, 216] },
  { at: 0.78, rgb: [217, 70, 239] },
  { at: 1.0, rgb: [155, 89, 182] },
];

function sampleColor(position: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, position));
  for (let i = 1; i < STOPS.length; i++) {
    if (x <= STOPS[i].at) {
      const a = STOPS[i - 1];
      const b = STOPS[i];
      const f = (x - a.at) / Math.max(1e-6, b.at - a.at);
      return [
        Math.round(a.rgb[0] + (b.rgb[0] - a.rgb[0]) * f),
        Math.round(a.rgb[1] + (b.rgb[1] - a.rgb[1]) * f),
        Math.round(a.rgb[2] + (b.rgb[2] - a.rgb[2]) * f),
      ];
    }
  }
  return STOPS[STOPS.length - 1].rgb;
}

export default function AudioWave() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    let levels: number[] = [];
    let frame = 0;
    let elapsed = 0;
    let lastTimestamp = 0;
    let redraw: (() => void) | null = null;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      // setTransform (not scale) so repeated resizes do not compound.
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      levels = [];
      // Setting canvas.width clears the bitmap; repaint a static frame when the
      // animation loop is not running (reduced motion).
      redraw?.();
    };
    resize();

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    const draw = (time: number) => {
      const dt = Math.min(0.05, (time - lastTimestamp) / 1000 || 0);
      lastTimestamp = time;
      elapsed += dt;

      const baselineY = height * 0.76;
      const maxBarH = height * 0.6;

      const count = Math.max(14, Math.min(64, Math.round(width / 30)));
      const barW = width / (1.5 * count - 0.5);
      const gapX = barW * 0.5;

      const segH = Math.max(4, Math.min(9, height * 0.026));
      const segGap = Math.max(2, segH * 0.45);
      const step = segH + segGap;
      const maxSegs = Math.max(1, Math.floor(maxBarH / step));

      if (levels.length !== count) {
        levels = new Array(count).fill(0.32);
      }

      for (let i = 0; i < count; i++) {
        let target: number;
        if (reduceMotion) {
          target = 0.34 + 0.34 * Math.abs(Math.sin(i * 0.9));
        } else {
          // Pseudo-random per-column weight so the row never looks uniform.
          const personality = 0.72 + 0.28 * (0.5 + 0.5 * Math.sin(i * 12.9898));
          target =
            0.3 +
            0.3 * Math.abs(Math.sin(elapsed * 1.7 + i * 0.5)) +
            0.2 * Math.abs(Math.sin(elapsed * 2.9 - i * 0.85 + 1.1)) +
            0.12 * Math.abs(Math.sin(elapsed * 4.7 + i * 1.6 + 2.3)) +
            0.07 * (0.5 + 0.5 * Math.sin(elapsed * 0.9 - i * 0.18));
          target *= personality;
        }
        target = Math.min(1, Math.max(0.05, target));
        // Ease toward the target so column tips glide instead of snapping.
        levels[i] += (target - levels[i]) * Math.min(1, dt * 14);
      }

      ctx.clearRect(0, 0, width, height);

      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      for (const stop of STOPS) {
        gradient.addColorStop(stop.at, `rgb(${stop.rgb[0]}, ${stop.rgb[1]}, ${stop.rgb[2]})`);
      }

      const radius = Math.min(2.5, barW * 0.22);

      const cellPath = (x: number, y: number) => {
        ctx.beginPath();
        if (typeof ctx.roundRect === "function") {
          ctx.roundRect(x, y, barW, segH, radius);
        } else {
          ctx.rect(x, y, barW, segH);
        }
      };

      // Soft glow: one blurred block per column, not per cell.
      ctx.save();
      for (let i = 0; i < count; i++) {
        const x = i * (barW + gapX);
        const barH = Math.max(segH, levels[i] * maxBarH);
        const [r, g, b] = sampleColor(i / Math.max(1, count - 1));
        ctx.shadowColor = `rgba(${r}, ${g}, ${b}, 0.55)`;
        ctx.shadowBlur = 18;
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.14)`;
        ctx.fillRect(x, baselineY - barH, barW, barH);
      }
      ctx.restore();

      // Crisp cells above the baseline.
      for (let i = 0; i < count; i++) {
        const x = i * (barW + gapX);
        const barH = Math.max(segH, levels[i] * maxBarH);
        const segs = Math.min(maxSegs, Math.max(1, Math.round(barH / step)));
        ctx.fillStyle = gradient;
        for (let s = 0; s < segs; s++) {
          const y = baselineY - (s + 1) * segH - s * segGap;
          cellPath(x, y);
          ctx.fill();
        }
      }

      // Mirrored reflection, dimming with distance from the baseline.
      for (let i = 0; i < count; i++) {
        const x = i * (barW + gapX);
        const barH = Math.max(segH, levels[i] * maxBarH);
        const segs = Math.min(maxSegs, Math.max(1, Math.round(barH / step)));
        const [r, g, b] = sampleColor(i / Math.max(1, count - 1));
        for (let s = 0; s < segs; s++) {
          const y = baselineY + s * step + segGap;
          if (y + segH > height) break;
          const fade = 0.32 * (1 - s / Math.max(1, segs)) * (1 - y / height);
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${Math.max(0, fade)})`;
          cellPath(x, y);
          ctx.fill();
        }
      }

      // Baseline hairline.
      ctx.fillStyle = gradient;
      ctx.globalAlpha = 0.22;
      ctx.fillRect(0, baselineY, width, 1);
      ctx.globalAlpha = 1;

      // Fade the tops and the deepest reflection out, so the band has no hard
      // edges against the hero.
      ctx.globalCompositeOperation = "destination-out";
      const topFade = ctx.createLinearGradient(0, 0, 0, height * 0.34);
      topFade.addColorStop(0, "rgba(0, 0, 0, 1)");
      topFade.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = topFade;
      ctx.fillRect(0, 0, width, height * 0.34);

      const bottomFade = ctx.createLinearGradient(0, height * 0.9, 0, height);
      bottomFade.addColorStop(0, "rgba(0, 0, 0, 0)");
      bottomFade.addColorStop(1, "rgba(0, 0, 0, 1)");
      ctx.fillStyle = bottomFade;
      ctx.fillRect(0, height * 0.9, width, height * 0.1);
      ctx.globalCompositeOperation = "source-over";

      if (!reduceMotion) frame = requestAnimationFrame(draw);
    };

    lastTimestamp = performance.now();
    redraw = () => {
      if (reduceMotion) draw(performance.now());
    };
    draw(lastTimestamp);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} aria-hidden="true" className="absolute inset-0 h-full w-full" />;
}
