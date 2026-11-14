"use client";

import { useEffect, useState } from "react";

/**
 * The hero's capability line. It rotates through the things the product
 * actually does, one phrase every ten seconds. The first phrase is the one
 * people already liked, so it stays the opening frame.
 */
const PHRASES = [
  "Multitrack · mix · master",
  "Trim · fade · loop",
  "Separate · clean · restore",
  "Combine · align · polish",
  "Stems · EDL · FCPXML",
  "MIDI · automation · loudness",
  "Podcast · dialogue · noise",
  "Video audio · tone · timing",
];

export default function RotatingTagline({ intervalMs = 10_000 }: { intervalMs?: number }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => {
      setIndex((current) => (current + 1) % PHRASES.length);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);

  return (
    <span
      className="inline-flex items-baseline justify-end text-right"
      aria-live="off"
      title="Audelle capabilities"
    >
      <span key={index} className="tagline-cycle inline-block">
        {PHRASES[index]}
      </span>
    </span>
  );
}
