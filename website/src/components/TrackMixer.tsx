"use client";

/**
 * A four-lane manual mixer for browser-only use.
 *
 * It is deliberately honest: these are frequency bands (bass < 180 Hz,
 * low-mid drums, 2-6 kHz presence, air), not Demucs stems. The faders give
 * immediate WebAudio preview and render through the local EQ engine.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { runLocalPlanAsync, type LocalAudio, type LocalPlan } from "@/lib/audio";
import { bandPeaks, stemGainsToEq, type BandPeaks, type StemBand } from "@/lib/audio/mix";

interface TrackMixerProps {
  src: string;
  audio: LocalAudio;
  onApply: (payload: { audio: LocalAudio; blob: Blob; summary: string }) => void;
  onBusyChange?: (busy: boolean) => void;
  disabled?: boolean;
}

const BANDS: { id: StemBand; label: string; hint: string; color: string }[] = [
  { id: "vocals", label: "Vocals", hint: "2–6 kHz presence", color: "#ef7fa6" },
  { id: "drums", label: "Drums", hint: "Body + snap", color: "#f6c445" },
  { id: "bass", label: "Bass", hint: "Below 180 Hz", color: "#6aa9ff" },
  { id: "other", label: "Other", hint: "Air / cymbals", color: "#7fd1a8" },
];

const PRESETS: { name: string; gains: Record<StemBand, number> }[] = [
  { name: "Balanced", gains: { vocals: 0, drums: 0, bass: 0, other: 0 } },
  { name: "Vocal up", gains: { vocals: 5, drums: -1, bass: 0, other: -1 } },
  { name: "Karaoke", gains: { vocals: -10, drums: 2, bass: 1, other: 2 } },
  { name: "Drums punch", gains: { vocals: -1, drums: 5, bass: 2, other: 0 } },
  { name: "Bass weight", gains: { vocals: 0, drums: 2, bass: 6, other: -1 } },
];

const FLAT: Record<StemBand, number> = { vocals: 0, drums: 0, bass: 0, other: 0 };

function fmtTime(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  const m = Math.floor(value / 60);
  const s = Math.floor(value % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function BandCanvas({ peaks, color }: { peaks: Float32Array | null; color: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 240;
    const height = canvas.clientHeight || 56;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(255,255,255,0.035)";
    ctx.fillRect(0, 0, width, height);
    if (!peaks?.length) return;
    const bar = width / peaks.length;
    ctx.fillStyle = color;
    for (let i = 0; i < peaks.length; i++) {
      const h = Math.max(1, peaks[i] * (height - 6));
      ctx.globalAlpha = 0.82;
      ctx.fillRect(i * bar, (height - h) / 2, Math.max(1, bar - 0.6), h);
    }
    ctx.globalAlpha = 1;
  }, [peaks, color]);
  return <canvas ref={ref} className="h-14 w-full rounded-md" aria-hidden="true" />;
}

export default function TrackMixer({ src, audio, onApply, onBusyChange, disabled }: TrackMixerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const filtersRef = useRef<BiquadFilterNode[]>([]);
  const masterRef = useRef<GainNode | null>(null);
  const [gains, setGains] = useState<Record<StemBand, number>>({ ...FLAT });
  const [peaks, setPeaks] = useState<BandPeaks | null>(null);
  const [peaksBusy, setPeaksBusy] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const touched = BANDS.some((band) => Math.abs(gains[band.id]) > 0.01);

  // Envelopes are computed once per audio version. One pass over the samples
  // keeps this cheap enough for a phone; it runs after paint, not during it.
  useEffect(() => {
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPeaksBusy(true);
    const timer = window.setTimeout(() => {
      const next = bandPeaks(audio);
      if (!cancelled) {
        setPeaks(next);
        setPeaksBusy(false);
      }
    }, 30);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [audio]);

  const applyGainsToGraph = useCallback(() => {
    const filters = filtersRef.current;
    if (!filters.length) return;
    const eq = stemGainsToEq(gains);
    const frequencies = [60, 230, 910, 4000, 14000];
    filters.forEach((filter, index) => {
      filter.gain.value = eq[String(frequencies[index])] ?? 0;
    });
  }, [gains]);

  useEffect(() => {
    applyGainsToGraph();
  }, [applyGainsToGraph]);

  const ensureGraph = useCallback(async () => {
    const el = audioRef.current;
    if (!el) return;
    if (!ctxRef.current) {
      const Ctor =
        window.AudioContext ??
        (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) throw new Error("This browser does not support WebAudio.");
      ctxRef.current = new Ctor();
    }
    const ctx = ctxRef.current;
    if (!ctx) return;
    if (ctx.state === "suspended") await ctx.resume();
    if (!sourceRef.current) {
      sourceRef.current = ctx.createMediaElementSource(el);
      const master = ctx.createGain();
      master.gain.value = 1;
      master.connect(ctx.destination);
      masterRef.current = master;

      // Match the offline renderer exactly: five peaking filters at the same
      // frequencies and Q as `applyEq`, with the fader gains mapped onto them.
      const bands = [60, 230, 910, 4000, 14000];
      const filters = bands.map((frequency) => {
        const filter = ctx.createBiquadFilter();
        filter.type = "peaking";
        filter.frequency.value = frequency;
        filter.Q.value = 1.1;
        filter.gain.value = 0;
        return filter;
      });
      let last: AudioNode = sourceRef.current;
      for (const filter of filters) {
        last.connect(filter);
        last = filter;
      }
      last.connect(master);
      filtersRef.current = filters;
      applyGainsToGraph();
    }
  }, [applyGainsToGraph]);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;
    const onTime = () => setCurrent(el.currentTime);
    const onMeta = () => setDuration(el.duration || 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("loadedmetadata", onMeta);
    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    return () => {
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("loadedmetadata", onMeta);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
    };
  }, [src]);

  useEffect(() => {
    return () => {
      try {
        sourceRef.current?.disconnect();
      } catch {
        /* already disconnected */
      }
      void ctxRef.current?.close().catch(() => undefined);
      ctxRef.current = null;
      sourceRef.current = null;
      filtersRef.current = [];
      masterRef.current = null;
    };
  }, []);

  const toggle = async () => {
    const el = audioRef.current;
    if (!el) return;
    try {
      await ensureGraph();
      if (el.paused) await el.play();
      else el.pause();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Playback failed");
    }
  };

  const apply = async () => {
    if (!touched) return;
    setBusy(true);
    onBusyChange?.(true);
    setError("");
    try {
      const parts = BANDS.filter((band) => Math.abs(gains[band.id]) > 0.01)
        .map((band) => `${band.label} ${gains[band.id] > 0 ? "+" : ""}${gains[band.id]} dB`)
        .join(", ");
      // Render in the DSP worker: the same five peaking filters as the live
      // preview, without blocking the sidebar on a three-minute file.
      const plan: LocalPlan = {
        intent: "lab",
        params: { gains: stemGainsToEq(gains) },
        summary: "",
      };
      const result = await runLocalPlanAsync(audio, plan);
      onApply({
        audio: result.audio,
        blob: result.blob,
        summary: `4-band mix — ${parts || "flat"}`,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Mix render failed");
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  };

  return (
    <div className="rounded-none border border-white/10 bg-white/[0.02] p-4">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-white/90">4-band mixer</h3>
          <p className="text-xs text-white/55">
            Manual faders with live preview. Render to WAV when it sounds right.
          </p>
        </div>
        <span className="border border-white/15 px-2 py-1 font-mono text-[0.6rem] uppercase tracking-[0.12em] text-white/45">
          Band mix · not AI stems
        </span>
      </div>

      <audio ref={audioRef} src={src} preload="metadata" className="hidden" />

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {BANDS.map((band) => (
          <div key={band.id} className="rounded-lg border border-white/8 bg-black/35 p-2.5">
            <div className="flex items-center justify-between gap-1">
              <span className="text-xs font-medium text-white/85">{band.label}</span>
              <span className="font-mono text-[0.65rem] text-white/55">
                {gains[band.id] > 0 ? "+" : ""}
                {gains[band.id]} dB
              </span>
            </div>
            <p className="mb-1.5 text-[0.65rem] text-white/40">{band.hint}</p>
            <BandCanvas peaks={peaks?.[band.id] ?? null} color={band.color} />
            <input
              type="range"
              min={-12}
              max={12}
              step={1}
              value={gains[band.id]}
              onChange={(e) =>
                setGains((prev) => ({ ...prev, [band.id]: Number(e.target.value) }))
              }
              className="mt-2 w-full accent-white"
              aria-label={`${band.label} level`}
            />
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {PRESETS.map((preset) => (
          <button
            key={preset.name}
            type="button"
            onClick={() => setGains({ ...preset.gains })}
            className="rounded-full border border-white/12 bg-white/[0.03] px-2.5 py-1 text-xs text-white/70 transition-colors hover:border-white/30 hover:text-white"
          >
            {preset.name}
          </button>
        ))}
        <button
          type="button"
          onClick={() => setGains({ ...FLAT })}
          className="rounded-full border border-white/12 px-2.5 py-1 text-xs text-white/50 transition-colors hover:text-white/80"
        >
          Flat
        </button>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void toggle()}
          className="rounded-lg border border-white/20 bg-white/[0.06] px-3 py-1.5 text-xs font-medium text-white/85 transition-colors hover:border-white/45"
        >
          {playing ? "Pause" : "Preview mix"}
        </button>
        <span className="font-mono text-[0.7rem] text-white/45">
          {fmtTime(current)} / {fmtTime(duration)}
        </span>
        {peaksBusy && <span className="text-[0.7rem] text-white/40">drawing bands…</span>}
        <span className="flex-1" />
        <button
          type="button"
          onClick={() => void apply()}
          disabled={!touched || busy || disabled}
          className="rounded-lg bg-[#f2f1ec] px-4 py-1.5 text-xs font-semibold text-black transition-colors hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "Rendering…" : "Apply mix →"}
        </button>
      </div>

      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}
