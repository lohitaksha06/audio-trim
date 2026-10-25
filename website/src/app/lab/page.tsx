"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Nav from "@/components/Nav";
import FileUpload from "@/components/FileUpload";
import { uploadFile, downloadUrl, isBackendConfigured, type UploadResponse } from "@/services/api";
import {
  analyzeLocal,
  decodeFile,
  runLocalPlanAsync,
  type LocalAnalysis,
  type LocalAudio,
  type LocalPlan,
} from "@/lib/audio";

// 127.0.0.1, not localhost: `localhost` resolves to ::1 first on this machine
// and the backend binds IPv4 only, so the server path fails with "Failed to
// fetch". Browser-only builds never call this.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const BANDS = [60, 230, 910, 4000, 14000] as const;
const FLAT_GAINS: Gains = { "60": 0, "230": 0, "910": 0, "4000": 0, "14000": 0 };

type Gains = Record<string, number>;

interface Preset {
  gains_db: Gains;
  lowpass_hz: number | null;
  highpass_hz: number | null;
  pitch_semitones?: number;
  speed?: number;
  gain_db: number;
  reverb_mix: number;
  reason: string;
}

interface EvalMetrics {
  prompt_bench: { accuracy: number; correct: number; total: number };
  genre: { available: boolean; model_type?: string; classes?: string[] };
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return typeof error === "string" ? error : "Something went wrong.";
}

function LabKnob({ label, value, min, max, step, onChange, unit = "dB" }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; unit?: string }) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="text-xs tracking-widest text-white/58 uppercase">{label}</div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} className="h-1.5 w-20 accent-neon-blue bg-white/10 rounded-full appearance-none cursor-pointer" />
      <div className="text-xs font-mono text-white/90">{value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1)} {unit}</div>
    </div>
  );
}

export default function LabPage() {
  const localMode = !isBackendConfigured;
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [uploading, setUploading] = useState(false);

  // direct controls
  const [gains, setGains] = useState<Gains>({ ...FLAT_GAINS });
  const [lowpass, setLowpass] = useState<number | null>(null);
  const [highpass, setHighpass] = useState<number | null>(null);
  const [pitch, setPitch] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [gainDb, setGainDb] = useState(0);
  const [reverb, setReverb] = useState(0);
  const [preset, setPreset] = useState<Preset | null>(null);
  const [suggesting, setSuggesting] = useState(false);
  const [applyingPreset, setApplyingPreset] = useState(false);
  const [evalMetrics, setEvalMetrics] = useState<EvalMetrics | null>(null);

  // browser-only state
  const [localOriginal, setLocalOriginal] = useState<LocalAudio | null>(null);
  const [localStats, setLocalStats] = useState<LocalAnalysis | null>(null);
  const [localExportUrl, setLocalExportUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ fraction: number; label: string } | null>(null);
  const [exportSummary, setExportSummary] = useState("");

  // audio preview (WebAudio for instant, zero-server-cost)
  const audioElRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const biquadsRef = useRef<BiquadFilterNode[]>([]);
  const hpRef = useRef<BiquadFilterNode | null>(null);
  const lpRef = useRef<BiquadFilterNode | null>(null);
  const animationRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [playing, setPlaying] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [serverExportKey, setServerExportKey] = useState<string | null>(null);

  const clearAnimation = () => {
    if (animationRef.current) {
      clearInterval(animationRef.current);
      animationRef.current = null;
    }
    setApplyingPreset(false);
  };

  // A MediaElementAudioSourceNode is tied to one <audio> element. The element
  // is re-created when the file changes, so the old graph must be released or
  // the knobs would keep controlling a detached element while the new one plays
  // straight to the speakers.
  const releaseGraph = useCallback(() => {
    audioElRef.current?.pause();
    try {
      sourceRef.current?.disconnect();
    } catch {
      // Already disconnected.
    }
    sourceRef.current = null;
    analyserRef.current = null;
    gainNodeRef.current = null;
    biquadsRef.current = [];
    hpRef.current = null;
    lpRef.current = null;
    setPlaying(false);
  }, []);

  // upload
  const handleFileSelected = useCallback(async (f: File) => {
    releaseGraph();
    setFile(f);
    setErrorMsg("");
    setServerExportKey(null);
    setPreset(null);
    setExportSummary("");
    setLocalStats(null);
    setLocalOriginal(null);
    clearAnimation();
    setLocalExportUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setObjectUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
    setUploading(true);
    try {
      if (localMode) {
        const decoded = await decodeFile(f);
        const stats = analyzeLocal(decoded);
        setLocalOriginal(decoded);
        setLocalStats(stats);
        setUploadResult({
          filename: f.name,
          size_bytes: f.size,
          is_video: false,
          audio_path: `local:${f.name}`,
          analysis: {
            duration_seconds: stats.durationSeconds,
            bpm: stats.bpm ?? 0,
            key: "—",
            energy: stats.rms,
            sample_rate: stats.sampleRate,
            channels: stats.channels,
          },
        });
      } else {
        const r = await uploadFile(f);
        setUploadResult(r);
      }
    } catch (e) {
      setUploadResult(null);
      setErrorMsg(errorMessage(e) || "Could not read that file.");
    } finally {
      setUploading(false);
    }
  }, [localMode, releaseGraph]);

  useEffect(() => () => {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }, [objectUrl]);
  useEffect(() => () => {
    if (localExportUrl) URL.revokeObjectURL(localExportUrl);
  }, [localExportUrl]);
  useEffect(() => () => clearAnimation(), []);

  useEffect(() => {
    if (localMode) return;
    fetch(`${API_BASE}/api/eval/metrics`).then(r => r.ok ? r.json() : null).then(setEvalMetrics).catch(() => {});
  }, [localMode]);

  const applyToGraph = useCallback(() => {
    const biquads = biquadsRef.current;
    if (biquads.length) {
      BANDS.forEach((freq, i) => {
        if (biquads[i]) biquads[i].gain.value = gains[String(freq)] ?? 0;
      });
    }
    if (hpRef.current) hpRef.current.frequency.value = highpass ?? 20;
    if (lpRef.current) lpRef.current.frequency.value = lowpass ?? 20000;
    if (gainNodeRef.current) gainNodeRef.current.gain.value = Math.pow(10, gainDb / 20);
    const el = audioElRef.current;
    if (el) {
      el.playbackRate = speed;
      // The local export uses a pitch-preserving stretcher, so the preview must
      // not turn into a chipmunk when the speed knob moves.
      el.preservesPitch = true;
      (el as HTMLAudioElement & { webkitPreservesPitch?: boolean }).webkitPreservesPitch = true;
    }
  }, [gains, lowpass, highpass, gainDb, speed]);

  // WebAudio wiring — cheap preview, no server
  const ensureGraph = useCallback(async () => {
    const el = audioElRef.current;
    if (!el) return;
    if (!audioCtxRef.current) {
      const AudioCtor =
        window.AudioContext ??
        (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtor) throw new Error("This browser does not support WebAudio.");
      audioCtxRef.current = new AudioCtor();
    }
    const ctx = audioCtxRef.current;
    if (!ctx) return;
    if (ctx.state === "suspended") await ctx.resume();
    if (!sourceRef.current) {
      try { sourceRef.current = ctx.createMediaElementSource(el); } catch { return; }
      const biquads: BiquadFilterNode[] = [];
      // 5 peaking EQs, same frequencies and Q as the offline renderer.
      BANDS.forEach((freq) => {
        const b = ctx.createBiquadFilter();
        b.type = "peaking";
        b.frequency.value = freq;
        b.Q.value = 1.1;
        b.gain.value = 0;
        biquads.push(b);
      });
      const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 20; hp.Q.value = 0.707;
      const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 20000; lp.Q.value = 0.707;
      const gainN = ctx.createGain(); gainN.gain.value = 1;
      const analyser = ctx.createAnalyser(); analyser.fftSize = 1024;
      // chain: source -> biquads -> hp -> lp -> gain -> analyser -> dest
      let last: AudioNode = sourceRef.current;
      biquads.forEach((b) => { last.connect(b); last = b; });
      last.connect(hp); last = hp;
      last.connect(lp); last = lp;
      last.connect(gainN); last = gainN;
      last.connect(analyser); analyser.connect(ctx.destination);
      biquadsRef.current = biquads;
      hpRef.current = hp;
      lpRef.current = lp;
      gainNodeRef.current = gainN;
      analyserRef.current = analyser;
      applyToGraph();
    }
  }, [applyToGraph]);

  // apply knob values to WebAudio nodes (instant)
  useEffect(() => {
    applyToGraph();
  }, [applyToGraph]);

  // spectrum animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const analyser = analyserRef.current;
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== canvas.clientWidth * dpr) { canvas.width = canvas.clientWidth * dpr; canvas.height = 96 * dpr; ctx.scale(dpr, dpr); }
      ctx.clearRect(0, 0, canvas.clientWidth, 96);
      if (!analyser) {
        ctx.fillStyle = "rgba(255,255,255,0.03)";
        ctx.fillRect(0, 0, canvas.clientWidth, 96);
        return;
      }
      const data = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(data);
      const barW = canvas.clientWidth / 64;
      for (let i = 0; i < 64; i++) {
        const v = data[Math.floor((i / 64) * data.length)] / 255;
        const h = Math.max(2, v * 88);
        const x = i * barW;
        const grad = ctx.createLinearGradient(0, 96, 0, 96 - h);
        grad.addColorStop(0, applyingPreset ? "#f59e0b" : "#0affc9");
        grad.addColorStop(1, applyingPreset ? "#fb7185" : "#7c3aed");
        ctx.fillStyle = grad;
        ctx.fillRect(x + 1, 96 - h, barW - 2, h);
      }
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [applyingPreset]);

  const togglePlay = async () => {
    const el = audioElRef.current;
    if (!el) return;
    await ensureGraph();
    if (el.paused) { await el.play(); setPlaying(true); } else { el.pause(); setPlaying(false); }
  };

  const animatePreset = (next: Preset) => {
    clearAnimation();
    setPreset(next);
    setApplyingPreset(true);
    const steps = 24;
    const startGains = { ...gains };
    let step = 0;
    animationRef.current = setInterval(() => {
      step++;
      const t = step / steps;
      const interp: Gains = {};
      BANDS.forEach((b) => {
        const k = String(b);
        interp[k] = startGains[k] * (1 - t) + (next.gains_db[k] ?? 0) * t;
      });
      setGains(interp);
      setReverb((next.reverb_mix ?? 0) * t);
      setGainDb((next.gain_db ?? 0) * t);
      if (step >= steps) {
        clearAnimation();
        setGains({ ...next.gains_db });
        setLowpass(next.lowpass_hz ?? null);
        setHighpass(next.highpass_hz ?? null);
        setPitch(next.pitch_semitones ?? 0);
        setSpeed(next.speed ?? 1);
        setReverb(next.reverb_mix ?? 0);
        setGainDb(next.gain_db ?? 0);
      }
    }, 45);
  };

  const localPreset = (stats: LocalAnalysis): Preset => {
    const bpm = stats.bpm;
    const bright = stats.brightnessHz;
    if (bpm !== null && bpm < 95) {
      return {
        gains_db: { "60": 2, "230": 1, "910": 0, "4000": -1.5, "14000": -2 },
        lowpass_hz: null,
        highpass_hz: null,
        gain_db: 0,
        reverb_mix: 0.18,
        reason: `Measured ${bpm.toFixed(0)} BPM and ${Math.round(bright)} Hz brightness. Slow track: lift the low end, soften the highs, add a little space.`,
      };
    }
    if (bright > 3200) {
      return {
        gains_db: { "60": -1, "230": 0, "910": 0, "4000": 1.5, "14000": 1 },
        lowpass_hz: null,
        highpass_hz: null,
        gain_db: 0,
        reverb_mix: 0.08,
        reason: `Measured ${bpm ? `${bpm.toFixed(0)} BPM` : "no steady beat"} and ${Math.round(bright)} Hz brightness. Bright mix: trim the sub, keep the air.`,
      };
    }
    if (bpm !== null && bpm > 130) {
      return {
        gains_db: { "60": 1.5, "230": 1, "910": 0, "4000": 1, "14000": -0.5 },
        lowpass_hz: null,
        highpass_hz: 60,
        gain_db: 1,
        reverb_mix: 0.05,
        reason: `Measured ${bpm.toFixed(0)} BPM and ${Math.round(bright)} Hz brightness. Fast track: add weight and presence without more reverb.`,
      };
    }
    return {
      gains_db: { "60": 0.5, "230": 0.5, "910": 0, "4000": 0.5, "14000": 0.5 },
      lowpass_hz: null,
      highpass_hz: 40,
      gain_db: 0,
      reverb_mix: 0.08,
      reason: `Measured ${bpm ? `${bpm.toFixed(0)} BPM` : "no steady beat"} and ${Math.round(bright)} Hz brightness. Balanced starting point: a small presence lift and a low cut.`,
    };
  };

  // preset chooser
  const handleSuggest = async () => {
    if (!uploadResult) return;
    setSuggesting(true);
    setErrorMsg("");
    try {
      if (localMode) {
        const stats = localStats ?? (localOriginal ? analyzeLocal(localOriginal) : null);
        if (!stats) throw new Error("The file is still loading.");
        animatePreset(localPreset(stats));
        return;
      }
      const res = await fetch(`${API_BASE}/api/lab/suggest`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audio_path: uploadResult.audio_path }) });
      if (!res.ok) throw new Error(await res.text());
      const j = await res.json();
      animatePreset(j.preset as Preset);
    } catch (e) {
      setErrorMsg(errorMessage(e) || "Preset failed");
    } finally {
      setSuggesting(false);
    }
  };

  const handleExport = async () => {
    if (!uploadResult) return;
    setExporting(true);
    setErrorMsg("");
    setExportSummary("");
    try {
      if (localMode) {
        if (!localOriginal) throw new Error("The file is still loading.");
        const plan: LocalPlan = {
          intent: "lab",
          params: { gains, lowpass, highpass, pitch, speed, gainDb, reverb },
          summary: "",
        };
        setProgress({ fraction: 0, label: "Starting" });
        const result = await runLocalPlanAsync(localOriginal, plan, (fraction, label) => {
          setProgress({ fraction, label });
        });
        const url = URL.createObjectURL(result.blob);
        setLocalExportUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
        setExportSummary(result.plan.summary);
        setServerExportKey(null);
        return;
      }

      const res = await fetch(`${API_BASE}/api/lab/tweak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          audio_path: uploadResult.audio_path,
          gains_db: gains,
          lowpass_hz: lowpass,
          highpass_hz: highpass,
          pitch_semitones: pitch,
          speed,
          gain_db: gainDb,
          reverb_mix: reverb,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const j = await res.json();
      setServerExportKey(j.download_key);
      setLocalExportUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
    } catch (e) {
      setErrorMsg(errorMessage(e) || "Export failed");
    } finally {
      setExporting(false);
      setProgress(null);
    }
  };

  const downloadLocal = () => {
    if (!localExportUrl) return;
    const a = document.createElement("a");
    a.href = localExportUrl;
    a.download = "audelle-lab.wav";
    a.click();
  };

  const resetKnobs = () => {
    setGains({ ...FLAT_GAINS });
    setLowpass(null); setHighpass(null); setPitch(0); setSpeed(1); setGainDb(0); setReverb(0);
  };

  const resetFile = () => {
    clearAnimation();
    releaseGraph();
    setFile(null); setUploadResult(null); setLocalOriginal(null); setLocalStats(null);
    setPreset(null); setExportSummary(""); setServerExportKey(null); setErrorMsg("");
    setObjectUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null; });
    setLocalExportUrl((prev) => { if (prev) URL.revokeObjectURL(prev); return null; });
  };

  return (
    <div className="flex min-h-screen flex-col bg-black">
      <Nav />
      <div className="pt-[var(--nav-h)] flex flex-1 flex-col">
        {/* Header */}
        <div className="border-b border-white/5 px-4 sm:px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold text-white">Lab — Direct Controls</h1>
            <p className="text-xs sm:text-sm text-white/68">
              {localMode
                ? "Measured analysis, hands-on tweaking, and local rendering. Nothing leaves this device."
                : "Measured analysis, hands-on tweaking, and server rendering. Preview is always local."}
            </p>
          </div>
          <div className="flex gap-2">
            <a href="/prompt" className="rounded-xl border border-white/10 px-4 py-2 text-xs text-white/84 hover:border-white/20 hover:text-white transition-colors">← Prompt Mode</a>
          </div>
        </div>

        {!uploadResult ? (
          <div className="flex flex-1 flex-col items-center justify-center px-4 py-12">
            <div className="w-full max-w-xl">
              <FileUpload onFileSelected={handleFileSelected} onInvalid={setErrorMsg} />
              <p className="mt-3 text-center text-xs text-white/48">
                {localMode
                  ? "Upload audio or video. Decoding, analysis and export all run in your browser."
                  : "Upload any audio/video. Lab preview is local; export is rendered by the backend."}
              </p>
              {uploading && <p className="mt-2 text-center text-xs text-neon-blue animate-pulse">Reading the file…</p>}
              {errorMsg && <p className="mt-2 text-center text-xs text-red-400">{errorMsg}</p>}
            </div>
          </div>
        ) : (
          <>
            {/* Transport + Spectrum */}
            <div className="border-b border-white/5 bg-white/[0.02] px-4 sm:px-6 py-4">
              <div className="mx-auto max-w-6xl">
                <div className="flex items-center gap-3 mb-3">
                  <button onClick={togglePlay} className="rounded-full bg-gradient-to-r from-neon-blue to-neon-purple p-3 text-black hover:scale-105 transition-transform" aria-label={playing ? "Pause" : "Play"}>
                    {playing ? <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg> : <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>}
                  </button>
                  <div className="flex-1 min-w-0">
                    <div className="truncate text-sm font-medium text-white">{file?.name}</div>
                    <div className="text-xs text-white/58">
                      {uploadResult.analysis.duration_seconds.toFixed(1)}s · {uploadResult.analysis.bpm ? `${uploadResult.analysis.bpm.toFixed(0)} BPM` : "— BPM"} · {uploadResult.analysis.key}
                      {applyingPreset && <span className="ml-2 inline-flex items-center gap-1 text-amber-400"><span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse"/>Applying preset…</span>}
                    </div>
                  </div>
                  <button onClick={resetFile} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/68 hover:border-white/20">New file</button>
                </div>
                <canvas ref={canvasRef} className="w-full rounded-xl bg-black/40 border border-white/5" style={{ height: 96 }} />
                <audio ref={audioElRef} src={objectUrl || undefined} preload="metadata" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} className="hidden" />
                <div className="mt-2 flex items-center justify-between gap-3">
                  <span className="text-xs text-white/48">Live preview: EQ, filters, gain and speed. Pitch and reverb apply on export.</span>
                  <span className="text-xs text-white/48">{pitch !== 0 ? `Pitch ${pitch > 0 ? "+" : ""}${pitch} st (export only)` : "Pitch preview on export"}</span>
                </div>
              </div>
            </div>

            <div className="mx-auto w-full max-w-6xl flex-1 grid gap-6 px-4 sm:px-6 py-6 lg:grid-cols-[1.35fr_0.85fr]">
              {/* Direct Controls */}
              <div className="space-y-5">
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 sm:p-5">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-sm font-semibold tracking-widest text-white/84 uppercase">5-Band EQ</h2>
                    <button onClick={resetKnobs} className="text-xs text-white/58 hover:text-white/84">Reset</button>
                  </div>
                  <div className="grid grid-cols-3 sm:grid-cols-5 gap-4">
                    {BANDS.map((f) => (
                      <LabKnob key={f} label={`${f >= 1000 ? f/1000+"k" : f} Hz`} value={gains[String(f)] ?? 0} min={-12} max={12} step={0.5} onChange={(v) => setGains((g) => ({ ...g, [String(f)]: v }))} />
                    ))}
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <label className="rounded-xl border border-white/5 bg-black/30 p-3 flex items-center justify-between">
                      <span className="text-xs text-white/78">High-pass</span>
                      <select value={highpass ?? ""} onChange={(e) => setHighpass(e.target.value ? parseFloat(e.target.value) : null)} className="bg-black border border-white/10 rounded-lg px-2 py-1 text-xs text-white">
                        <option value="">Off</option><option value="60">60 Hz</option><option value="90">90 Hz</option><option value="120">120 Hz</option><option value="200">200 Hz</option>
                      </select>
                    </label>
                    <label className="rounded-xl border border-white/5 bg-black/30 p-3 flex items-center justify-between">
                      <span className="text-xs text-white/78">Low-pass</span>
                      <select value={lowpass ?? ""} onChange={(e) => setLowpass(e.target.value ? parseFloat(e.target.value) : null)} className="bg-black border border-white/10 rounded-lg px-2 py-1 text-xs text-white">
                        <option value="">Off</option><option value="8000">8 kHz</option><option value="12000">12 kHz</option><option value="16000">16 kHz</option>
                      </select>
                    </label>
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 sm:p-5 grid grid-cols-2 sm:grid-cols-4 gap-6">
                  <LabKnob label="Gain" value={gainDb} min={-12} max={12} step={0.5} onChange={setGainDb} />
                  <LabKnob label="Pitch" value={pitch} min={-12} max={12} step={1} onChange={setPitch} unit="st" />
                  <LabKnob label="Speed" value={speed} min={0.5} max={2} step={0.05} onChange={setSpeed} unit="x" />
                  <LabKnob label="Reverb" value={reverb} min={0} max={1} step={0.05} onChange={setReverb} unit="" />
                </div>

                <div className="flex gap-3">
                  <button onClick={handleExport} disabled={exporting || uploading} className="flex-1 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-6 py-3.5 text-sm font-semibold text-black hover:scale-[1.01] transition-transform disabled:opacity-50">
                    {exporting ? "Rendering…" : "Apply & Export WAV"}
                  </button>
                  {localExportUrl ? (
                    <button onClick={downloadLocal} className="rounded-xl border border-neon-blue/30 bg-neon-blue/10 px-6 py-3.5 text-sm font-medium text-neon-blue hover:bg-neon-blue/20 transition-colors">Download</button>
                  ) : serverExportKey ? (
                    <a href={downloadUrl(serverExportKey)} target="_blank" rel="noopener" className="rounded-xl border border-neon-blue/30 bg-neon-blue/10 px-6 py-3.5 text-sm font-medium text-neon-blue hover:bg-neon-blue/20 transition-colors">Download</a>
                  ) : null}
                </div>
                {progress && (
                  <div className="rounded-xl border border-neon-blue/20 bg-neon-blue/[0.04] p-3">
                    <div className="flex justify-between text-xs text-neon-blue/90">
                      <span>{progress.label}</span>
                      <span>{Math.round(progress.fraction * 100)}%</span>
                    </div>
                    <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/10">
                      <div className="h-full rounded-full bg-neon-blue transition-[width]" style={{ width: `${Math.round(progress.fraction * 100)}%` }} />
                    </div>
                  </div>
                )}
                {exportSummary && <p className="text-xs text-white/58">{exportSummary}</p>}
                <p className="text-xs text-white/58">
                  {localMode
                    ? "Rendered in your browser with the local DSP — no upload, no server render."
                    : "Exported via POST /api/lab/tweak — lightweight STFT, no model load."}
                </p>
                {errorMsg && <p className="text-xs text-red-400">{errorMsg}</p>}
              </div>

              {/* Preset */}
              <div className="space-y-5">
                <div className="rounded-2xl border border-amber-500/20 bg-gradient-to-br from-amber-500/[0.06] to-orange-500/[0.04] p-4 sm:p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="h-2 w-2 rounded-full bg-amber-400" />
                    <h2 className="text-sm font-semibold tracking-widest text-amber-300/80 uppercase">Auto preset</h2>
                  </div>
                  <p className="text-xs leading-relaxed text-white/78 mb-4">
                    {localMode
                      ? "A deterministic starting point chosen from this file's measured tempo and brightness. No model is called."
                      : "A deterministic starting point chosen from the backend's measured mood and genre. No extra model is called."}
                  </p>
                  <button onClick={handleSuggest} disabled={suggesting || applyingPreset || !uploadResult} className="w-full rounded-xl bg-amber-500/15 border border-amber-500/20 px-4 py-3 text-sm font-medium text-amber-300 hover:bg-amber-500/25 transition-colors disabled:opacity-50">
                    {suggesting ? "Reading…" : applyingPreset ? "Applying…" : "Apply auto preset"}
                  </button>
                  {preset && (
                    <div className="mt-4 rounded-xl border border-amber-500/15 bg-black/30 p-3">
                      <div className="text-xs font-medium text-amber-200/80">Why this preset?</div>
                      <div className="mt-1 text-xs leading-relaxed text-white/84">{preset.reason}</div>
                      <div className="mt-3 grid grid-cols-5 gap-1.5">
                        {BANDS.map((b) => (
                          <div key={b} className="rounded-lg bg-white/[0.04] p-1.5 text-center">
                            <div className="text-xs text-white/58">{b >= 1000 ? b/1000+"k" : b}</div>
                            <div className="text-xs font-mono text-amber-300">{(preset.gains_db[String(b)] ?? 0) > 0 ? "+" : ""}{(preset.gains_db[String(b)] ?? 0).toFixed(1)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-4 sm:p-5">
                  <h3 className="text-xs font-semibold tracking-widest text-white/68 uppercase mb-2">How it stays light</h3>
                  <ul className="space-y-1.5 text-xs leading-relaxed text-white/68 list-disc list-inside">
                    {localMode ? (
                      <>
                        <li>Preview runs in <b className="text-white/84">WebAudio BiquadFilterNode</b> — no server load while you drag.</li>
                        <li>Export runs the same DSP in a <b className="text-white/84">Web Worker</b> on this device.</li>
                        <li>Nothing is uploaded. No model is loaded in browser mode.</li>
                        <li>Pitch and reverb are applied on export; the live preview covers EQ, filters, gain and speed.</li>
                      </>
                    ) : (
                      <>
                        <li>Preview runs in <b className="text-white/84">WebAudio BiquadFilterNode</b> in your browser — 0 server CPU while you drag.</li>
                        <li>Only <b className="text-white/84">Export</b> hits the server: one <code className="text-white/78">librosa.stft → gain → istft</code> pass.</li>
                        <li>Auto preset reuses the existing <code className="text-white/78">/api/ml/understand</code> mood/genre — no extra model.</li>
                        <li>Heavy stem separation stays lazy — Lab never loads Demucs unless you ask.</li>
                      </>
                    )}
                  </ul>
                </div>

                {!localMode && evalMetrics && (
                  <div className="rounded-2xl border border-neon-blue/15 bg-neon-blue/[0.04] p-4 sm:p-5">
                    <h3 className="text-xs font-semibold tracking-widest text-neon-blue/70 uppercase mb-2">Model Metrics (live)</h3>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between"><span className="text-white/68">Prompt NLU accuracy</span><span className="font-mono text-white">{(evalMetrics.prompt_bench.accuracy*100).toFixed(1)}% · {evalMetrics.prompt_bench.correct}/{evalMetrics.prompt_bench.total}</span></div>
                      <div className="flex justify-between"><span className="text-white/68">Genre model</span><span className="font-mono text-white">{evalMetrics.genre.available ? `${evalMetrics.genre.model_type} · ${evalMetrics.genre.classes?.length ?? 0} classes` : "not trained"}</span></div>
                      <div className="text-xs text-white/58">Bench: {evalMetrics.prompt_bench.total} prompts · {(evalMetrics.prompt_bench.accuracy*100).toFixed(0)}% NLU · <code className="text-white/78">/api/eval/metrics</code></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
