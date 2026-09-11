"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Nav from "@/components/Nav";
import FileUpload from "@/components/FileUpload";
import { uploadFile, downloadUrl, type UploadResponse } from "@/services/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const BANDS = [60, 230, 910, 4000, 14000] as const;

type Gains = Record<string, number>;

function LabKnob({ label, value, min, max, step, onChange, unit = "dB" }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; unit?: string }) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="text-[10px] tracking-widest text-white/30 uppercase">{label}</div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} className="h-1.5 w-20 accent-neon-blue bg-white/10 rounded-full appearance-none cursor-pointer" />
      <div className="text-xs font-mono text-white/70">{value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1)} {unit}</div>
    </div>
  );
}

export default function LabPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [uploading, setUploading] = useState(false);

  // direct controls
  const [gains, setGains] = useState<Gains>({ "60": 0, "230": 0, "910": 0, "4000": 0, "14000": 0 });
  const [lowpass, setLowpass] = useState<number | null>(null);
  const [highpass, setHighpass] = useState<number | null>(null);
  const [pitch, setPitch] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [gainDb, setGainDb] = useState(0);
  const [reverb, setReverb] = useState(0);
  const [aiSuggest, setAiSuggest] = useState<{ preset: any; reason: string } | null>(null);
  const [suggesting, setSuggesting] = useState(false);
  const [liveAi, setLiveAi] = useState(false);
  const [evalMetrics, setEvalMetrics] = useState<any>(null);

  // audio preview (WebAudio for instant, zero-server-cost)
  const audioElRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const biquadsRef = useRef<BiquadFilterNode[]>([]);
  const [playing, setPlaying] = useState(false);
  const [tweaking, setTweaking] = useState(false);
  const [lastExport, setLastExport] = useState<string | null>(null);

  // upload
  const handleFileSelected = useCallback(async (f: File) => {
    setFile(f);
    setErrorMsg("");
    setLastExport(null);
    setAiSuggest(null);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    const url = URL.createObjectURL(f);
    setObjectUrl(url);
    setUploading(true);
    try {
      const r = await uploadFile(f);
      setUploadResult(r);
    } catch (e: any) {
      setErrorMsg(e.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }, [objectUrl]);

  useEffect(() => () => { if (objectUrl) URL.revokeObjectURL(objectUrl); }, [objectUrl]);
  useEffect(() => { fetch(`${API_BASE}/api/eval/metrics`).then(r => r.ok ? r.json() : null).then(setEvalMetrics).catch(()=>{}); }, []);

  // suggest
  const handleSuggest = async () => {
    if (!uploadResult) return;
    setSuggesting(true);
    try {
      const res = await fetch(`${API_BASE}/api/lab/suggest`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audio_path: uploadResult.audio_path }) });
      if (!res.ok) throw new Error(await res.text());
      const j = await res.json();
      const preset = j.preset;
      setAiSuggest({ preset, reason: preset.reason });
      // animate knobs like AI is tweaking
      setLiveAi(true);
      const steps = 24;
      const startGains = { ...gains };
      const target = preset.gains_db as Gains;
      let step = 0;
      const iv = setInterval(() => {
        step++;
        const t = step / steps;
        const interp: Gains = {};
        BANDS.forEach((b) => { const k = String(b); interp[k] = startGains[k] * (1 - t) + (target[k] ?? 0) * t; });
        setGains(interp);
        if (preset.reverb_mix != null) setReverb(preset.reverb_mix * t);
        if (preset.gain_db != null) setGainDb(preset.gain_db * t);
        if (step >= steps) { clearInterval(iv); setLiveAi(false); if (preset.lowpass_hz) setLowpass(preset.lowpass_hz); if (preset.highpass_hz) setHighpass(preset.highpass_hz); }
      }, 45);
    } catch (e: any) {
      setErrorMsg(e.message || "Suggest failed");
    } finally {
      setSuggesting(false);
    }
  };

  // WebAudio wiring — cheap preview, no server
  const ensureGraph = useCallback(async () => {
    const el = audioElRef.current;
    if (!el) return;
    if (!audioCtxRef.current) audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    const ctx = audioCtxRef.current!;
    if (ctx.state === "suspended") await ctx.resume();
    if (!sourceRef.current) {
      try { sourceRef.current = ctx.createMediaElementSource(el); } catch { return; }
      const biquads: BiquadFilterNode[] = [];
      // 5 peaking EQs log-spaced
      BANDS.forEach((freq) => {
        const b = ctx.createBiquadFilter();
        b.type = "peaking";
        b.frequency.value = freq;
        b.Q.value = 1.1;
        b.gain.value = 0;
        biquads.push(b);
      });
      const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 20; hp.Q.value = 0.7;
      const lp = ctx.createBiquadFilter(); lp.type = "lowpass"; lp.frequency.value = 20000; lp.Q.value = 0.7;
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
      // stash hp/lp/gain for updates
      (biquadsRef as any).hp = hp;
      (biquadsRef as any).lp = lp;
      gainNodeRef.current = gainN;
      analyserRef.current = analyser;
    }
  }, []);

  // apply knob values to WebAudio nodes (instant)
  useEffect(() => {
    const biquads = biquadsRef.current;
    if (!biquads.length) return;
    BANDS.forEach((freq, i) => { if (biquads[i]) biquads[i].gain.value = gains[String(freq)] ?? 0; });
    const hp = (biquadsRef as any).hp as BiquadFilterNode | undefined;
    const lp = (biquadsRef as any).lp as BiquadFilterNode | undefined;
    if (hp) hp.frequency.value = highpass ?? 20;
    if (lp) lp.frequency.value = lowpass ?? 20000;
    if (gainNodeRef.current) gainNodeRef.current.gain.value = Math.pow(10, gainDb / 20);
    const el = audioElRef.current;
    if (el) {
      el.playbackRate = speed;
      el.preservesPitch = false;
      // pitch via detune not widely supported on MediaElement; we show it as hint and apply server-side
    }
  }, [gains, lowpass, highpass, gainDb, speed]);

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
        // idle shimmer
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
        grad.addColorStop(0, liveAi ? "#f59e0b" : "#0affc9");
        grad.addColorStop(1, liveAi ? "#fb7185" : "#7c3aed");
        ctx.fillStyle = grad;
        ctx.fillRect(x + 1, 96 - h, barW - 2, h);
      }
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [liveAi]);

  const togglePlay = async () => {
    const el = audioElRef.current;
    if (!el) return;
    await ensureGraph();
    if (el.paused) { await el.play(); setPlaying(true); } else { el.pause(); setPlaying(false); }
  };

  const handleExport = async () => {
    if (!uploadResult) return;
    setTweaking(true);
    setErrorMsg("");
    try {
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
      setLastExport(j.download_key);
    } catch (e: any) {
      setErrorMsg(e.message || "Export failed");
    } finally {
      setTweaking(false);
    }
  };

  const resetKnobs = () => {
    setGains({ "60": 0, "230": 0, "910": 0, "4000": 0, "14000": 0 });
    setLowpass(null); setHighpass(null); setPitch(0); setSpeed(1); setGainDb(0); setReverb(0);
  };

  return (
    <div className="flex min-h-screen flex-col bg-black">
      <Nav />
      <div className="pt-14 sm:pt-16 flex flex-1 flex-col">
        {/* Header */}
        <div className="border-b border-white/5 px-4 sm:px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold text-white">Lab — Direct Controls</h1>
            <p className="text-xs sm:text-sm text-white/40">Hands-on tweaking. AI shows its work live. Instant preview — export only when you like it.</p>
          </div>
          <div className="flex gap-2">
            <a href="/prompt" className="rounded-xl border border-white/10 px-4 py-2 text-xs text-white/60 hover:border-white/20 hover:text-white transition-colors">← Prompt Mode</a>
          </div>
        </div>

        {!uploadResult ? (
          <div className="flex flex-1 flex-col items-center justify-center px-4 py-12">
            <div className="w-full max-w-xl">
              <FileUpload onFileSelected={handleFileSelected} onInvalid={setErrorMsg} />
              <p className="mt-3 text-center text-xs text-white/20">Upload any audio/video — Lab works entirely in-browser until you export.</p>
              {uploading && <p className="mt-2 text-center text-xs text-neon-blue animate-pulse">Analyzing…</p>}
              {errorMsg && <p className="mt-2 text-center text-xs text-red-400">{errorMsg}</p>}
            </div>
          </div>
        ) : (
          <>
            {/* Transport + Spectrum */}
            <div className="border-b border-white/5 bg-white/[0.02] px-4 sm:px-6 py-4">
              <div className="mx-auto max-w-6xl">
                <div className="flex items-center gap-3 mb-3">
                  <button onClick={togglePlay} className="rounded-full bg-gradient-to-r from-neon-blue to-neon-purple p-3 text-black hover:scale-105 transition-transform">
                    {playing ? <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg> : <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>}
                  </button>
                  <div className="flex-1 min-w-0">
                    <div className="truncate text-sm font-medium text-white">{file?.name}</div>
                    <div className="text-xs text-white/30">{uploadResult.analysis.duration_seconds}s · {uploadResult.analysis.bpm?.toFixed(0)} BPM · {uploadResult.analysis.key} {liveAi && <span className="ml-2 inline-flex items-center gap-1 text-amber-400"><span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse"/> AI tweaking…</span>}</div>
                  </div>
                  <button onClick={() => { setFile(null); setUploadResult(null); if (objectUrl) URL.revokeObjectURL(objectUrl); setObjectUrl(null); }} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:border-white/20">New file</button>
                </div>
                <canvas ref={canvasRef} className="w-full rounded-xl bg-black/40 border border-white/5" style={{ height: 96 }} />
                <audio ref={audioElRef} src={objectUrl || undefined} preload="metadata" crossOrigin="anonymous" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} className="hidden" />
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-[11px] text-white/20">Live WebAudio preview — no server load while you tweak</span>
                  <span className="text-[11px] text-white/20">{pitch !== 0 ? `Pitch ${pitch > 0 ? "+" : ""}${pitch} st (applied on export)` : "Pitch preview on export"}</span>
                </div>
              </div>
            </div>

            <div className="mx-auto w-full max-w-6xl flex-1 grid gap-6 px-4 sm:px-6 py-6 lg:grid-cols-[1.35fr_0.85fr]">
              {/* Direct Controls */}
              <div className="space-y-5">
                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 sm:p-5">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-sm font-semibold tracking-widest text-white/60 uppercase">5-Band EQ</h2>
                    <button onClick={resetKnobs} className="text-xs text-white/30 hover:text-white/60">Reset</button>
                  </div>
                  <div className="grid grid-cols-3 sm:grid-cols-5 gap-4">
                    {BANDS.map((f) => (
                      <LabKnob key={f} label={`${f >= 1000 ? f/1000+"k" : f} Hz`} value={gains[String(f)] ?? 0} min={-12} max={12} step={0.5} onChange={(v) => setGains((g) => ({ ...g, [String(f)]: v }))} />
                    ))}
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <label className="rounded-xl border border-white/5 bg-black/30 p-3 flex items-center justify-between">
                      <span className="text-xs text-white/50">High-pass</span>
                      <select value={highpass ?? ""} onChange={(e) => setHighpass(e.target.value ? parseFloat(e.target.value) : null)} className="bg-black border border-white/10 rounded-lg px-2 py-1 text-xs text-white">
                        <option value="">Off</option><option value="60">60 Hz</option><option value="90">90 Hz</option><option value="120">120 Hz</option><option value="200">200 Hz</option>
                      </select>
                    </label>
                    <label className="rounded-xl border border-white/5 bg-black/30 p-3 flex items-center justify-between">
                      <span className="text-xs text-white/50">Low-pass</span>
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
                  <button onClick={handleExport} disabled={tweaking} className="flex-1 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-6 py-3.5 text-sm font-semibold text-black hover:scale-[1.01] transition-transform disabled:opacity-50">
                    {tweaking ? "Rendering…" : "Apply & Export WAV"}
                  </button>
                  {lastExport && <a href={downloadUrl(lastExport)} target="_blank" rel="noopener" className="rounded-xl border border-neon-blue/30 bg-neon-blue/10 px-6 py-3.5 text-sm font-medium text-neon-blue hover:bg-neon-blue/20 transition-colors">Download</a>}
                </div>
                {lastExport && <p className="text-xs text-white/30">Exported via <code className="text-white/50">POST /api/lab/tweak</code> — lightweight STFT, no model load.</p>}
                {errorMsg && <p className="text-xs text-red-400">{errorMsg}</p>}
              </div>

              {/* AI Live Tweaking */}
              <div className="space-y-5">
                <div className="rounded-2xl border border-amber-500/20 bg-gradient-to-br from-amber-500/[0.06] to-orange-500/[0.04] p-4 sm:p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
                    <h2 className="text-sm font-semibold tracking-widest text-amber-300/80 uppercase">AI Live Tweaking</h2>
                    <span className="ml-auto text-[10px] px-2 py-1 rounded-full bg-amber-500/15 text-amber-300">Audience view</span>
                  </div>
                  <p className="text-xs leading-relaxed text-white/50 mb-4">
                    AI watches your track (mood, genre, instruments) and suggests knob positions. Hit <b className="text-white/70">Let AI tweak</b> and watch the EQ animate live — same DSP, but the reasoning is visible.
                  </p>
                  <button onClick={handleSuggest} disabled={suggesting || !uploadResult} className="w-full rounded-xl bg-amber-500/15 border border-amber-500/20 px-4 py-3 text-sm font-medium text-amber-300 hover:bg-amber-500/25 transition-colors disabled:opacity-50">
                    {suggesting ? "Thinking…" : liveAi ? "AI is tweaking…" : "✨ Let AI tweak for me"}
                  </button>
                  {aiSuggest && (
                    <div className="mt-4 rounded-xl border border-amber-500/15 bg-black/30 p-3">
                      <div className="text-xs font-medium text-amber-200/80">Why this preset?</div>
                      <div className="mt-1 text-xs leading-relaxed text-white/60">{aiSuggest.reason}</div>
                      <div className="mt-3 grid grid-cols-5 gap-1.5">
                        {BANDS.map((b) => (
                          <div key={b} className="rounded-lg bg-white/[0.04] p-1.5 text-center">
                            <div className="text-[10px] text-white/30">{b >= 1000 ? b/1000+"k" : b}</div>
                            <div className="text-xs font-mono text-amber-300">{(aiSuggest.preset.gains_db[String(b)] ?? 0) > 0 ? "+" : ""}{(aiSuggest.preset.gains_db[String(b)] ?? 0).toFixed(1)}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-4 sm:p-5">
                  <h3 className="text-xs font-semibold tracking-widest text-white/40 uppercase mb-2">How it stays light</h3>
                  <ul className="space-y-1.5 text-xs leading-relaxed text-white/40 list-disc list-inside">
                    <li>Preview runs in <b className="text-white/60">WebAudio BiquadFilterNode</b> in your browser — 0 server CPU while you drag.</li>
                    <li>Only <b className="text-white/60">Export</b> hits the server: one <code className="text-white/50">librosa.stft → gain → istft</code> pass (~200ms, no GPU/torch).</li>
                    <li>AI suggestion reuses the existing <code className="text-white/50">/api/ml/understand</code> mood/genre — no extra model.</li>
                    <li>Heavy stem separation stays lazy — Lab never loads Demucs unless you ask.</li>
                  </ul>
                </div>

                {evalMetrics && (
                  <div className="rounded-2xl border border-neon-blue/15 bg-neon-blue/[0.04] p-4 sm:p-5">
                    <h3 className="text-xs font-semibold tracking-widest text-neon-blue/70 uppercase mb-2">Model Metrics (live)</h3>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between"><span className="text-white/40">Prompt NLU accuracy</span><span className="font-mono text-white">{(evalMetrics.prompt_bench.accuracy*100).toFixed(1)}% · {evalMetrics.prompt_bench.correct}/{evalMetrics.prompt_bench.total}</span></div>
                      <div className="flex justify-between"><span className="text-white/40">Genre model</span><span className="font-mono text-white">{evalMetrics.genre.available ? `${evalMetrics.genre.model_type} · ${evalMetrics.genre.classes?.length ?? 0} classes` : "not trained"}</span></div>
                      <div className="text-[11px] text-white/30">Bench: {evalMetrics.prompt_bench.total} prompts · {(evalMetrics.prompt_bench.accuracy*100).toFixed(0)}% NLU · <code className="text-white/50">/api/eval/metrics</code> · <code className="text-white/50">docs/models/genre_classifier_card.md</code></div>
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
