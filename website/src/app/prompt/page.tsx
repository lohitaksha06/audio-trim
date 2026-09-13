"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import AudioPreview from "@/components/AudioPreview";
import {
  uploadFile,
  processAudio,
  understandAudio,
  exportZip,
  downloadUrl,
  type UploadResponse,
  type ProcessResponse,
  type UnderstandResponse,
} from "@/services/api";
import { describeResult, isOutputFollowUp } from "@/utils/followUp";

type PageState = "idle" | "uploading" | "analyzed" | "processing" | "completed" | "error";

export default function PromptPage() {
  const [state, setState] = useState<PageState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [understand, setUnderstand] = useState<UnderstandResponse | null>(null);
  const [lastResult, setLastResult] = useState<ProcessResponse | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [urlInput, setUrlInput] = useState("");
  const [urlLoading, setUrlLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [history, setHistory] = useState<{ role: string; text: string }[]>([]);
  const [promptHistory, setPromptHistory] = useState<string[]>([]);
  const [edits, setEdits] = useState(0);
  const [fromOriginal, setFromOriginal] = useState(false);
  // import-your-own-stem: second file that gets BPM-matched + beat-aligned into the song
  const [stemFile, setStemFile] = useState<File | null>(null);
  const [stemUpload, setStemUpload] = useState<UploadResponse | null>(null);
  const [stemLevel, setStemLevel] = useState(0.6);
  const [stemBusy, setStemBusy] = useState(false);
  // chaining: each edit builds on the latest output, not the raw upload
  const activeAudioPath = !fromOriginal && lastResult?.download_key
    ? lastResult.download_key
    : uploadResult?.audio_path;
  useEffect(() => {
    try { const v = localStorage.getItem("audelle:promptHistory"); if (v) setPromptHistory(JSON.parse(v)); } catch {}
  }, []);

  useEffect(() => () => { if (objectUrl) URL.revokeObjectURL(objectUrl); }, [objectUrl]);
  useEffect(() => {
    try { localStorage.setItem("audelle:promptHistory", JSON.stringify(promptHistory.slice(-30))); } catch {}
  }, [promptHistory]);

  const handleFileSelected = useCallback(async (f: File) => {
    setFile(f);
    setUrlInput("");
    setState("uploading");
    setErrorMsg("");
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    setObjectUrl(URL.createObjectURL(f));
    setUnderstand(null);
    setLastResult(null);
    try {
      const result = await uploadFile(f);
      setUploadResult(result);
      try { localStorage.setItem("audelle:lastAudio", JSON.stringify({ audio_path: result.audio_path, filename: f.name })); } catch {}
      setState("analyzed");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      setErrorMsg(msg);
      setState("error");
    }
  }, [objectUrl]);

  const handleURLUpload = async () => {
    const url = urlInput.trim();
    if (!url) return;
    setUrlLoading(true);
    setErrorMsg("");
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const name = url.split("/").pop()?.split("?")[0] || "audio.blob";
      const f = new File([blob], name, { type: blob.type || "audio/mpeg" });
      await handleFileSelected(f);
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Could not load URL");
      setState("idle");
    } finally {
      setUrlLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!uploadResult) return;
    setAnalyzing(true);
    setErrorMsg("");
    try {
      const res = await understandAudio(uploadResult.audio_path);
      setUnderstand(res);
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setAnalyzing(false);
    }
  };

  const handleProcess = async () => {
    if (!prompt.trim()) return;
    if (!uploadResult) {
      setHistory((prev) => [...prev, { role: "ai", text: "Please upload a file first before processing." }]);
      return;
    }
    // Conversational follow-up ("show me the output", "send the download"):
    // resurface the last result instead of re-processing as an unknown edit.
    if (isOutputFollowUp(prompt)) {
      const currentPrompt = prompt;
      setHistory((prev) => [...prev, { role: "user", text: currentPrompt }]);
      setPrompt("");
      if (lastResult?.download_key) {
        setHistory((prev) => [...prev, { role: "ai", text: "Here's your latest output — opening the download now. The player and Download buttons are in the Output panel too." }]);
        handleDownload(lastResult.download_key);
      } else {
        setHistory((prev) => [...prev, { role: "ai", text: "No output yet — describe an edit first, e.g. \"Add drums\"." }]);
      }
      return;
    }
    const srcPath = activeAudioPath;
    if (!srcPath) {
      setHistory((prev) => [...prev, { role: "ai", text: "Please upload a file first before processing." }]);
      return;
    }
    setHistory((prev) => [...prev, { role: "user", text: prompt }]);
    setPromptHistory((prev) => [prompt, ...prev.filter((p) => p !== prompt)].slice(0, 30));
    const currentPrompt = prompt;
    setPrompt("");
    setState("processing");
    try {
      const result = await processAudio(srcPath, currentPrompt);
      setLastResult(result);
      if (result.intent === "unknown") {
        setHistory((prev) => [...prev, { role: "ai", text: `I didn't understand "${currentPrompt}". Try: "Trim from 0:05 to 0:10" or "Remove the drums" — or see the Guide for everything I can do.` }]);
        setState("analyzed");
        return;
      }
      setEdits((n) => n + 1);
      setFromOriginal(false);
      const detail = describeResult(result.intent, result.metadata as { added_instrument?: string; combined?: string[]; groove?: string; tempo_bpm?: number; beat_count?: number; hits?: number; style?: string; enhanced?: string; boosted?: string; removed_stem?: string; isolated_stem?: string } | null);
      const msg = detail ? `Done — ${detail}. Preview it in the Output panel.` : `Done! Applied: "${currentPrompt}"`;
      setHistory((prev) => [...prev, { role: "ai", text: msg }]);
      setState("completed");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Processing failed";
      setHistory((prev) => [...prev, { role: "ai", text: `Error: ${msg}` }]);
      setState("analyzed");
    }
  };

  const handlePromptSelect = (p: string) => setPrompt(p);

  const handleStemSelected = async (f: File) => {
    setStemFile(f);
    setStemBusy(true);
    setErrorMsg("");
    try {
      const result = await uploadFile(f);
      setStemUpload(result);
      setHistory((prev) => [...prev, { role: "ai", text: `Stem "${f.name}" analyzed: ${result.analysis.bpm?.toFixed(1) ?? "?"} BPM in ${result.analysis.key ?? "?"}. Press "Mix stem in" and I'll tempo-match + beat-align it into your song.` }]);
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Stem upload failed");
    } finally {
      setStemBusy(false);
    }
  };

  const handleMixStem = async () => {
    const srcPath = activeAudioPath;
    const stemPath = stemUpload?.audio_path;
    if (!srcPath || !stemPath) {
      setHistory((prev) => [...prev, { role: "ai", text: "Upload your song AND a stem file first, then press Mix." }]);
      return;
    }
    setHistory((prev) => [...prev, { role: "user", text: `Mix my stem "${stemFile?.name ?? "stem"}" in at ${Math.round(stemLevel * 100)}%` }]);
    setState("processing");
    try {
      const result = await processAudio(srcPath, "mix my imported stem in", { stemPath, stemLevel });
      setLastResult(result);
      setEdits((n) => n + 1);
      setFromOriginal(false);
      const m = result.metadata as { song_bpm?: number; stem_bpm?: number; stretch_factor?: number; beat_offset_sec?: number } | null | undefined;
      const detail = m ? `Mixed your stem (${m.stem_bpm ?? "?"} → ${m.song_bpm ?? "?"} BPM, stretched ×${m.stretch_factor ?? 1}, offset ${m.beat_offset_sec ?? 0}s). Preview it in the Output panel.` : "Stem mixed — preview it in the Output panel.";
      setHistory((prev) => [...prev, { role: "ai", text: `Done — ${detail}` }]);
      setState("completed");
    } catch (e: unknown) {
      setHistory((prev) => [...prev, { role: "ai", text: `Error: ${e instanceof Error ? e.message : "Mix failed"}` }]);
      setState("analyzed");
    }
  };

  const handleVoice = () => {
    type SRCtor = new () => {
      start: () => void;
      lang: string;
      interimResults: boolean;
      onresult: (e: { results?: ArrayLike<ArrayLike<{ transcript?: string }>> }) => void;
      onerror: () => void;
    };
    const w = window as unknown as { SpeechRecognition?: SRCtor; webkitSpeechRecognition?: SRCtor };
    const Recognition = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!Recognition) {
      setErrorMsg("Voice input not supported in this browser.");
      return;
    }
    const rec = new Recognition();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.onresult = (e) => {
      const text = e.results?.[0]?.[0]?.transcript || "";
      setPrompt((p) => (p ? `${p} ${text}` : text));
    };
    rec.onerror = () => setErrorMsg("Voice input failed.");
    rec.start();
  };

  const handleDownload = (key?: string | null) => {
    if (!key) return;
    window.open(downloadUrl(key), "_blank", "noopener");
  };

  const handleExportZip = async () => {
    const stemsKeys = lastResult?.stems_keys;
    if (!stemsKeys || Object.keys(stemsKeys).length === 0 || !uploadResult) return;
    setExporting(true);
    try {
      const zip = await exportZip(Object.values(stemsKeys));
      window.open(downloadUrl(zip.key), "_blank", "noopener");
    } catch {
      setErrorMsg("ZIP export failed");
    } finally {
      setExporting(false);
    }
  };

  const reset = () => {
    setState("idle");
    setFile(null);
    setPrompt("");
    setUploadResult(null);
    setUnderstand(null);
    setLastResult(null);
    setErrorMsg("");
    setHistory([]);
    setEdits(0);
    setFromOriginal(false);
    setStemFile(null);
    setStemUpload(null);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    setObjectUrl(null);
  };

  const newChat = () => {
    // fresh session on the same file: clear conversation + outputs, keep upload
    setHistory([]);
    setLastResult(null);
    setPrompt("");
    setErrorMsg("");
    setEdits(0);
    setFromOriginal(false);
    if (uploadResult) setState("analyzed");
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const formatDuration = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const analysis = uploadResult?.analysis;
  const inst = understand?.instruments;
  const sections = understand?.structure?.sections ?? [];
  const mood = understand?.mood;
  const genre = understand?.genre;

  const details = analysis
    ? [
        { label: "Duration", value: formatDuration(analysis.duration_seconds) },
        { label: "BPM", value: analysis.bpm?.toFixed(1) ?? "—" },
        { label: "Key", value: analysis.key ?? "—" },
        { label: "Sample Rate", value: `${analysis.sample_rate / 1000} kHz` },
        { label: "Channels", value: analysis.channels === 1 ? "Mono" : "Stereo" },
      ]
    : [];

  const hasResult = !!lastResult;
  const stemsKeys = lastResult?.stems_keys ?? null;

  return (
    <div className="flex h-screen flex-col bg-black">
      <Nav />

      <div className="flex flex-1 mt-14 sm:mt-16 overflow-hidden">
        <Sidebar onPromptSelect={handlePromptSelect} />

        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          <div className="flex-1 flex flex-col overflow-hidden min-h-0">
            {!file ? (
              <div className="flex-1 flex flex-col items-center justify-center px-4 sm:px-8">
                <div className="w-full max-w-2xl">
                  <div className="mb-8 text-center">
                    <h1 className="mb-3 text-3xl sm:text-4xl lg:text-5xl font-bold text-white">Edit with AI</h1>
                    <p className="text-base sm:text-lg text-white/40">Upload audio or video and describe what you want to do.</p>
                  </div>
                  <FileUpload onFileSelected={handleFileSelected} onInvalid={(m) => { setErrorMsg(m); }} />

                  <div className="mt-6 rounded-2xl border border-white/5 bg-white/[0.02] p-4">
                    <p className="mb-2 text-xs text-white/30">...or import from a URL</p>
                    <div className="flex gap-2">
                      <input
                        type="url"
                        value={urlInput}
                        onChange={(e) => setUrlInput(e.target.value)}
                        placeholder="https://example.com/audio.mp3"
                        className="flex-1 rounded-xl border border-white/10 bg-white/5 px-3 py-2.5 text-sm text-white placeholder-white/20 outline-none focus:border-neon-blue/50"
                      />
                      <button
                        onClick={handleURLUpload}
                        disabled={!urlInput.trim() || urlLoading}
                        className="shrink-0 rounded-xl bg-white/10 px-4 py-2.5 text-sm text-white/70 hover:bg-white/15 disabled:opacity-40 transition-colors"
                      >
                        {urlLoading ? "Loading..." : "Import"}
                      </button>
                    </div>
                    {errorMsg && state === "idle" && <p className="mt-2 text-xs text-red-400">{errorMsg}</p>}
                  </div>

                  <div className="mt-8 text-center">
                    <p className="mb-3 text-sm text-white/30">Try saying:</p>
                    <div className="flex flex-wrap justify-center gap-2">
                      {["Remove the vocals", "Trim from 1:00 to 2:30", "Add funky drums following the groove", "Add plucky bass guitar", "Add tropical synth", "Add techno drums", "Add an 808 bass", "Add a supersaw lead", "Add edm drums and synth", "Convert to synthwave style", "Make the drums louder and the vocals quieter", "Make voices clearer", "Convert to house style"].map((s) => (
                        <button
                          key={s}
                          onClick={() => handlePromptSelect(s)}
                          className="rounded-full border border-white/10 px-3 py-1.5 text-xs sm:text-sm text-white/40 transition-colors hover:border-neon-blue/30 hover:text-neon-blue/70"
                        >
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div className="shrink-0 flex items-center gap-3 border-b border-white/5 px-4 sm:px-6 py-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-neon-blue/20 to-neon-purple/20">
                    <svg className="h-4 w-4 text-neon-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="truncate text-sm font-medium text-white">{file.name}</p>
                    <p className="text-xs text-white/40">
                      {uploadResult?.is_video ? "Video" : "Audio"} · {formatSize(file.size)}
                      {analysis && <> · {formatDuration(analysis.duration_seconds)}</>}
                    </p>
                  </div>
                  {state === "completed" && (
                    <span className="rounded-full bg-green-500/10 px-2.5 py-1 text-xs font-medium text-green-400">Done</span>
                  )}
                  <Link
                    href="/features/manual"
                    className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:border-white/20 hover:text-white/70 transition-colors"
                  >
                    Manual Mode
                  </Link>
                  <button onClick={newChat} title="Clear conversation, keep this file" className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:border-white/20 hover:text-white/70 transition-colors">
                    New chat
                  </button>
                  <button onClick={reset} title="Upload a different file" className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:border-white/20 hover:text-white/70 transition-colors">
                    New file
                  </button>
                </div>
                {edits > 0 && lastResult?.download_key && (
                  <div className="shrink-0 flex items-center gap-2 border-b border-neon-blue/10 bg-neon-blue/[0.04] px-4 sm:px-6 py-1.5 text-[11px] text-neon-blue/80">
                    <span>Layering on output {edits} — each edit builds on the last.</span>
                    {!fromOriginal ? (
                      <button onClick={() => setFromOriginal(true)} className="underline hover:text-neon-blue">Start next edit from original instead</button>
                    ) : (
                      <button onClick={() => setFromOriginal(false)} className="underline hover:text-neon-blue">Back to layering on latest</button>
                    )}
                    {fromOriginal && <span className="text-white/40">(next edit starts from your upload)</span>}
                  </div>
                )}

                <Link
                  href="/features/manual"
                  className="shrink-0 flex items-center gap-2 border-b border-neon-purple/10 bg-neon-purple/[0.04] px-4 sm:px-6 py-1.5 text-[11px] text-neon-purple/80 hover:bg-neon-purple/[0.08] transition-colors"
                >
                  <span>Want hands-on control? The Manual Editor has waveform select, trims, fades, gain & speed — it picks up your upload automatically.</span>
                  <span className="underline shrink-0">Open Manual Editor →</span>
                </Link>
                <Link
                  href="/features/mix"
                  className="shrink-0 flex items-center gap-2 border-b border-neon-blue/10 bg-neon-blue/[0.04] px-4 sm:px-6 py-1.5 text-[11px] text-neon-blue/80 hover:bg-neon-blue/[0.08] transition-colors"
                >
                  <span>Drums louder, vocals lower? Mix Lab has stem faders, wave editing and one-click mix fixes.</span>
                  <span className="underline shrink-0">Open Mix Lab →</span>
                </Link>

                <div className="flex-1 flex flex-col lg:flex-row overflow-hidden min-h-0">
                  <div className="lg:w-80 shrink-0 border-r border-white/5 overflow-y-auto p-4 space-y-4">
                    <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">Analysis</h3>

                    {objectUrl && (
                      <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
                        <p className="mb-2 text-[10px] text-white/30 uppercase tracking-wider">Input</p>
                        <AudioPreview
                          src={objectUrl}
                          curve={understand?.energy_curve?.curve}
                          height={40}
                        />
                      </div>
                    )}

                    <div className="grid grid-cols-2 gap-2">
                      {details.map((d) => (
                        <div key={d.label} className="rounded-lg border border-white/5 bg-white/[0.02] p-2.5">
                          <span className="block text-[10px] text-white/30">{d.label}</span>
                          <span className="text-sm font-medium text-white">{d.value}</span>
                        </div>
                      ))}
                    </div>

                    <div className="rounded-xl border border-neon-purple/20 bg-neon-purple/[0.04] p-3">
                      <h4 className="mb-1 text-xs font-semibold text-neon-purple/80 uppercase tracking-wider">Import your own stem</h4>
                      <p className="mb-2 text-[11px] text-white/40">AI detects both BPMs, tempo-matches + beat-aligns, then mixes.</p>
                      <label className="block w-full cursor-pointer rounded-lg bg-white/5 px-3 py-2 text-center text-xs text-white/60 hover:bg-white/10 transition-colors">
                        {stemBusy ? "Analyzing stem..." : stemFile ? stemFile.name : "Choose stem audio..."}
                        <input type="file" accept="audio/*,.mp3,.wav,.flac,.m4a,.ogg" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleStemSelected(f); }} />
                      </label>
                      {stemUpload && (
                        <p className="mt-1.5 text-[11px] text-white/40">Stem: {stemUpload.analysis.bpm?.toFixed(1) ?? "?"} BPM · {stemUpload.analysis.key ?? ""}</p>
                      )}
                      <div className="mt-2 flex items-center gap-2">
                        <span className="text-[11px] text-white/40">Level</span>
                        <input type="range" min={10} max={100} value={Math.round(stemLevel * 100)} onChange={(e) => setStemLevel(Number(e.target.value) / 100)} className="flex-1" />
                        <span className="text-[11px] text-white/60 w-9 text-right">{Math.round(stemLevel * 100)}%</span>
                      </div>
                      <button onClick={handleMixStem} disabled={!stemUpload || state === "processing"} className="mt-2 w-full rounded-lg bg-neon-purple/20 px-3 py-2 text-xs font-medium text-neon-purple hover:bg-neon-purple/30 transition-colors disabled:opacity-40">
                        Mix stem in
                      </button>
                    </div>

                    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
                      <h4 className="mb-2 text-xs font-semibold text-white/50 uppercase tracking-wider">Synths & EDM</h4>
                      <div className="flex flex-wrap gap-1.5">
                        {["Add warm synth pad", "Add tropical synth", "Add futuristic synth", "Add dubstep wobble", "Add techno drums", "Add an 808 bass", "Add a supersaw lead", "Add phonk cowbell", "Add synthwave pad", "Add trumpet", "Add choir pad", "Add edm drums and synth", "Add tropical synth and dubstep wobble", "Convert to tropical style", "Convert to techno style", "Convert to hardstyle style", "Convert to dubstep style"].map((s) => (
                          <button key={s} onClick={() => handlePromptSelect(s)} className="rounded-full border border-white/10 bg-white/5 px-2 py-1 text-[11px] text-white/50 hover:border-neon-blue/30 hover:text-neon-blue transition-colors">{s}</button>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-xl border border-neon-purple/20 bg-neon-purple/[0.04] p-3">
                      <h4 className="mb-1 text-xs font-semibold text-neon-purple/80 uppercase tracking-wider">Mix Lab</h4>
                      <p className="mb-2 text-[11px] text-white/40">Stem faders, wave editing and one-click mix fixes.</p>
                      <div className="flex flex-wrap gap-1.5">
                        {["Make the drums louder and the vocals quieter", "Prioritize drums over vocals", "Balance the mix: vocals -3dB, drums +6dB"].map((s) => (
                          <button key={s} onClick={() => handlePromptSelect(s)} className="rounded-full border border-neon-purple/20 bg-white/5 px-2 py-1 text-[11px] text-white/50 hover:border-neon-purple/40 hover:text-neon-purple transition-colors">{s}</button>
                        ))}
                      </div>
                      <Link href="/features/mix" className="mt-2 block text-center text-[11px] text-neon-purple/80 underline hover:text-neon-purple">Open Mix Lab →</Link>
                    </div>

                    {!understand ? (
                      <button
                        onClick={handleAnalyze}
                        disabled={analyzing}
                        className="w-full rounded-xl bg-neon-blue/15 px-4 py-2.5 text-sm font-medium text-neon-blue hover:bg-neon-blue/25 transition-colors disabled:opacity-50"
                      >
                        {analyzing ? "Analyzing..." : "Run AI Analysis"}
                      </button>
                    ) : (
                      <>
                        {inst && (
                          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
                            <h4 className="mb-2 text-xs font-semibold text-white/50 uppercase tracking-wider">Instruments</h4>
                            <div className="flex flex-wrap gap-1.5">
                              {inst.instruments.map((el) => (
                                <span key={el.instrument} className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/50">
                                  {el.instrument} · {(el.confidence * 100).toFixed(0)}%
                                </span>
                              ))}
                            </div>
                            <p className="mt-2 text-[11px] text-white/30">{inst.texture}</p>
                          </div>
                        )}

                        {genre?.genre && (
                          <div className="rounded-xl border border-neon-blue/20 bg-neon-blue/[0.04] p-3">
                            <h4 className="mb-2 text-xs font-semibold text-neon-blue/70 uppercase tracking-wider">Genre</h4>
                            <p className="text-sm font-medium text-white capitalize">
                              {genre.genre} <span className="text-xs text-white/30">· {(genre.confidence * 100).toFixed(0)}%</span>
                            </p>
                            {genre.suggested_actions?.length > 0 && (
                              <div className="mt-2 space-y-1">
                                {(genre.suggested_actions ?? []).slice(0, 3).map((a) => (
                                  <button
                                    key={a}
                                    onClick={() => handlePromptSelect(a)}
                                    className="block w-full rounded-lg bg-white/5 px-2 py-1 text-left text-[11px] text-neon-blue/80 hover:bg-neon-blue/10 transition-colors"
                                  >
                                    {a}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {sections.length > 0 && (
                          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
                            <h4 className="mb-2 text-xs font-semibold text-white/50 uppercase tracking-wider">Structure</h4>
                            <div className="space-y-1">
                              {sections.map((s, i) => (
                                <div key={i} className="flex items-center justify-between text-xs">
                                  <span className="text-white/60 capitalize">{s.label}</span>
                                  <span className="text-white/30">{formatDuration(s.start)}–{formatDuration(s.end)}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {mood && (
                          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
                            <h4 className="mb-1 text-xs font-semibold text-white/50 uppercase tracking-wider">Mood</h4>
                            <p className="text-sm font-medium capitalize text-white/80">{mood.mood}</p>
                            <p className="mt-1 text-[11px] text-white/30">{mood.description}</p>
                          </div>
                        )}
                      </>
                    )}

                    {hasResult && (
                      <div className="space-y-2">
                        <h4 className="text-xs font-semibold text-white/50 uppercase tracking-wider">Output</h4>
                        {lastResult?.download_key && (
                          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
                            <AudioPreview src={downloadUrl(lastResult.download_key)} height={48} />
                            <button
                              onClick={() => handleDownload(lastResult.download_key)}
                              className="mt-2 w-full rounded-lg bg-neon-blue/15 px-3 py-2 text-xs font-medium text-neon-blue hover:bg-neon-blue/25 transition-colors"
                            >
                              Download result
                            </button>
                          </div>
                        )}
                        {lastResult?.layer_download_key && (
                          <button
                            onClick={() => handleDownload(lastResult.layer_download_key)}
                            className="w-full rounded-lg bg-neon-blue/10 px-3 py-2 text-xs font-medium text-neon-blue hover:bg-neon-blue/20 transition-colors"
                          >
                            Download {lastResult.layer_label ?? "added layer"} only
                          </button>
                        )}
                        {stemsKeys && Object.keys(stemsKeys).length > 0 && (
                          <button
                            onClick={handleExportZip}
                            disabled={exporting}
                            className="w-full rounded-lg bg-neon-purple/15 px-3 py-2 text-xs font-medium text-neon-purple hover:bg-neon-purple/25 transition-colors disabled:opacity-50"
                          >
                            {exporting ? "Zipping..." : `Download stems ZIP (${Object.keys(stemsKeys).length})`}
                          </button>
                        )}
                      </div>
                    )}

                    {errorMsg && state !== "idle" && <p className="text-xs text-red-400">{errorMsg}</p>}
                    {promptHistory.length > 0 && (
                      <div className="rounded-xl border border-white/5 bg-white/[0.01] p-2">
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-[10px] text-white/30 uppercase tracking-wider">Recent prompts</span>
                          <button onClick={() => { setPromptHistory([]); localStorage.removeItem("audelle:promptHistory"); }} className="text-[10px] text-white/20 hover:text-white/40">Clear</button>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {promptHistory.slice(0, 8).map((ph) => (
                            <button key={ph} onClick={() => setPrompt(ph)} className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[11px] text-white/50 hover:border-neon-blue/30 hover:text-neon-blue truncate max-w-[150px]">{ph}</button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="flex-1 flex flex-col overflow-hidden min-w-0">
                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                      {history.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-center">
                          <p className="text-base text-white/30 mb-2">What do you want to do?</p>
                          <p className="text-xs text-white/20">Type a prompt below, or <Link href="/features/manual" className="underline hover:text-white/40">open the Manual Editor</Link>.</p>
                        </div>
                      )}
                      {history.map((h, i) => (
                        <div key={i} className={`flex ${h.role === "user" ? "justify-end" : "justify-start"}`}>
                          <div className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm ${h.role === "user" ? "bg-gradient-to-r from-neon-blue/20 to-neon-purple/20 text-white/90" : "bg-white/5 text-white/60"}`}>
                            {h.text}
                          </div>
                        </div>
                      ))}
                      {state === "processing" && (
                        <div className="flex justify-start">
                          <div className="flex items-center gap-2 rounded-xl bg-neon-blue/10 px-4 py-2.5 text-sm text-neon-blue">
                            <div className="h-3 w-3 animate-spin rounded-full border border-neon-blue border-t-transparent" />
                            Processing...
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="shrink-0 border-t border-white/5 p-3 sm:p-4 bg-black/80 backdrop-blur-sm">
            <div className="flex gap-3">
              <div className="relative flex-1">
                <input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder='e.g. "Remove the kick drum" or "Make this sound darker"'
                  className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3.5 pr-12 text-sm sm:text-base text-white placeholder-white/20 outline-none transition-all focus:border-neon-blue/50 focus:ring-2 focus:ring-neon-blue/20 focus:bg-white/[0.04]"
                  disabled={state === "processing" || state === "uploading"}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleProcess(); } }}
                  autoFocus
                />
                <button
                  onClick={handleVoice}
                  title="Voice input"
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 flex h-8 w-8 items-center justify-center rounded-lg text-white/30 hover:bg-white/5 hover:text-white/60 transition-colors"
                >
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
                  </svg>
                </button>
              </div>
              <button
                onClick={handleProcess}
                disabled={!prompt.trim() || state === "processing" || state === "uploading"}
                className="shrink-0 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-6 py-3.5 text-sm sm:text-base font-semibold text-black transition-all duration-300 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {state === "processing" ? "..." : "Process"}
              </button>
            </div>
            <p className="mt-2 text-[11px] text-white/20">
              {state === "uploading" ? "Uploading your file — hold on…" : file ? "Press Enter to send · mic for voice input" : "Upload a file first, then describe what you want"}
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}