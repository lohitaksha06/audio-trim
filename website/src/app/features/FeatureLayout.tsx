"use client";

import { useState, useEffect, useRef, ReactNode } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import AudioPreview from "@/components/AudioPreview";
import { uploadFile, processAudio, fetchSuggestions, understandAudio, exportZip, downloadUrl, isBackendConfigured, type UploadResponse, type UnderstandResponse, type ProcessResponse } from "@/services/api";
import {
  analyze as analyzeLocal,
  decodeFile,
  durationOf,
  parseLocalPrompt,
  runLocalPlanAsync,
  type LocalAudio,
} from "@/lib/audio";
import { describeResult, isOutputFollowUp, type ResultMeta } from "@/utils/followUp";
import { getSettings } from "@/utils/settings";
import { FeaturePromptContext } from "./FeaturePromptContext";
import { Conversation } from "./FeaturePanels";

interface FeatureLayoutProps {
  children: ReactNode;
  title: string;
  subtitle: string;
}

/** Backend result, plus the object URL a browser-only edit produces. */
type ChatResult = ProcessResponse & { local_url?: string | null };

export default function FeatureLayout({ children, title, subtitle }: FeatureLayoutProps) {
  // No NEXT_PUBLIC_API_URL means this build has no backend. Decode and run the
  // DSP in the browser instead of failing with "Failed to fetch".
  const localMode = !isBackendConfigured;
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [prompt, setPrompt] = useState("");
  const [state, setState] = useState<"idle" | "uploading" | "analyzed" | "processing" | "completed">("idle");
  const [history, setHistory] = useState<{ role: string; text: string }[]>([]);
  const [understand, setUnderstand] = useState<UnderstandResponse | null>(null);
  const [lastResult, setLastResult] = useState<ChatResult | null>(null);
  const [localAudio, setLocalAudio] = useState<LocalAudio | null>(null);
  const [localOriginal, setLocalOriginal] = useState<LocalAudio | null>(null);
  const [localUrl, setLocalUrl] = useState<string | null>(null);
  const [localProgress, setLocalProgress] = useState<{ fraction: number; label: string } | null>(null);
  const processingRef = useRef(false);
  const [exporting, setExporting] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [notice, setNotice] = useState("");
  const [edits, setEdits] = useState(0);
  const [fromOriginal, setFromOriginal] = useState(false);
  const activeAudioPath = !fromOriginal && lastResult?.download_key
    ? lastResult.download_key
    : uploadResult?.audio_path;

  // Revoke the previous object URL when it is replaced or the page unmounts.
  useEffect(() => () => { if (localUrl) URL.revokeObjectURL(localUrl); }, [localUrl]);

  const handleFileSelected = async (f: File) => {
    setFile(f);
    setState("uploading");
    setErrorMsg("");
    setNotice("");
    setUnderstand(null);
    setLastResult(null);
    setLocalUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setLocalAudio(null);
    setLocalOriginal(null);

    if (localMode) {
      try {
        const decoded = await decodeFile(f);
        const info = analyzeLocal(decoded);
        setLocalAudio(decoded);
        setLocalOriginal(decoded);
        setUploadResult({
          filename: f.name,
          size_bytes: f.size,
          is_video: false,
          audio_path: `local:${f.name}`,
          analysis: {
            duration_seconds: info.durationSeconds,
            bpm: 0,
            key: "—",
            energy: info.rms,
            sample_rate: info.sampleRate,
            channels: info.channels,
          },
        });
        setState("analyzed");
      } catch {
        setFile(null);
        setErrorMsg(
          "This browser could not decode that file. WAV, MP3, M4A, OGG and FLAC all work in Chrome, Edge and Safari.",
        );
        setState("idle");
      }
      return;
    }

    try {
      const result = await uploadFile(f);
      setUploadResult(result);
      setState("analyzed");
      if (getSettings().autoAnalyze) void handleAnalyze(result.audio_path);
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Upload failed");
      setState("idle");
    }
  };

  const handleAnalyze = async (audioPath?: string) => {
    if (!isBackendConfigured) return;
    const target = audioPath ?? uploadResult?.audio_path;
    if (!target) return;
    setErrorMsg("");
    try {
      const res = await understandAudio(target);
      setUnderstand(res);
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Analysis failed");
    }
  };

  const downloadLocal = () => {
    if (!lastResult?.local_url) return;
    const a = document.createElement("a");
    a.href = lastResult.local_url;
    a.download = `audelle-${lastResult.intent}.wav`;
    a.click();
  };

  const handleDownload = (key?: string | null) => {
    if (lastResult?.local_url) {
      downloadLocal();
      return;
    }
    if (!key) return;
    window.open(downloadUrl(key), "_blank", "noopener");
  };

  const handleProcess = async (override?: string) => {
    const raw = (override ?? prompt).trim();
    if (!raw) return;
    if (!uploadResult) {
      // The chat log renders in both states here, so a reply is enough — no
      // duplicate notice above the composer.
      setHistory((prev) => [...prev, { role: "ai", text: "Please upload a file first before processing." }]);
      return;
    }
    setNotice("");
    if (isOutputFollowUp(raw)) {
      setHistory((prev) => [...prev, { role: "user", text: raw }]);
      setPrompt("");
      if (lastResult?.download_key || lastResult?.local_url) {
        setHistory((prev) => [...prev, { role: "ai", text: "Here's your latest output — opening the download now." }]);
        handleDownload(lastResult.download_key);
      } else {
        setHistory((prev) => [...prev, { role: "ai", text: "No output yet — describe an edit first, e.g. \"Add drums\"." }]);
      }
      return;
    }

    if (localMode) {
      // Browser-only edit. Nothing leaves the machine.
      if (processingRef.current) return;
      const base = (!fromOriginal && localAudio) || localOriginal;
      setHistory((prev) => [...prev, { role: "user", text: raw }]);
      const currentPrompt = raw;
      setPrompt("");
      if (!base) {
        setNotice("Decode the file again, then send your prompt.");
        return;
      }
      const plan = parseLocalPrompt(currentPrompt, durationOf(base));
      if (plan.unsupported || plan.intent === "unknown") {
        setHistory((prev) => [
          ...prev,
          { role: "ai", text: plan.unsupported ?? "I couldn't map that to a local edit." },
        ]);
        return;
      }
      setState("processing");
      setErrorMsg("");
      processingRef.current = true;
      setLocalProgress({ fraction: 0, label: "Starting" });
      try {
        const result = await runLocalPlanAsync(base, plan, (fraction, label) => {
          setLocalProgress({ fraction, label });
        });
        const url = URL.createObjectURL(result.blob);
        setLocalUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
        setLocalAudio(result.audio);
        setLastResult({
          intent: plan.intent,
          params: plan.params,
          raw_prompt: currentPrompt,
          metadata: result.metadata,
          download_key: null,
          local_url: url,
        });
        setEdits((n) => n + 1);
        setFromOriginal(false);
        setHistory((prev) => [...prev, { role: "ai", text: `Done — ${result.plan.summary}.` }]);
        setState("completed");
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Processing failed";
        setHistory((prev) => [...prev, { role: "ai", text: `Error: ${msg}` }]);
        setState("analyzed");
      } finally {
        processingRef.current = false;
        setLocalProgress(null);
      }
      return;
    }

    const srcPath = activeAudioPath;
    if (!srcPath) {
      setHistory((prev) => [...prev, { role: "ai", text: "Please upload a file first before processing." }]);
      return;
    }
    setHistory((prev) => [...prev, { role: "user", text: raw }]);
    const currentPrompt = raw;
    setPrompt("");
    setState("processing");
    setErrorMsg("");
    try {
      const result = await processAudio(srcPath, currentPrompt);
      setLastResult(result);
      if (result.intent === "unknown") {
        const ideas = await fetchSuggestions(currentPrompt);
        const list = ideas.length
          ? ideas.map((t) => `"${t}"`).join(", ")
          : '"Trim from 0:05 to 0:10" or "Remove the drums"';
        setHistory((prev) => [...prev, { role: "ai", text: `I couldn't turn "${currentPrompt}" into a specific edit yet. Things I can do near that: ${list}.` }]);
        setState("analyzed");
        return;
      }
      setEdits((n) => n + 1);
      setFromOriginal(false);
      const detail = describeResult(result.intent, result.metadata as ResultMeta | null);
      const msg = detail ? `Done — ${detail}.` : `Done! Intent: ${result.intent}. Applied: "${currentPrompt}"`;
      setHistory((prev) => [...prev, { role: "ai", text: msg }]);
      setState("completed");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Processing failed";
      const asksBpm = /what BPM/i.test(msg);
      setHistory((prev) => [...prev, { role: "ai", text: asksBpm ? msg : `Error: ${msg}` }]);
      setErrorMsg(msg);
      setState("analyzed");
    }
  };

  const handlePromptSelect = (p: string) => setPrompt(p);

  const handleExportZip = async () => {
    const stemsKeys = lastResult?.stems_keys;
    if (!stemsKeys || Object.keys(stemsKeys).length === 0) return;
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
    setLocalAudio(null);
    setLocalOriginal(null);
    setLocalUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setLocalProgress(null);
    setHistory([]);
    setErrorMsg("");
    setNotice("");
    setEdits(0);
    setFromOriginal(false);
  };

  const newChat = () => {
    setHistory([]);
    setLastResult(null);
    setLocalAudio(localOriginal);
    setLocalUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setPrompt("");
    setErrorMsg("");
    setNotice("");
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
  const hasResult = !!lastResult;

  const ctx = {
    setPrompt: handlePromptSelect,
    triggerPrompt: (p: string) => void handleProcess(p),
    hasFile: !!file,
    analysis,
    understand,
    busy: state === "processing" || state === "uploading" || exporting,
    hasResult,
    fileName: file?.name ?? null,
    edits,
    lastMetadata: (lastResult?.metadata ?? null) as Record<string, unknown> | null,
  };

  return (
    <FeaturePromptContext.Provider value={ctx}>
    <div className="flex h-screen flex-col bg-black">
      <Nav />

      <div className="flex flex-1 mt-[var(--nav-h)] overflow-hidden">
        <Sidebar onPromptSelect={handlePromptSelect} />

        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          {/* Page Header */}
          <div className="shrink-0 border-b border-white/5 px-4 sm:px-6 py-3 flex items-center justify-between">
            <div>
              <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white">{title}</h1>
              <p className="text-[0.9375rem] text-white/72">{subtitle}</p>
            </div>
            {file && (
              <div className="flex gap-2">
                <button onClick={newChat} title="Clear conversation, keep this file" className="rounded-lg border border-white/10 px-3 py-1.5 text-[0.8125rem] text-white/68 hover:border-white/20 hover:text-white/90 transition-colors">
                  New chat
                </button>
                <button onClick={reset} className="rounded-lg border border-white/10 px-3 py-1.5 text-[0.8125rem] text-white/68 hover:border-white/20 hover:text-white/90 transition-colors">
                  New file
                </button>
              </div>
            )}
            {edits > 0 && (lastResult?.download_key || lastResult?.local_url) && (
              <div className="shrink-0 text-xs text-neon-blue/70 px-1">
                Layering on output {edits}
                {!fromOriginal ? (
                  <button onClick={() => setFromOriginal(true)} className="underline ml-1">from original instead</button>
                ) : (
                  <button onClick={() => setFromOriginal(false)} className="underline ml-1">back to latest</button>
                )}
              </div>
            )}
          </div>

          {/* Content area */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-6">
            {localMode && (
              <p className="mb-3 rounded-lg border border-neon-blue/20 bg-neon-blue/[0.05] px-3 py-2 text-xs text-neon-blue/90">
                Browser mode: the file is decoded on this device and every edit runs here. Nothing is uploaded.
              </p>
            )}
            {/* File info bar when file loaded */}
            {file && (
              <div className="flex items-center gap-3 mb-4 pb-4 border-b border-white/5">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-neon-blue/20 to-neon-purple/20">
                  <svg className="h-4 w-4 text-neon-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="truncate text-sm font-medium text-white">{file.name}</p>
                  <p className="text-xs text-white/68">
                    {uploadResult?.is_video ? "Video" : "Audio"} · {formatSize(file.size)}
                    {analysis && <> · {formatDuration(analysis.duration_seconds)} · {analysis.bpm ? `${analysis.bpm.toFixed(0)} BPM` : "— BPM"}</>}
                  </p>
                </div>
                {state === "completed" && (
                  <span className="rounded-full bg-green-500/10 px-2.5 py-1 text-xs font-medium text-green-400">Done</span>
                )}
              </div>
            )}

            {/* Upload zone or feature content */}
            {!file ? (
              <div className="flex flex-col items-center justify-center py-12">
                <div className="w-full max-w-xl">
                  <FileUpload onFileSelected={handleFileSelected} onInvalid={setErrorMsg} />
                </div>
                {errorMsg && <p className="mt-3 text-sm text-red-400">{errorMsg}</p>}
              </div>
            ) : (
              <>
                {/* Analysis panel (shared) */}
                <div className="mb-4 grid gap-3">
                  <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 grid grid-cols-3 gap-2">
                    <div className="rounded-lg border border-white/5 bg-black/40 p-2.5 text-center"><div className="text-[0.8125rem] text-white/58">Duration</div><div className="mt-0.5 text-[0.9375rem] text-white">{analysis ? formatDuration(analysis.duration_seconds) : "—"}</div></div>
                    <div className="rounded-lg border border-white/5 bg-black/40 p-2.5 text-center"><div className="text-[0.8125rem] text-white/58">BPM</div><div className="mt-0.5 text-[0.9375rem] text-white">{analysis?.bpm ? analysis.bpm.toFixed(0) : "—"}</div></div>
                    <div className="rounded-lg border border-white/5 bg-black/40 p-2.5 text-center"><div className="text-[0.8125rem] text-white/58">Key</div><div className="mt-0.5 text-[0.9375rem] text-white">{analysis?.key ?? "—"}</div></div>
                  </div>
                  {localProgress && (
                    <div className="rounded-xl border border-neon-blue/20 bg-neon-blue/[0.04] p-3">
                      <div className="flex justify-between text-xs text-neon-blue/90">
                        <span>{localProgress.label}</span>
                        <span>{Math.round(localProgress.fraction * 100)}%</span>
                      </div>
                      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/10">
                        <div className="h-full rounded-full bg-neon-blue transition-[width]" style={{ width: `${Math.round(localProgress.fraction * 100)}%` }} />
                      </div>
                    </div>
                  )}
                  {understand && (
                    <div className="grid gap-2">
                      {inst && <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3"><div className="text-xs text-white/68 uppercase tracking-wider mb-1">Instruments</div><div className="flex flex-wrap gap-1.5">{inst.instruments.map((el) => <span key={el.instrument} className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/78">{el.instrument} · {(el.confidence*100).toFixed(0)}%</span>)}</div></div>}
                      {understand?.genre && <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3"><div className="text-xs text-white/68 uppercase tracking-wider mb-1">Genre / Style</div><div className="text-[0.875rem] text-white/84">{understand.genre.genre} · {(understand.genre.confidence*100).toFixed(0)}%{understand?.rhythm ? ` · ${understand.rhythm.auto_groove.replace(/_/g, " ")} · ${Math.round(understand.rhythm.tempo_bpm)} BPM` : ""}</div></div>}
                      {sections.length>0 && <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3"><div className="eyebrow mb-1.5">Structure</div>{sections.map((s,i)=><div key={i} className="flex justify-between text-[0.875rem]"><span className="text-white/84 capitalize">{s.label}</span><span className="text-white/58">{formatDuration(s.start)}–{formatDuration(s.end)}</span></div>)}</div>}
                    </div>
                  )}
                  {hasResult && (lastResult?.download_key || lastResult?.local_url) && (
                    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
                      <AudioPreview src={lastResult.local_url ?? downloadUrl(lastResult.download_key!)} height={48} />
                      {lastResult.layer_download_key && (
                        <button onClick={()=>handleDownload(lastResult.layer_download_key)} className="mt-2 w-full rounded-lg bg-neon-blue/10 px-3 py-2.5 text-[0.875rem] font-medium text-neon-blue hover:bg-neon-blue/20 transition-colors">Download {lastResult.layer_label ?? "added layer"} only</button>
                      )}
                      <div className="mt-2 flex gap-2">
                        <button onClick={()=>handleDownload(lastResult.download_key)} className="flex-1 rounded-lg bg-neon-blue/15 px-3 py-2.5 text-[0.875rem] font-medium text-neon-blue hover:bg-neon-blue/25 transition-colors">Download result</button>
                        {lastResult.stems_keys && Object.keys(lastResult.stems_keys).length>0 && (
                          <button onClick={handleExportZip} disabled={exporting} className="flex-1 rounded-lg bg-neon-purple/15 px-3 py-2.5 text-[0.875rem] font-medium text-neon-purple hover:bg-neon-purple/25 disabled:opacity-50 transition-colors">{exporting?"Zipping...":`Stems ZIP (${Object.keys(lastResult.stems_keys).length})`}</button>
                        )}
                      </div>
                    </div>
                  )}
                  {errorMsg && <p className="text-xs text-red-400">{errorMsg}</p>}
                </div>
                <Conversation history={history} busy={state === "processing"} />
                <div>{children}</div>
              </>
            )}
            {/* Same conversation when no file is loaded, so replies are never
                hidden below the action list. */}
            {!file && <Conversation history={history} busy={state === "processing"} />}
          </div>

          {/* ALWAYS VISIBLE Prompt Input at bottom */}
          <div className="shrink-0 border-t border-white/10 bg-black/90 p-3 sm:p-4 backdrop-blur-sm">
            {notice && (
              <p role="alert" className="mb-2 flex items-start gap-2 rounded-lg border border-amber-300/30 bg-amber-300/10 px-3 py-2 text-sm text-amber-100">
                <span aria-hidden className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-300" />
                {notice}
              </p>
            )}
            <p className="mb-2 text-[0.8125rem] text-white/58">
              {file ? "Press Enter to send · mic for voice input" : "Upload a file first, then describe what you want"}
            </p>
            <div className="flex flex-col gap-2.5 sm:flex-row sm:gap-3">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe what you want to do..."
                className="w-full flex-1 rounded-xl border border-white/15 bg-white/[0.06] px-4 py-3.5 text-base text-white placeholder-white/48 outline-none transition-colors focus:border-neon-blue/60"
                disabled={state === "processing"}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleProcess(); } }}
                autoFocus
              />
              <button
                onClick={() => void handleProcess()}
                disabled={!prompt.trim() || state === "processing"}
                className="shrink-0 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-6 py-3.5 text-base font-semibold text-black transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {state === "processing" ? "..." : "Process"}
              </button>
            </div>
          </div>
        </main>
      </div>
    </div>
    </FeaturePromptContext.Provider>
  );
}
