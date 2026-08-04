"use client";

import { useCallback, useEffect, useState } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import AudioPreview from "@/components/AudioPreview";
import ManualEditor from "@/components/ManualEditor";
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
  const [showManual, setShowManual] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [history, setHistory] = useState<{ role: string; text: string }[]>([]);

  useEffect(() => () => { if (objectUrl) URL.revokeObjectURL(objectUrl); }, [objectUrl]);

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
    setHistory((prev) => [...prev, { role: "user", text: prompt }]);
    const currentPrompt = prompt;
    setPrompt("");
    setState("processing");
    try {
      const result = await processAudio(uploadResult.audio_path, currentPrompt);
      setLastResult(result);
      const msg = `Done! Applied: "${currentPrompt}"`;
      setHistory((prev) => [...prev, { role: "ai", text: msg }]);
      setState("completed");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Processing failed";
      setHistory((prev) => [...prev, { role: "ai", text: `Error: ${msg}` }]);
      setState("analyzed");
    }
  };

  const handleManualTrim = async (start: number, end: number) => {
    if (!uploadResult) return;
    setHistory((prev) => [...prev, { role: "user", text: `Trim from ${Math.round(start)}s to ${Math.round(end)}s (manual)` }]);
    setShowManual(false);
    setState("processing");
    try {
      const result = await processAudio(uploadResult.audio_path, `trim from ${start.toFixed(2)} to ${end.toFixed(2)}`);
      setLastResult(result);
      setHistory((prev) => [...prev, { role: "ai", text: `Trimmed to ${(end - start).toFixed(1)}s.` }]);
      setState("completed");
    } catch (e: unknown) {
      setHistory((prev) => [...prev, { role: "ai", text: `Error: ${e instanceof Error ? e.message : "Trim failed"}` }]);
      setState("analyzed");
    }
  };

  const handlePromptSelect = (p: string) => setPrompt(p);

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
    setShowManual(false);
    setErrorMsg("");
    setHistory([]);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    setObjectUrl(null);
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
                  <FileUpload onFileSelected={handleFileSelected} />

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
                      {["Remove the vocals", "Trim from 1:00 to 2:30", "Make this sound darker", "Extract just the drums"].map((s) => (
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
                  <button
                    onClick={() => setShowManual((v) => !v)}
                    className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:border-white/20 hover:text-white/70 transition-colors"
                  >
                    {showManual ? "AI Mode" : "Manual Mode"}
                  </button>
                  <button onClick={reset} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:border-white/20 hover:text-white/70 transition-colors">
                    New file
                  </button>
                </div>

                {objectUrl && showManual && (
                  <div className="shrink-0 border-b border-white/5 bg-white/[0.02] px-4 sm:px-6 py-3">
                    <ManualEditor src={objectUrl} onApply={handleManualTrim} />
                  </div>
                )}

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
                  </div>

                  <div className="flex-1 flex flex-col overflow-hidden min-w-0">
                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                      {history.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-center">
                          <p className="text-base text-white/30 mb-2">What do you want to do?</p>
                          <p className="text-xs text-white/20">Type a prompt below, or try Manual Mode.</p>
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
                  disabled={state === "processing"}
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
                disabled={!prompt.trim() || state === "processing"}
                className="shrink-0 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-6 py-3.5 text-sm sm:text-base font-semibold text-black transition-all duration-300 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {state === "processing" ? "..." : "Process"}
              </button>
            </div>
            <p className="mt-2 text-[11px] text-white/20">
              {file ? "Press Enter to send · mic for voice input" : "Upload a file first, then describe what you want"}
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}