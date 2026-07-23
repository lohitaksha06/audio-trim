"use client";

import { useState, ReactNode } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import { uploadFile, processAudio, type UploadResponse } from "@/services/api";

interface FeatureLayoutProps {
  children: ReactNode;
  title: string;
  subtitle: string;
}

export default function FeatureLayout({ children, title, subtitle }: FeatureLayoutProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [prompt, setPrompt] = useState("");
  const [state, setState] = useState<"idle" | "uploading" | "analyzed" | "processing" | "completed">("idle");
  const [history, setHistory] = useState<{ role: string; text: string }[]>([]);

  const handleFileSelected = async (f: File) => {
    setFile(f);
    setState("uploading");
    try {
      const result = await uploadFile(f);
      setUploadResult(result);
      setState("analyzed");
    } catch {
      setState("idle");
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
      const msg = `Done! Intent: ${result.intent}. Applied: "${currentPrompt}"`;
      setHistory((prev) => [...prev, { role: "ai", text: msg }]);
      setState("completed");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Processing failed";
      setHistory((prev) => [...prev, { role: "ai", text: `Error: ${msg}` }]);
      setState("analyzed");
    }
  };

  const handlePromptSelect = (p: string) => setPrompt(p);

  const reset = () => {
    setState("idle");
    setFile(null);
    setPrompt("");
    setUploadResult(null);
    setHistory([]);
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

  return (
    <div className="flex h-screen flex-col bg-black">
      <Nav />

      <div className="flex flex-1 mt-14 sm:mt-16 overflow-hidden">
        <Sidebar onPromptSelect={handlePromptSelect} />

        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          {/* Page Header */}
          <div className="shrink-0 border-b border-white/5 px-4 sm:px-6 py-3 flex items-center justify-between">
            <div>
              <h1 className="text-lg sm:text-xl font-bold text-white">{title}</h1>
              <p className="text-xs text-white/40">{subtitle}</p>
            </div>
            {file && (
              <button onClick={reset} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:border-white/20 hover:text-white/70 transition-colors">
                New file
              </button>
            )}
          </div>

          {/* Content area */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-6">
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
                  <p className="text-xs text-white/40">
                    {uploadResult?.is_video ? "Video" : "Audio"} · {formatSize(file.size)}
                    {analysis && <> · {formatDuration(analysis.duration_seconds)} · {analysis.bpm?.toFixed(0)} BPM</>}
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
                  <FileUpload onFileSelected={handleFileSelected} />
                </div>
              </div>
            ) : (
              <div>{children}</div>
            )}

            {/* Chat history */}
            {history.length > 0 && (
              <div className="mt-4 space-y-3">
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
            )}
          </div>

          {/* ALWAYS VISIBLE Prompt Input at bottom */}
          <div className="shrink-0 border-t border-white/5 p-3 sm:p-4 bg-black/80 backdrop-blur-sm">
            <div className="flex gap-3">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe what you want to do..."
                className="flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3.5 text-sm sm:text-base text-white placeholder-white/20 outline-none transition-all focus:border-neon-blue/50 focus:ring-2 focus:ring-neon-blue/20 focus:bg-white/[0.04]"
                disabled={state === "processing"}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleProcess(); } }}
                autoFocus
              />
              <button
                onClick={handleProcess}
                disabled={!prompt.trim() || state === "processing"}
                className="shrink-0 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-6 py-3.5 text-sm sm:text-base font-semibold text-black transition-all duration-300 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {state === "processing" ? "..." : "Process"}
              </button>
            </div>
            <p className="mt-2 text-[11px] text-white/20">
              {file ? "Press Enter to send" : "Upload a file first, then describe what you want"}
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}
