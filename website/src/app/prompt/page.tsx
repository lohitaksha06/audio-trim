"use client";

import { useState } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import { uploadFile, processAudio, type UploadResponse } from "@/services/api";

type PageState = "idle" | "uploading" | "analyzed" | "processing" | "completed" | "error";

export default function PromptPage() {
  const [state, setState] = useState<PageState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [history, setHistory] = useState<{ role: string; text: string }[]>([]);

  const handleFileSelected = async (f: File) => {
    setFile(f);
    setState("uploading");
    setErrorMsg("");
    try {
      const result = await uploadFile(f);
      setUploadResult(result);
      setState("analyzed");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      setErrorMsg(msg);
      setState("error");
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
    setErrorMsg("");
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
  const details = analysis
    ? [
        { label: "Duration", value: formatDuration(analysis.duration_seconds) },
        { label: "BPM", value: analysis.bpm?.toFixed(1) ?? "—" },
        { label: "Key", value: analysis.key ?? "—" },
        { label: "Sample Rate", value: `${analysis.sample_rate / 1000} kHz` },
        { label: "Channels", value: analysis.channels === 1 ? "Mono" : "Stereo" },
      ]
    : [];

  return (
    <div className="flex h-screen flex-col bg-black">
      <Nav />

      <div className="flex flex-1 mt-14 sm:mt-16 overflow-hidden">
        <Sidebar onPromptSelect={handlePromptSelect} />

        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          {/* Content area */}
          <div className="flex-1 flex flex-col overflow-hidden min-h-0">
            {!file ? (
              /* IDLE - Upload */
              <div className="flex-1 flex flex-col items-center justify-center px-4 sm:px-8">
                <div className="w-full max-w-2xl">
                  <div className="mb-8 text-center">
                    <h1 className="mb-3 text-3xl sm:text-4xl lg:text-5xl font-bold text-white">
                      Edit with AI
                    </h1>
                    <p className="text-base sm:text-lg text-white/40">
                      Upload audio or video and describe what you want to do.
                    </p>
                  </div>
                  <FileUpload onFileSelected={handleFileSelected} />

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
              /* File loaded */
              <>
                {/* Top bar */}
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
                  <button onClick={reset} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:border-white/20 hover:text-white/70 transition-colors">
                    New file
                  </button>
                </div>

                {/* Middle: Analysis + Chat */}
                <div className="flex-1 flex flex-col lg:flex-row overflow-hidden min-h-0">
                  {/* Left: Analysis */}
                  <div className="lg:w-80 shrink-0 border-r border-white/5 overflow-y-auto p-4 space-y-4">
                    <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">Analysis</h3>
                    <div className="grid grid-cols-2 gap-2">
                      {details.map((d) => (
                        <div key={d.label} className="rounded-lg border border-white/5 bg-white/[0.02] p-2.5">
                          <span className="block text-[10px] text-white/30">{d.label}</span>
                          <span className="text-sm font-medium text-white">{d.value}</span>
                        </div>
                      ))}
                    </div>
                    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
                      <h4 className="mb-2 text-xs font-semibold text-white/50 uppercase tracking-wider">Detected</h4>
                      <div className="flex flex-wrap gap-1.5">
                        {["Vocals", "Drums", "Bass", "Keys", "Guitar", "Synth"].map((el) => (
                          <span key={el} className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/50">{el}</span>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Right: Chat */}
                  <div className="flex-1 flex flex-col overflow-hidden min-w-0">
                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                      {history.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-center">
                          <p className="text-base text-white/30 mb-2">What do you want to do?</p>
                          <p className="text-xs text-white/20">Type a prompt below</p>
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

          {/* ALWAYS VISIBLE Prompt Input at bottom */}
          <div className="shrink-0 border-t border-white/5 p-3 sm:p-4 bg-black/80 backdrop-blur-sm">
            <div className="flex gap-3">
              <input
                type="text"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder='e.g. "Remove the kick drum" or "Make this sound darker"'
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
