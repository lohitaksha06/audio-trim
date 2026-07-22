"use client";

import { useState } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import { uploadFile, type UploadResponse } from "@/services/api";

type PageState = "idle" | "uploading" | "analyzed" | "processing" | "completed" | "error";

export default function PromptPage() {
  const [state, setState] = useState<PageState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [history, setHistory] = useState<{ role: string; text: string }[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const isVideo = (f: File) => f.type.startsWith("video/");

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

  const handleProcess = () => {
    if (!prompt.trim()) return;
    setHistory((prev) => [...prev, { role: "user", text: prompt }]);
    setState("processing");
    setTimeout(() => {
      setHistory((prev) => [...prev, { role: "ai", text: `Done! Applied: "${prompt}" to your audio.` }]);
      setState("completed");
    }, 2000);
  };

  const handlePromptSelect = (p: string) => {
    setPrompt(p);
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

  const reset = () => {
    setState("idle");
    setFile(null);
    setPrompt("");
    setUploadResult(null);
    setErrorMsg("");
    setHistory([]);
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
    <div className="flex h-screen flex-col bg-black overflow-hidden">
      <Nav />

      <div className="flex flex-1 mt-14 sm:mt-16 overflow-hidden">
        {/* Sidebar */}
        <div className={`${sidebarOpen ? "block" : "hidden"} md:block`}>
          <Sidebar onPromptSelect={handlePromptSelect} />
        </div>

        {/* Sidebar toggle */}
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="hidden md:flex fixed left-0 top-20 z-40 items-center justify-center w-6 h-12 bg-white/5 border border-white/10 border-l-0 rounded-r-lg hover:bg-white/10 transition-colors"
          style={{ left: sidebarOpen ? "320px" : "0" }}
        >
          <svg
            className={`h-4 w-4 text-white/40 transition-transform ${sidebarOpen ? "rotate-180" : ""}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
        </button>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto">
          {/* IDLE */}
          {state === "idle" && (
            <div className="flex flex-col items-center justify-center h-full px-4 sm:px-8">
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

                {/* Quick suggestions */}
                <div className="mt-8 text-center">
                  <p className="mb-3 text-sm text-white/30">Try saying:</p>
                  <div className="flex flex-wrap justify-center gap-2">
                    {[
                      "Remove the vocals",
                      "Trim from 1:00 to 2:30",
                      "Make this sound darker",
                      "Extract just the drums",
                    ].map((s) => (
                      <button
                        key={s}
                        onClick={() => { setPrompt(s); handlePromptSelect(s); }}
                        className="rounded-full border border-white/10 px-3 py-1.5 text-xs sm:text-sm text-white/40 transition-colors hover:border-neon-blue/30 hover:text-neon-blue/70"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* UPLOADING */}
          {state === "uploading" && (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="mb-6 h-12 w-12 animate-spin rounded-full border-2 border-neon-blue border-t-transparent" />
              <p className="text-lg text-white/60">Analyzing your file...</p>
              <p className="mt-1 text-sm text-white/30">Detecting instruments, structure, and energy</p>
            </div>
          )}

          {/* ANALYZED / PROCESSING / COMPLETED */}
          {(state === "analyzed" || state === "processing" || state === "completed") && file && (
            <div className="h-full flex flex-col">
              {/* File bar */}
              <div className="flex items-center gap-4 border-b border-white/5 px-4 sm:px-6 py-3 sm:py-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-neon-blue/20 to-neon-purple/20">
                  <span className="text-xl">{uploadResult?.is_video ? "🎬" : "🎵"}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="truncate text-base sm:text-lg font-medium text-white">{file.name}</p>
                  <p className="text-xs sm:text-sm text-white/40">
                    {uploadResult?.is_video ? "Video (audio extracted)" : "Audio"} · {formatSize(file.size)}
                  </p>
                </div>
                {state === "completed" && (
                  <span className="rounded-full bg-green-500/10 px-3 py-1 text-xs sm:text-sm font-medium text-green-400">
                    Done
                  </span>
                )}
                <button
                  onClick={reset}
                  className="rounded-lg border border-white/10 px-3 py-1.5 text-xs sm:text-sm text-white/40 transition-colors hover:border-white/20 hover:text-white/70"
                >
                  New file
                </button>
              </div>

              {/* Analysis + Prompt split */}
              <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
                {/* Left: Analysis */}
                <div className="lg:w-1/3 border-r border-white/5 overflow-y-auto p-4 sm:p-6 space-y-4 sm:space-y-5">
                  <h3 className="text-xs sm:text-sm font-semibold text-white/50 uppercase tracking-wider">Analysis</h3>
                  <div className="grid grid-cols-2 gap-2">
                    {details.map((d) => (
                      <div key={d.label} className="rounded-lg border border-white/5 bg-white/[0.02] p-2.5 sm:p-3">
                        <span className="block text-[10px] sm:text-xs text-white/30">{d.label}</span>
                        <span className="text-sm sm:text-base font-medium text-white">{d.value}</span>
                      </div>
                    ))}
                  </div>

                  <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
                    <h4 className="mb-2 text-xs sm:text-sm font-semibold text-white/50 uppercase tracking-wider">Detected</h4>
                    <div className="flex flex-wrap gap-1.5">
                      {["🎤 Vocals", "🥁 Drums", "🎸 Bass", "🎹 Keys", "🎸 Guitar", "🎛️ Synth"].map((el) => (
                        <span key={el} className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs sm:text-sm text-white/60">
                          {el}
                        </span>
                      ))}
                    </div>
                    <p className="mt-2 text-[10px] sm:text-xs text-white/30 italic">
                      Click a feature in the sidebar to auto-fill a prompt.
                    </p>
                  </div>
                </div>

                {/* Right: Prompt + Chat */}
                <div className="flex-1 flex flex-col overflow-hidden p-4 sm:p-6">
                  {/* Chat history */}
                  <div className="flex-1 overflow-y-auto space-y-3 mb-4">
                    {history.length === 0 && (
                      <div className="flex flex-col items-center justify-center h-full text-center">
                        <p className="text-base sm:text-lg text-white/30 mb-2">What do you want to do?</p>
                        <p className="text-xs sm:text-sm text-white/20">Use the sidebar or type a prompt below</p>
                      </div>
                    )}
                    {history.map((h, i) => (
                      <div
                        key={i}
                        className={`flex ${h.role === "user" ? "justify-end" : "justify-start"}`}
                      >
                        <div
                          className={`max-w-[80%] rounded-xl px-4 py-2.5 text-sm sm:text-base ${
                            h.role === "user"
                              ? "bg-gradient-to-r from-neon-blue/20 to-neon-purple/20 text-white/90"
                              : "bg-white/5 text-white/60"
                          }`}
                        >
                          {h.text}
                        </div>
                      </div>
                    ))}
                    {state === "processing" && (
                      <div className="flex justify-start">
                        <div className="flex items-center gap-2 rounded-xl bg-neon-blue/10 px-4 py-2.5 text-sm sm:text-base text-neon-blue">
                          <div className="h-3 w-3 animate-spin rounded-full border border-neon-blue border-t-transparent" />
                          Processing...
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Prompt input */}
                  <div className="shrink-0 rounded-xl border border-white/10 bg-white/[0.02] p-3 sm:p-4">
                    <div className="flex gap-3">
                      <input
                        type="text"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder='e.g. "Remove the kick drum" or "Make this sound darker"'
                        className="flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm sm:text-base text-white placeholder-white/20 outline-none transition-colors focus:border-neon-blue/50 focus:ring-1 focus:ring-neon-blue/20"
                        disabled={state === "processing"}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            handleProcess();
                          }
                        }}
                      />
                      <button
                        onClick={handleProcess}
                        disabled={!prompt.trim() || state === "processing"}
                        className="shrink-0 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-5 sm:px-6 py-3 text-sm sm:text-base font-semibold text-black transition-all duration-300 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-40"
                      >
                        {state === "processing" ? "..." : "Process"}
                      </button>
                    </div>
                    <p className="mt-2 text-[10px] sm:text-xs text-white/20">
                      Press Enter to send · All features are available in the sidebar →
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ERROR */}
          {state === "error" && (
            <div className="flex flex-col items-center justify-center h-full">
              <span className="mb-4 text-5xl">⚠️</span>
              <p className="mb-2 text-xl font-medium text-white">Something went wrong</p>
              <p className="mb-6 max-w-md text-center text-sm sm:text-base text-white/40">
                {errorMsg || "Please try again with a different file."}
              </p>
              <button
                onClick={reset}
                className="rounded-xl bg-white/10 px-6 py-3 text-base font-medium text-white transition-colors hover:bg-white/20"
              >
                Try again
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
