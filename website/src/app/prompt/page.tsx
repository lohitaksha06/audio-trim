"use client";

import { useState } from "react";
import Nav from "@/components/Nav";
import FileUpload from "@/components/FileUpload";
import { uploadFile, type UploadResponse } from "@/services/api";

type PageState = "idle" | "uploading" | "analyzed" | "processing" | "completed" | "error";

export default function PromptPage() {
  const [state, setState] = useState<PageState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

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
    setState("processing");
    setTimeout(() => {
      setState("completed");
      setResultUrl("#");
    }, 2000);
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
    setResultUrl(null);
    setErrorMsg("");
  };

  const analysis = uploadResult?.analysis;
  const structure = [
    { label: "Intro", time: "0:00" },
    { label: "Full track", time: analysis ? formatDuration(analysis.duration_seconds) : "0:00" },
  ];

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
    <div className="flex min-h-screen flex-col bg-black">
      <Nav />

      <main className="mx-auto mt-24 w-full max-w-5xl flex-1 px-4 pb-16">
        {state === "idle" && (
          <div className="flex flex-col items-center">
            <div className="mb-8 text-center">
              <h1 className="mb-2 text-3xl font-bold text-white">Edit with AI</h1>
              <p className="text-white/40">
                Upload audio (or video) and describe what you want to do.
              </p>
            </div>
            <div className="w-full max-w-2xl">
              <FileUpload onFileSelected={handleFileSelected} />
            </div>
          </div>
        )}

        {state === "uploading" && (
          <div className="flex flex-col items-center justify-center py-24">
            <div className="mb-6 flex h-16 w-16 items-center justify-center">
              <div className="h-10 w-10 animate-spin rounded-full border-2 border-neon-blue border-t-transparent" />
            </div>
            <p className="text-white/60">Uploading and analyzing your file...</p>
            <p className="mt-1 text-sm text-white/30">
              Extracting audio info, BPM, key, and energy profile
            </p>
          </div>
        )}

        {(state === "analyzed" || state === "processing" || state === "completed") && file && (
          <div className="space-y-6">
            <div className="flex items-center gap-4 rounded-xl border border-white/10 bg-white/[0.02] p-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-neon-blue/20 to-neon-purple/20">
                <span className="text-2xl">
                  {uploadResult?.is_video ? "🎬" : "🎵"}
                </span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="truncate font-medium text-white">{file.name}</p>
                <p className="text-sm text-white/40">
                  {uploadResult?.is_video ? "Video (audio extracted)" : "Audio"} &middot;{" "}
                  {formatSize(file.size)}
                </p>
              </div>
              {state === "completed" && (
                <span className="rounded-full bg-green-500/10 px-3 py-1 text-xs font-medium text-green-400">
                  Completed
                </span>
              )}
              <button
                onClick={reset}
                className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 transition-colors hover:border-white/20 hover:text-white/70"
              >
                New file
              </button>
            </div>

            <div className="grid gap-6 lg:grid-cols-3">
              <div className="space-y-6 lg:col-span-2">
                <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5">
                  <h3 className="mb-3 text-sm font-semibold text-white/50 uppercase tracking-wider">
                    Analysis
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {details.map((d) => (
                      <span
                        key={d.label}
                        className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-white/70"
                      >
                        <span className="text-white/40">{d.label}:</span> {d.value}
                      </span>
                    ))}
                  </div>
                  <p className="mt-3 text-xs text-white/30 italic">
                    What would you like to change? Just describe it below.
                  </p>
                </div>

                <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5">
                  <h3 className="mb-3 text-sm font-semibold text-white/50 uppercase tracking-wider">
                    Structure
                  </h3>
                  <div className="flex flex-wrap gap-1.5">
                    {structure.map((s, i) => (
                      <span
                        key={i}
                        className="inline-flex items-center gap-1 rounded-md border border-white/5 bg-white/[0.03] px-2.5 py-1 text-xs text-white/50"
                      >
                        {s.label}
                        <span className="text-white/20">{s.time}</span>
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5">
                  <h3 className="mb-3 text-sm font-semibold text-white/50 uppercase tracking-wider">
                    Details
                  </h3>
                  <dl className="space-y-3">
                    {details.map((d) => (
                      <div key={d.label} className="flex justify-between">
                        <dt className="text-sm text-white/40">{d.label}</dt>
                        <dd className="text-sm font-medium text-white">{d.value}</dd>
                      </div>
                    ))}
                  </dl>
                </div>

                {state === "processing" && (
                  <div className="rounded-xl border border-neon-blue/20 bg-neon-blue/5 p-5">
                    <div className="mb-2 flex items-center gap-2">
                      <div className="h-3 w-3 animate-spin rounded-full border border-neon-blue border-t-transparent" />
                      <span className="text-sm font-medium text-neon-blue">
                        Processing your prompt...
                      </span>
                    </div>
                    <p className="text-xs text-neon-blue/50">
                      AI is editing the audio based on your request
                    </p>
                  </div>
                )}

                {state === "completed" && (
                  <div className="rounded-xl border border-green-500/20 bg-green-500/5 p-5">
                    <div className="mb-3 flex items-center gap-2">
                      <span className="text-lg">✅</span>
                      <span className="text-sm font-medium text-green-400">
                        Done!
                      </span>
                    </div>
                    <div className="flex items-center gap-2 rounded-lg bg-white/5 px-3 py-2 text-sm text-white/70">
                      <svg className="h-4 w-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                      </svg>
                      Preview
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5">
              <label htmlFor="prompt" className="mb-3 block text-sm font-semibold text-white/50 uppercase tracking-wider">
                What do you want to do?
              </label>
              <div className="flex gap-3">
                <input
                  id="prompt"
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder='e.g. "Remove the kick drum" or "Give me just the vocals" or "Make this sound darker"'
                  className="flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white placeholder-white/20 outline-none transition-colors focus:border-neon-blue/50 focus:ring-1 focus:ring-neon-blue/20"
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
                  className="shrink-0 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-6 py-3 text-sm font-semibold text-black transition-all duration-300 hover:scale-105 hover:shadow-[0_0_20px_rgba(0,212,255,0.3)] disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100 disabled:hover:shadow-none"
                >
                  {state === "processing" ? "Processing..." : "Process"}
                </button>
              </div>
            </div>
          </div>
        )}

        {state === "error" && (
          <div className="flex flex-col items-center justify-center py-24">
            <span className="mb-4 text-4xl">⚠️</span>
            <p className="mb-2 text-lg font-medium text-white">Something went wrong</p>
            <p className="mb-6 max-w-md text-center text-sm text-white/40">
              {errorMsg || "Please try again with a different file."}
            </p>
            <button
              onClick={reset}
              className="rounded-xl bg-white/10 px-6 py-3 text-sm font-medium text-white transition-colors hover:bg-white/20"
            >
              Try again
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
