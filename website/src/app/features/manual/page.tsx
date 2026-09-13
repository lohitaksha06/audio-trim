"use client";

import { useEffect, useState } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import AudioPreview from "@/components/AudioPreview";
import ManualEditor from "@/components/ManualEditor";
import { uploadFile, downloadUrl, type UploadResponse, type ProcessResponse } from "@/services/api";

interface LastAudio {
  audio_path: string;
  filename: string;
}

export default function ManualPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [activePath, setActivePath] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<ProcessResponse | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [edits, setEdits] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [uploading, setUploading] = useState(false);
  const [resumable, setResumable] = useState<LastAudio | null>(null);

  useEffect(() => {
    try {
      const v = localStorage.getItem("audelle:lastAudio");
      if (v) setResumable(JSON.parse(v));
    } catch { /* ignore */ }
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const remember = (audio_path: string, filename: string) => {
    try {
      localStorage.setItem("audelle:lastAudio", JSON.stringify({ audio_path, filename }));
    } catch { /* ignore */ }
  };

  const handleFileSelected = async (f: File) => {
    setFile(f);
    setErrorMsg("");
    setLastResult(null);
    setHistory([]);
    setEdits(0);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    setObjectUrl(URL.createObjectURL(f));
    setUploading(true);
    try {
      const result = await uploadFile(f);
      setUploadResult(result);
      setActivePath(result.audio_path);
      remember(result.audio_path, f.name);
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleResume = () => {
    if (!resumable) return;
    setFile(null);
    setUploadResult(null);
    setLastResult(null);
    setHistory([]);
    setEdits(0);
    setErrorMsg("");
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    setObjectUrl(null);
    setActivePath(resumable.audio_path);
  };

  const handleResult = (res: ProcessResponse, label: string) => {
    setLastResult(res);
    setEdits((n) => n + 1);
    setHistory((prev) => [`${label} applied`, ...prev].slice(0, 20));
    if (res.download_key) setActivePath(res.download_key);
  };

  const reset = () => {
    setFile(null);
    setUploadResult(null);
    setLastResult(null);
    setHistory([]);
    setEdits(0);
    setErrorMsg("");
    setActivePath(null);
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    setObjectUrl(null);
  };

  const playSrc = lastResult?.download_key
    ? downloadUrl(lastResult.download_key)
    : activePath && !objectUrl
      ? downloadUrl(activePath)
      : objectUrl;
  const ready = !!activePath && !!playSrc;

  return (
    <div className="flex h-screen flex-col bg-black">
      <Nav />
      <div className="flex flex-1 mt-14 sm:mt-16 overflow-hidden">
        <Sidebar onPromptSelect={() => {}} />
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          <div className="shrink-0 border-b border-white/5 px-4 sm:px-6 py-3 flex items-center justify-between">
            <div>
              <h1 className="text-lg sm:text-xl font-bold text-white">Manual Editor</h1>
              <p className="text-xs text-white/40">Hands-on waveform editing — select, trim, fade, gain, speed. Each edit builds on the last.</p>
            </div>
            {ready && (
              <button onClick={reset} className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 hover:border-white/20 hover:text-white/70 transition-colors">
                New file
              </button>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-4 sm:p-6">
            {!ready ? (
              <div className="flex flex-col items-center justify-center py-10">
                <div className="w-full max-w-xl space-y-3">
                  {resumable && (
                    <button
                      onClick={handleResume}
                      className="w-full rounded-2xl border border-neon-blue/20 bg-neon-blue/[0.05] p-4 text-left hover:bg-neon-blue/10 transition-colors"
                    >
                      <span className="block text-sm font-medium text-neon-blue">Continue where you left off</span>
                      <span className="block truncate text-xs text-white/40">{resumable.filename}</span>
                    </button>
                  )}
                  <FileUpload onFileSelected={handleFileSelected} onInvalid={setErrorMsg} />
                  <p className="text-center text-xs text-white/20">…or describe it in words in Prompt Mode — same engine underneath.</p>
                  {uploading && <p className="text-center text-xs text-neon-blue animate-pulse">Uploading…</p>}
                  {errorMsg && <p className="text-center text-xs text-red-400">{errorMsg}</p>}
                </div>
              </div>
            ) : (
              <div className="mx-auto w-full max-w-5xl space-y-4">
                <div className="flex items-center gap-3">
                  <div className="flex-1 min-w-0">
                    <p className="truncate text-sm font-medium text-white">
                      {file?.name ?? resumable?.filename ?? "Audio"}
                    </p>
                    <p className="text-xs text-white/40">
                      {edits === 0 ? "Original" : `Layering on output ${edits} — each edit builds on the last`}
                    </p>
                  </div>
                  {lastResult?.download_key && (
                    <a
                      href={downloadUrl(lastResult.download_key)}
                      target="_blank"
                      rel="noopener"
                      className="shrink-0 rounded-lg bg-neon-blue/15 px-4 py-2 text-xs font-medium text-neon-blue hover:bg-neon-blue/25 transition-colors"
                    >
                      Download result
                    </a>
                  )}
                </div>

                <ManualEditor src={playSrc!} audioPath={activePath!} onResult={handleResult} />

                {lastResult?.download_key && (
                  <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
                    <p className="mb-2 text-[10px] uppercase tracking-wider text-white/30">Latest output</p>
                    <AudioPreview src={downloadUrl(lastResult.download_key)} height={48} />
                  </div>
                )}

                {history.length > 0 && (
                  <div className="rounded-2xl border border-white/5 bg-white/[0.01] p-3">
                    <p className="mb-1.5 text-[10px] uppercase tracking-wider text-white/30">Edit history</p>
                    <ul className="space-y-1">
                      {history.map((h, i) => (
                        <li key={i} className="text-xs text-white/50">✓ {h}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
