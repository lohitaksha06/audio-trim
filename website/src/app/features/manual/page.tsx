"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import AudioPreview from "@/components/AudioPreview";
import ManualEditor, { type LocalEditResult } from "@/components/ManualEditor";
import { useProject } from "@/components/ProjectProvider";
import {
  downloadUrl,
  isBackendConfigured,
  type ProcessResponse,
} from "@/services/api";
import {
  downloadBlob,
  loadJson,
  removeJson,
  safeFilename,
  saveJson,
} from "@/lib/storage";

type ChatResult = ProcessResponse & { local_url?: string | null; summary?: string };

interface SavedManual {
  history: string[];
  edits: number;
  lastAction: string;
}

const EMPTY: SavedManual = { history: [], edits: 0, lastAction: "" };
const STORAGE_KEY = "audelle:manual:state";

export default function ManualPage() {
  const project = useProject();
  const localMode = !isBackendConfigured;
  const [history, setHistory] = useState<string[]>([]);
  const [edits, setEdits] = useState(0);
  const [lastAction, setLastAction] = useState("");
  const [lastResult, setLastResult] = useState<ChatResult | null>(null);
  const [resultBlob, setResultBlob] = useState<Blob | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [notice, setNotice] = useState("");
  const [editorBusy, setEditorBusy] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const resultUrlRef = useRef<string | null>(null);
  const restoredRef = useRef(false);

  useEffect(() => {
    const saved = loadJson<SavedManual>(STORAGE_KEY, EMPTY);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setHistory(Array.isArray(saved.history) ? saved.history : []);
    setEdits(Number.isFinite(saved.edits) ? saved.edits : 0);
    setLastAction(saved.lastAction ?? "");
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    saveJson(STORAGE_KEY, { history, edits, lastAction } satisfies SavedManual);
  }, [hydrated, history, edits, lastAction]);

  useEffect(() => {
    return () => {
      if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (!hydrated || lastResult || restoredRef.current) return;
    if (edits > 0 && project.currentBlob && project.objectUrl) {
      restoredRef.current = true;
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setResultBlob(project.currentBlob);
      setLastResult({
        intent: "restored",
        params: {},
        raw_prompt: lastAction,
        metadata: { restored: true },
        download_key: null,
        local_url: project.objectUrl,
        summary: lastAction || "Restored your last render.",
      });
    }
  }, [hydrated, lastResult, edits, project.currentBlob, project.objectUrl, lastAction]);

  useEffect(() => {
    if (!hydrated || project.loading || lastResult) return;
    if (edits > 0 && !project.currentBlob) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setEdits(0);
      setHistory([]);
      setLastAction("");
    }
  }, [hydrated, project.loading, project.currentBlob, edits, lastResult]);

  const clearResultUrl = useCallback(() => {
    if (resultUrlRef.current) {
      URL.revokeObjectURL(resultUrlRef.current);
      resultUrlRef.current = null;
    }
  }, []);

  const resetWorkspace = useCallback(() => {
    clearResultUrl();
    setHistory([]);
    setEdits(0);
    setLastAction("");
    setLastResult(null);
    setResultBlob(null);
    setErrorMsg("");
    setNotice("");
    restoredRef.current = false;
    removeJson(STORAGE_KEY);
  }, [clearResultUrl]);

  const onLocalResult = (result: LocalEditResult, label: string) => {
    clearResultUrl();
    const url = URL.createObjectURL(result.blob);
    resultUrlRef.current = url;
    setResultBlob(result.blob);
    setLastResult({
      intent: result.intent,
      params: result.params,
      raw_prompt: label,
      metadata: { local: true },
      download_key: null,
      local_url: url,
      summary: result.summary,
    });
    setLastAction(result.summary);
    setEdits((value) => value + 1);
    restoredRef.current = true;
    project.replaceAudio(
      result.audio,
      result.blob,
      safeFilename(`${project.fileName ?? "audelle"}-${result.intent}`, "wav"),
    );
    setHistory((prev) => [`${label} — ${result.summary}`, ...prev].slice(0, 30));
    setErrorMsg("");
  };

  const onServerResult = (result: ProcessResponse, label: string) => {
    setLastResult(result);
    setResultBlob(null);
    setLastAction(result.raw_prompt || label);
    setEdits((value) => value + 1);
    restoredRef.current = true;
    setHistory((prev) => [`${label} applied`, ...prev].slice(0, 30));
    setErrorMsg("");
  };

  const activePath = lastResult?.download_key ?? project.upload?.audio_path;
  const playSrc = localMode
    ? project.objectUrl
    : lastResult?.download_key
      ? downloadUrl(lastResult.download_key)
      : project.objectUrl;

  const handleDownload = () => {
    if (resultBlob) {
      downloadBlob(resultBlob, safeFilename(`${project.fileName ?? "audelle"}-manual`, "wav"));
      return;
    }
    if (project.currentBlob && edits > 0) {
      downloadBlob(project.currentBlob, safeFilename(`${project.fileName ?? "audelle"}-manual`, "wav"));
      return;
    }
    if (lastResult?.download_key) {
      window.open(downloadUrl(lastResult.download_key), "_blank", "noopener");
      return;
    }
    setNotice("No rendered file yet. Make a cut or a fade first.");
  };

  const resetToOriginal = () => {
    clearResultUrl();
    setHistory([]);
    setEdits(0);
    setLastAction("");
    setLastResult(null);
    setResultBlob(null);
    setErrorMsg("");
    setNotice("");
    restoredRef.current = false;
    project.resetToOriginal();
    removeJson(STORAGE_KEY);
  };

  const newFile = async () => {
    resetWorkspace();
    await project.clearProject();
  };

  if (!project.file) {
    return (
      <div className="flex h-screen flex-col bg-black">
        <Nav />
        <div className="mt-[var(--nav-h)] flex flex-1 overflow-hidden">
          <Sidebar onPromptSelect={() => {}} />
          <main className="flex-1 overflow-y-auto p-4 sm:p-6">
            <div className="mx-auto w-full max-w-3xl">
              <div className="mb-6 border-l-2 border-[#6aa9ff] pl-3">
                <p className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-[#9cc4ff]">
                  Workspace · waveform
                </p>
                <h1 className="mt-0.5 font-display text-3xl font-bold lowercase tracking-[-0.03em] text-[#f2f1ec]">
                  manual editor
                </h1>
                <p className="mt-1 max-w-2xl text-[0.9375rem] text-white/55">
                  Cut, trim, fade, change gain and speed by dragging on the waveform. Every edit builds
                  on the last one, and the result downloads as a WAV.
                </p>
              </div>
              <FileUpload
                onFileSelected={(picked) => {
                  resetWorkspace();
                  void project.selectFile(picked);
                }}
                onInvalid={setErrorMsg}
              />
              {(project.loading || project.error || errorMsg) && (
                <p className={`mt-3 text-center text-sm ${project.error || errorMsg ? "text-red-400" : "text-white/50"}`}>
                  {project.error || errorMsg || "Decoding the waveform on this device…"}
                </p>
              )}
              <div className="mt-8 grid gap-3 sm:grid-cols-3">
                {[
                  ["Waveform selection", "Drag a range, then keep or cut it."],
                  ["Fades & gain", "Shape the edges and level without a server."],
                  ["Speed", "Time-stretch with pitch preserved."],
                ].map(([title, body]) => (
                  <div key={title} className="border border-white/10 bg-white/[0.02] p-3">
                    <p className="text-sm text-white/85">{title}</p>
                    <p className="mt-1 text-xs leading-relaxed text-white/50">{body}</p>
                  </div>
                ))}
              </div>
            </div>
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen flex-col bg-black">
      <Nav />
      <div className="mt-[var(--nav-h)] flex flex-1 overflow-hidden">
        <Sidebar onPromptSelect={() => {}} />
        <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <div className="shrink-0 border-b border-white/8 px-4 py-3 sm:px-6">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="border-l-2 border-[#6aa9ff] pl-3">
                <p className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-[#9cc4ff]">
                  Workspace · waveform
                </p>
                <h1 className="mt-0.5 font-display text-2xl font-bold lowercase tracking-[-0.03em] text-[#f2f1ec] sm:text-3xl">
                  manual editor
                </h1>
                <p className="text-[0.9375rem] text-white/55">
                  Drag on the wave to select · trim, cut, fade, gain and speed. Each render chains.
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={resetToOriginal}
                  className="rounded-lg border border-white/12 px-3 py-1.5 text-xs text-white/65 hover:border-white/25 hover:text-white"
                >
                  Reset edits
                </button>
                <button
                  onClick={() => void newFile()}
                  className="rounded-lg border border-white/12 px-3 py-1.5 text-xs text-white/65 hover:border-white/25 hover:text-white"
                >
                  New file
                </button>
              </div>
            </div>
            {editorBusy && (
              <div className="mt-2 flex items-center gap-2 border border-white/10 bg-white/[0.025] px-2.5 py-1.5 text-[0.72rem] text-white/55">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white/70" />
                Rendering on this device — the sidebar still works.
              </div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-4 sm:p-6">
            <div className="mx-auto grid w-full max-w-6xl gap-4 lg:grid-cols-[1.5fr_0.7fr]">
              <div className="space-y-3">
                <div className="flex items-center gap-3 border border-white/8 bg-white/[0.02] p-3">
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-white">{project.file.name}</p>
                    <p className="text-xs text-white/55">
                      {edits === 0 ? "Original" : `Render ${edits} · ${lastAction || "edited"}`}
                      {project.loading ? " · decoding…" : ""}
                    </p>
                  </div>
                  {edits > 0 && (
                    <button
                      onClick={handleDownload}
                      className="shrink-0 rounded-lg bg-[#f2f1ec] px-3 py-2 text-xs font-semibold text-black hover:bg-white"
                    >
                      Download WAV
                    </button>
                  )}
                </div>

                {playSrc ? (
                  <ManualEditor
                    src={playSrc}
                    audioPath={activePath}
                    localAudio={localMode ? project.audio : null}
                    onResult={onServerResult}
                    onLocalResult={onLocalResult}
                    onBusyChange={setEditorBusy}
                    disabled={project.loading}
                  />
                ) : (
                  <div className="border border-white/10 p-6 text-sm text-white/55">
                    The waveform is still loading. If this stays empty, upload the file again.
                  </div>
                )}

                {history.length > 0 && (
                  <div className="border border-white/8 bg-white/[0.01] p-3">
                    <p className="eyebrow mb-1.5">Edit history</p>
                    <ul className="space-y-1">
                      {history.map((item, index) => (
                        <li key={`${item}-${index}`} className="text-xs text-white/70">
                          ✓ {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {(project.error || errorMsg) && (
                  <p className="text-xs text-red-400">{project.error || errorMsg}</p>
                )}
              </div>

              <aside className="space-y-3">
                {lastResult && playSrc && (
                  <div className="border border-white/12 bg-white/[0.025] p-3">
                    <p className="eyebrow mb-2">Latest render</p>
                    <AudioPreview src={playSrc} height={48} />
                    <button
                      onClick={handleDownload}
                      className="mt-2 w-full rounded-none border border-white/25 bg-white/[0.07] px-3 py-2.5 text-sm font-medium text-white/85 hover:border-white/50"
                    >
                      Download WAV
                    </button>
                    <p className="mt-2 text-xs leading-relaxed text-white/50">
                      {lastResult.summary || lastResult.raw_prompt || "Rendered edit."}
                    </p>
                  </div>
                )}

                <div className="border border-white/8 bg-white/[0.02] p-3">
                  <p className="eyebrow mb-2">How chaining works</p>
                  <p className="text-xs leading-relaxed text-white/55">
                    Each edit starts from the latest render. Use <strong className="text-white/75">Reset edits</strong>{" "}
                    to go back to the original without uploading again.
                  </p>
                </div>

                <div className="border border-white/8 bg-white/[0.02] p-3">
                  <p className="eyebrow mb-2">On this device</p>
                  <p className="text-xs leading-relaxed text-white/55">
                    The source file and the latest render are stored in this browser. Refreshing this
                    page restores both.
                  </p>
                </div>
              </aside>
            </div>
          </div>

          {notice && (
            <div className="shrink-0 border-t border-white/10 px-4 py-2 text-xs text-amber-200/80">
              {notice}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
