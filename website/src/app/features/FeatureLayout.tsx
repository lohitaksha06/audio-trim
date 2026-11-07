"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import AudioPreview from "@/components/AudioPreview";
import { useProject } from "@/components/ProjectProvider";
import {
  downloadUrl,
  exportZip,
  fetchSuggestions,
  isBackendConfigured,
  processAudio,
  understandAudio,
  type ProcessResponse,
  type UnderstandResponse,
} from "@/services/api";
import {
  durationOf,
  parseLocalPrompt,
  runLocalPlanAsync,
} from "@/lib/audio";
import { describeResult, isOutputFollowUp, type ResultMeta } from "@/utils/followUp";
import {
  downloadBlob,
  loadJson,
  removeJson,
  safeFilename,
  saveJson,
} from "@/lib/storage";
import { FeaturePromptContext } from "./FeaturePromptContext";
import { Conversation } from "./FeaturePanels";

interface FeatureLayoutProps {
  children: ReactNode;
  title: string;
  subtitle: string;
  /** Stable key for restoring this workspace's conversation/history. */
  featureKey?: string;
  /** Small label in the header, e.g. "source separation". */
  eyebrow?: string;
  /** One sentence that tells the user what belongs on this page. */
  intro?: string;
  /** A per-page accent so the workspaces do not all look identical. */
  accent?: string;
}

/** Backend result, plus the object URL a browser-only edit produces. */
type ChatResult = ProcessResponse & { local_url?: string | null; summary?: string };

interface SavedFeature {
  history: { role: string; text: string }[];
  prompt: string;
  edits: number;
  fromOriginal: boolean;
  lastAction: string;
  /** Server results are plain JSON; local object URLs are never stored. */
  lastResult?: ChatResult | null;
}

const EMPTY_SAVED: SavedFeature = {
  history: [],
  prompt: "",
  edits: 0,
  fromOriginal: false,
  lastAction: "",
  lastResult: null,
};

export default function FeatureLayout({
  children,
  title,
  subtitle,
  featureKey,
  eyebrow,
  intro,
  accent = "#f2f1ec",
}: FeatureLayoutProps) {
  const project = useProject();
  const localMode = !isBackendConfigured;
  const storageKey = `audelle:feature:${featureKey ?? title.toLowerCase().replace(/\s+/g, "-")}`;

  const [history, setHistory] = useState<{ role: string; text: string }[]>([]);
  const [prompt, setPrompt] = useState("");
  const [understand, setUnderstand] = useState<UnderstandResponse | null>(null);
  const [lastResult, setLastResult] = useState<ChatResult | null>(null);
  const [resultBlob, setResultBlob] = useState<Blob | null>(null);
  const [localProgress, setLocalProgress] = useState<{ fraction: number; label: string } | null>(null);
  const [processing, setProcessing] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [notice, setNotice] = useState("");
  const [edits, setEdits] = useState(0);
  const [fromOriginal, setFromOriginal] = useState(false);
  const [lastAction, setLastAction] = useState("");
  const [hydrated, setHydrated] = useState(false);

  const processingRef = useRef(false);
  const resultUrlRef = useRef<string | null>(null);
  const restoredRef = useRef(false);

  // Restore the small, serialisable part of this workspace. The audio itself
  // comes from ProjectProvider, so a refresh keeps both the file and the last
  // render instead of dropping the user back at an empty upload box.
  useEffect(() => {
    const saved = loadJson<SavedFeature>(storageKey, EMPTY_SAVED);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setHistory(Array.isArray(saved.history) ? saved.history : []);
    setPrompt(typeof saved.prompt === "string" ? saved.prompt : "");
    setEdits(Number.isFinite(saved.edits) ? saved.edits : 0);
    setFromOriginal(Boolean(saved.fromOriginal));
    setLastAction(typeof saved.lastAction === "string" ? saved.lastAction : "");
    if (saved.lastResult?.download_key) setLastResult(saved.lastResult);
    setHydrated(true);
  }, [storageKey]);

  useEffect(() => {
    if (!hydrated) return;
    saveJson(storageKey, {
      history,
      prompt,
      edits,
      fromOriginal,
      lastAction,
      lastResult: lastResult && !lastResult.local_url ? lastResult : null,
    } satisfies SavedFeature);
  }, [hydrated, storageKey, history, prompt, edits, fromOriginal, lastAction, lastResult]);

  useEffect(() => {
    return () => {
      if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current);
    };
  }, []);

  // If the page was refreshed after an edit, ProjectProvider has the rendered
  // audio. Re-attach it to the output panel instead of showing "no output".
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
        summary: lastAction || "Restored your last render from this device.",
      });
    }
  }, [hydrated, lastResult, edits, project.currentBlob, project.objectUrl, lastAction]);

  // If another workspace reset the shared project, this page's saved edit
  // count would otherwise say "1 EDIT" with no file to download. Clear the
  // stale workspace once the provider has finished loading.
  useEffect(() => {
    if (!hydrated || project.loading || lastResult) return;
    if (edits > 0 && !project.currentBlob) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setEdits(0);
      setHistory([]);
      setLastAction("");
      setFromOriginal(false);
    }
  }, [hydrated, project.loading, project.currentBlob, edits, lastResult]);

  // Server mode can analyze automatically; browser mode says so plainly.
  const analyzedPathRef = useRef<string | null>(null);
  useEffect(() => {
    if (localMode || !hydrated) return;
    const path = project.upload?.audio_path;
    if (!path || analyzedPathRef.current === path) return;
    analyzedPathRef.current = path;
    void (async () => {
      try {
        setUnderstand(await understandAudio(path));
      } catch {
        /* Analysis is optional; edits still work. */
      }
    })();
  }, [localMode, hydrated, project.upload?.audio_path]);

  const clearResultUrl = useCallback(() => {
    if (resultUrlRef.current) {
      URL.revokeObjectURL(resultUrlRef.current);
      resultUrlRef.current = null;
    }
  }, []);

  const clearWorkspaceState = useCallback(() => {
    clearResultUrl();
    setHistory([]);
    setPrompt("");
    setUnderstand(null);
    setLastResult(null);
    setResultBlob(null);
    setLocalProgress(null);
    setErrorMsg("");
    setNotice("");
    setEdits(0);
    setFromOriginal(false);
    setLastAction("");
    restoredRef.current = false;
    removeJson(storageKey);
  }, [clearResultUrl, storageKey]);

  const handleFileSelected = async (picked: File) => {
    clearWorkspaceState();
    setNotice("");
    await project.selectFile(picked);
  };

  const handleAnalyze = async () => {
    if (localMode) {
      setNotice(
        "Instrument, genre and structure analysis needs the server. The edit actions below are available now.",
      );
      return;
    }
    const target = project.upload?.audio_path;
    if (!target) return;
    setErrorMsg("");
    try {
      setUnderstand(await understandAudio(target));
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Analysis failed");
    }
  };

  const resultText = `${lastResult?.summary ?? ""} ${lastResult?.raw_prompt ?? ""}`.toLowerCase();
  const downloadKind = resultText.includes("karaoke") || resultText.includes("cut vocals")
    ? "instrumental"
    : resultText.includes("vocal-forward") || resultText.includes("soloed vocals")
      ? "vocals"
      : resultText.includes("soloed")
        ? "isolated"
        : lastResult?.intent ?? "edit";
  const downloadName = safeFilename(`${project.fileName ?? "audelle"}-${downloadKind}`, "wav");
  const downloadLabel = !localMode
    ? "Download result"
    : downloadKind === "vocals"
      ? "Download vocal mix (WAV)"
      : downloadKind === "instrumental"
        ? "Download instrumental (WAV)"
        : downloadKind === "isolated"
          ? "Download isolated part (WAV)"
          : "Download WAV";

  const handleDownload = useCallback(() => {
    if (resultBlob) {
      downloadBlob(resultBlob, downloadName);
      return;
    }
    if (project.currentBlob && edits > 0) {
      downloadBlob(project.currentBlob, downloadName);
      return;
    }
    if (lastResult?.download_key) {
      window.open(downloadUrl(lastResult.download_key), "_blank", "noopener");
      return;
    }
    setNotice("No rendered file yet. Run an edit first, then download the WAV.");
  }, [resultBlob, project.currentBlob, edits, lastResult, downloadName]);

  const handleDownloadOriginal = () => {
    if (!project.file) {
      setNotice("Upload a file first.");
      return;
    }
    downloadBlob(project.file, project.file.name || "audelle-original");
  };

  const handleProcess = async (override?: string) => {
    const raw = (override ?? prompt).trim();
    if (!raw) return;
    if (!project.file) {
      setHistory((prev) => [...prev, { role: "ai", text: "Please upload a file first before processing." }]);
      return;
    }

    setNotice("");
    if (isOutputFollowUp(raw)) {
      setHistory((prev) => [...prev, { role: "user", text: raw }]);
      setPrompt("");
      if (lastResult || project.currentBlob) {
        setHistory((prev) => [...prev, { role: "ai", text: "Here’s your latest output — starting the download." }]);
        handleDownload();
      } else {
        setHistory((prev) => [...prev, { role: "ai", text: "No output yet — describe an edit first." }]);
      }
      return;
    }

    if (localMode) {
      if (processingRef.current) return;
      const base = (!fromOriginal && project.audio) || project.original;
      if (!base) {
        setNotice("The file is still loading. Try again in a moment.");
        return;
      }
      const plan = parseLocalPrompt(raw, durationOf(base));
      setHistory((prev) => [...prev, { role: "user", text: raw }]);
      setPrompt("");
      if (plan.unsupported || plan.intent === "unknown") {
        setNotice(plan.unsupported ?? "That edit is not available yet.");
        return;
      }
      setProcessing(true);
      setErrorMsg("");
      processingRef.current = true;
      setLocalProgress({ fraction: 0, label: "Starting" });
      try {
        const result = await runLocalPlanAsync(base, plan, (fraction, label) => {
          setLocalProgress({ fraction, label });
        });
        if (result.metadata.info_only) {
          setHistory((prev) => [...prev, { role: "ai", text: result.plan.summary }]);
          return;
        }
        clearResultUrl();
        const url = URL.createObjectURL(result.blob);
        resultUrlRef.current = url;
        setResultBlob(result.blob);
        setLastResult({
          intent: plan.intent,
          params: plan.params,
          raw_prompt: raw,
          metadata: result.metadata,
          download_key: null,
          local_url: url,
          summary: result.plan.summary,
        });
        setLastAction(result.plan.summary);
        setEdits((n) => n + 1);
        setFromOriginal(false);
        restoredRef.current = true;
        project.replaceAudio(
          result.audio,
          result.blob,
          safeFilename(`${project.fileName ?? "audelle"}-${plan.intent}`, "wav"),
        );
        const detail = describeResult(plan.intent as ProcessResponse["intent"], result.metadata as ResultMeta);
        const message = detail ? `Done — ${detail}.` : `Done — ${result.plan.summary}.`;
        setHistory((prev) => [...prev, { role: "ai", text: message }]);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Processing failed";
        setHistory((prev) => [...prev, { role: "ai", text: `Error: ${msg}` }]);
        setErrorMsg(msg);
      } finally {
        processingRef.current = false;
        setProcessing(false);
        setLocalProgress(null);
      }
      return;
    }

    const srcPath = !fromOriginal && lastResult?.download_key
      ? lastResult.download_key
      : project.upload?.audio_path;
    if (!srcPath) {
      setHistory((prev) => [...prev, { role: "ai", text: "Please upload a file first before processing." }]);
      return;
    }
    setHistory((prev) => [...prev, { role: "user", text: raw }]);
    setPrompt("");
    setProcessing(true);
    setErrorMsg("");
    processingRef.current = true;
    try {
      const result = await processAudio(srcPath, raw);
      setLastResult(result);
      setResultBlob(null);
      if (result.intent === "unknown") {
        const ideas = await fetchSuggestions(raw);
        const list = ideas.length ? ideas.map((item) => `"${item}"`).join(", ") : '"Trim from 0:05 to 0:10"';
        setHistory((prev) => [
          ...prev,
          { role: "ai", text: `I couldn’t turn “${raw}” into a specific edit yet. Near matches: ${list}.` },
        ]);
        return;
      }
      setEdits((n) => n + 1);
      setFromOriginal(false);
      setLastAction(result.raw_prompt);
      restoredRef.current = true;
      const detail = describeResult(result.intent, result.metadata as ResultMeta | null);
      const message = detail ? `Done — ${detail}.` : `Done! Intent: ${result.intent}.`;
      setHistory((prev) => [...prev, { role: "ai", text: message }]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Processing failed";
      setHistory((prev) => [...prev, { role: "ai", text: /what BPM/i.test(msg) ? msg : `Error: ${msg}` }]);
      setErrorMsg(msg);
    } finally {
      processingRef.current = false;
      setProcessing(false);
    }
  };

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

  const newChat = () => {
    clearResultUrl();
    setHistory([]);
    setPrompt("");
    setLastResult(null);
    setResultBlob(null);
    setErrorMsg("");
    setNotice("");
    setEdits(0);
    setFromOriginal(false);
    setLastAction("");
    restoredRef.current = false;
    project.resetToOriginal();
    removeJson(storageKey);
  };

  const reset = async () => {
    clearWorkspaceState();
    await project.clearProject();
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  };

  const formatDuration = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const analysis = project.analysis;
  const inst = understand?.instruments;
  const sections = understand?.structure?.sections ?? [];
  const hasOutput = Boolean(lastResult || (edits > 0 && project.currentBlob));
  const outputSrc = lastResult?.local_url
    ?? (lastResult?.download_key ? downloadUrl(lastResult.download_key) : project.objectUrl)
    ?? "";

  const ctx = {
    setPrompt,
    triggerPrompt: (value: string) => void handleProcess(value),
    hasFile: Boolean(project.file),
    analysis,
    understand,
    busy: processing || project.loading || exporting,
    hasResult: hasOutput,
    fileName: project.fileName,
    edits,
    lastMetadata: (lastResult?.metadata ?? null) as Record<string, unknown> | null,
    localMode,
    objectUrl: project.objectUrl,
    localAudio: project.audio,
  };

  const busy = processing || project.loading || exporting;

  return (
    <FeaturePromptContext.Provider value={ctx}>
      <div className="flex h-screen flex-col bg-black">
        <Nav />

        <div className="mt-[var(--nav-h)] flex flex-1 overflow-hidden">
          <Sidebar onPromptSelect={setPrompt} />

          <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
            <div className="shrink-0 border-b border-white/8 px-4 py-3 sm:px-6">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 border-l-2 pl-3" style={{ borderColor: accent }}>
                  <p
                    className="font-mono text-[0.6rem] uppercase tracking-[0.2em]"
                    style={{ color: accent }}
                  >
                    Workspace · {eyebrow ?? "edit"}
                  </p>
                  <h1 className="mt-0.5 font-display text-2xl font-bold lowercase tracking-[-0.03em] text-[#f2f1ec] sm:text-3xl">
                    {title}
                  </h1>
                  <p className="max-w-2xl text-[0.9375rem] leading-snug text-white/55">
                    {intro ?? subtitle}
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {project.file && (
                    <>
                      <button
                        onClick={newChat}
                        title="Clear this conversation and go back to the original file"
                        className="rounded-lg border border-white/12 px-3 py-1.5 text-[0.8125rem] text-white/70 transition-colors hover:border-white/25 hover:text-white"
                      >
                        Reset edits
                      </button>
                      <button
                        onClick={() => void reset()}
                        className="rounded-lg border border-white/12 px-3 py-1.5 text-[0.8125rem] text-white/70 transition-colors hover:border-white/25 hover:text-white"
                      >
                        New file
                      </button>
                    </>
                  )}
                </div>
              </div>

              {processing && (
                <div className="mt-2 flex items-center gap-2 border border-white/10 bg-white/[0.025] px-2.5 py-1.5 text-[0.72rem] text-white/55">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white/70" />
                  Rendering in the background — the sidebar still works.
                </div>
              )}

              {edits > 0 && hasOutput && (
                <p className="mt-2 font-mono text-[0.62rem] uppercase tracking-[0.14em] text-white/40">
                  Layer {edits} ·{" "}
                  {fromOriginal ? (
                    <>
                      starting from the original{" "}
                      <button onClick={() => setFromOriginal(false)} className="underline">
                        back to latest
                      </button>
                    </>
                  ) : (
                    <>
                      building on the latest render{" "}
                      <button onClick={() => setFromOriginal(true)} className="underline">
                        start from original
                      </button>
                    </>
                  )}
                </p>
              )}
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-4 sm:px-6">
              {!project.file ? (
                <div className="mx-auto w-full max-w-3xl">
                  <FileUpload onFileSelected={(picked) => void handleFileSelected(picked)} onInvalid={setErrorMsg} />
                  {(project.loading || project.error || errorMsg) && (
                    <p
                      className={`mt-3 text-center text-sm ${
                        project.error || errorMsg ? "text-red-400" : "text-white/50"
                      }`}
                    >
                      {project.error || errorMsg || "Decoding and measuring…"}
                    </p>
                  )}

                  <div className="mt-10 border-t border-white/8 pt-5">
                    <div className="mb-3 flex items-end justify-between gap-3">
                      <div>
                        <p className="eyebrow">What this workspace does</p>
                        <p className="mt-1 text-sm text-white/55">{subtitle}</p>
                      </div>
                      <span className="hidden font-mono text-[0.6rem] uppercase tracking-[0.16em] text-white/35 sm:inline">
                        upload to unlock
                      </span>
                    </div>
                    <div className="opacity-70">{children}</div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="mb-4 flex items-center gap-3 border-b border-white/8 pb-3">
                    <div className="flex h-9 w-9 items-center justify-center border border-white/15 bg-white/[0.03]">
                      <svg className="h-4 w-4 text-white/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
                      </svg>
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-white">{project.file.name}</p>
                      <p className="text-xs text-white/60">
                        {project.upload?.is_video ? "Video" : "Audio"} · {formatSize(project.file.size)}
                        {analysis && (
                          <>
                            {" "}· {formatDuration(analysis.duration_seconds)} ·{" "}
                            {analysis.bpm ? `${analysis.bpm.toFixed(0)} BPM` : "— BPM"}
                          </>
                        )}
                        {project.loading && <span className="ml-2 animate-pulse text-white/45">decoding…</span>}
                      </p>
                    </div>
                    {edits > 0 && hasOutput && (
                      <span className="border border-white/20 bg-white/[0.06] px-2.5 py-1 font-mono text-[0.6rem] uppercase tracking-[0.14em] text-white/60">
                        {edits} edit{edits === 1 ? "" : "s"}
                      </span>
                    )}
                    {!localMode && (
                      <button
                        onClick={() => void handleAnalyze()}
                        className="rounded-lg border border-white/12 px-2.5 py-1 text-[0.75rem] text-white/60 transition-colors hover:border-white/30 hover:text-white/85"
                      >
                        Analyze
                      </button>
                    )}
                  </div>

                  <div className="mb-4 grid grid-cols-3 gap-2">
                    <div className="border border-white/8 bg-white/[0.02] p-2.5 text-center">
                      <div className="text-[0.75rem] text-white/50">Duration</div>
                      <div className="text-[0.9375rem] text-white">
                        {analysis ? formatDuration(analysis.duration_seconds) : "—"}
                      </div>
                    </div>
                    <div className="border border-white/8 bg-white/[0.02] p-2.5 text-center">
                      <div className="text-[0.75rem] text-white/50">BPM</div>
                      <div className="text-[0.9375rem] text-white">
                        {analysis?.bpm ? analysis.bpm.toFixed(0) : "—"}
                      </div>
                    </div>
                    <div className="border border-white/8 bg-white/[0.02] p-2.5 text-center">
                      <div className="text-[0.75rem] text-white/50">Key</div>
                      <div className="text-[0.9375rem] text-white">{analysis?.key ?? "—"}</div>
                    </div>
                  </div>

                  {localProgress && (
                    <div className="mb-4 border border-white/12 bg-white/[0.03] p-3">
                      <div className="flex justify-between text-xs text-white/60">
                        <span>{localProgress.label}</span>
                        <span>{Math.round(localProgress.fraction * 100)}%</span>
                      </div>
                      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/10">
                        <div
                          className="h-full rounded-full bg-white/70 transition-[width]"
                          style={{ width: `${Math.round(localProgress.fraction * 100)}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {understand && (
                    <div className="mb-4 grid gap-2">
                      {inst && (
                        <div className="border border-white/10 bg-white/[0.02] p-3">
                          <div className="mb-1 text-xs uppercase tracking-wider text-white/60">Instruments</div>
                          <div className="flex flex-wrap gap-1.5">
                            {inst.instruments.map((el) => (
                              <span key={el.instrument} className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-xs text-white/75">
                                {el.instrument} · {(el.confidence * 100).toFixed(0)}%
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                      {sections.length > 0 && (
                        <div className="border border-white/10 bg-white/[0.02] p-3">
                          <div className="eyebrow mb-1.5">Structure</div>
                          {sections.map((section, index) => (
                            <div key={index} className="flex justify-between text-[0.875rem]">
                              <span className="capitalize text-white/80">{section.label}</span>
                              <span className="text-white/55">
                                {formatDuration(section.start)}–{formatDuration(section.end)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Output first: after any action the download is visible
                      without scrolling past a long tool list. */}
                  {hasOutput && (
                    <section className="mb-4 border border-white/12 bg-white/[0.025] p-3" aria-label="Output">
                      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                        <div>
                          <p className="eyebrow">Latest output</p>
                          <p className="text-xs text-white/50">
                            {lastResult?.summary || "Rendered."}
                          </p>
                        </div>
                        <span className="font-mono text-[0.6rem] uppercase tracking-[0.14em] text-white/40">
                          ready to download
                        </span>
                      </div>
                      {outputSrc ? (
                        <AudioPreview src={outputSrc} height={48} />
                      ) : (
                        <p className="text-xs text-white/50">Preparing the preview…</p>
                      )}
                      <div className="mt-2 flex flex-wrap gap-2">
                        <button
                          onClick={handleDownload}
                          className="rounded-none border border-white/25 bg-white/[0.07] px-4 py-2.5 text-[0.875rem] font-medium text-white/85 transition-colors hover:border-white/50 hover:text-white"
                        >
                          {downloadLabel}
                        </button>
                        <button
                          onClick={handleDownloadOriginal}
                          className="rounded-none border border-white/12 px-3 py-2.5 text-[0.8125rem] text-white/55 transition-colors hover:border-white/30 hover:text-white/85"
                        >
                          Download original
                        </button>
                        {!localMode && lastResult?.stems_keys && Object.keys(lastResult.stems_keys).length > 0 && (
                          <button
                            onClick={() => void handleExportZip()}
                            disabled={exporting}
                            className="rounded-none border border-white/20 px-3 py-2.5 text-[0.8125rem] text-white/65 transition-colors hover:border-white/40 hover:text-white disabled:opacity-50"
                          >
                            {exporting ? "Zipping…" : `Stems ZIP (${Object.keys(lastResult.stems_keys).length})`}
                          </button>
                        )}
                      </div>
                    </section>
                  )}

                  {/* The feature's own tools. */}
                  <section className="mb-6" aria-label={`${title} tools`}>
                    <div className="mb-3 flex items-end justify-between gap-3">
                      <div>
                        <p className="eyebrow">Tools on this page</p>
                        <p className="mt-1 text-sm text-white/50">{subtitle}</p>
                      </div>
                      <span className="hidden font-mono text-[0.58rem] uppercase tracking-[0.16em] text-white/35 sm:inline">
                        1 · choose an action
                      </span>
                    </div>
                    {children}
                  </section>

                  <div className="mb-2 flex items-end justify-between gap-3">
                    <p className="eyebrow">Activity</p>
                    <span className="font-mono text-[0.58rem] uppercase tracking-[0.16em] text-white/35">
                      2 · review
                    </span>
                  </div>
                  <Conversation history={history} busy={processing} />
                </>
              )}

              {(project.error || errorMsg) && (
                <p className="mt-3 text-sm text-red-400">{project.error || errorMsg}</p>
              )}
            </div>

            <div className="shrink-0 border-t border-white/10 bg-black/90 p-3 backdrop-blur-sm sm:p-4">
              {notice && (
                <p role="alert" className="mb-2 flex items-start gap-2 border border-white/18 bg-white/[0.03] px-3 py-2 text-sm text-white/65">
                  <span aria-hidden className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-white/50" />
                  {notice}
                </p>
              )}
              <p className="mb-2 text-[0.8125rem] text-white/55">
                {project.file
                  ? "Describe an edit, or use the tools above. Enter sends · mic for voice input"
                  : "Upload a file first, then describe the edit you want"}
              </p>
              <div className="flex flex-col gap-2.5 sm:flex-row sm:gap-3">
                <input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={project.file ? "e.g. boost the vocals, trim 0:30–1:15, make it darker…" : "Upload a file first"}
                  className="w-full flex-1 rounded-none border border-white/20 bg-white/[0.03] px-4 py-3.5 text-base text-white placeholder-white/35 outline-none transition-colors focus:border-white/60"
                  disabled={busy}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void handleProcess();
                    }
                  }}
                />
                <button
                  onClick={() => void handleProcess()}
                  disabled={!prompt.trim() || busy}
                  className="shrink-0 rounded-none bg-[#f2f1ec] px-6 py-3.5 text-base font-medium text-black transition-colors hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {processing ? "Working…" : "Process"}
                </button>
              </div>
            </div>
          </main>
        </div>
      </div>
    </FeaturePromptContext.Provider>
  );
}
