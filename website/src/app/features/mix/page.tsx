"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import AudioPreview from "@/components/AudioPreview";
import ManualEditor, { type LocalEditResult } from "@/components/ManualEditor";
import TrackMixer from "@/components/TrackMixer";
import StemImport from "@/components/StemImport";
import StemMixer from "@/components/StemMixer";
import JamMixer from "@/components/JamMixer";
import OptimizePanel from "@/components/OptimizePanel";
import { useProject } from "@/components/ProjectProvider";
import {
  downloadUrl,
  isBackendConfigured,
  processAudio,
  type ProcessResponse,
} from "@/services/api";
import {
  analyzeLocalAsync,
  durationOf,
  parseLocalPrompt,
  runLocalPlanAsync,
  type LocalAnalysis,
} from "@/lib/audio";
import {
  downloadBlob,
  loadJson,
  removeJson,
  safeFilename,
  saveJson,
} from "@/lib/storage";

type ChatResult = ProcessResponse & { local_url?: string | null; summary?: string };

interface SavedMix {
  history: string[];
  edits: number;
  lastAction: string;
}

const EMPTY: SavedMix = { history: [], edits: 0, lastAction: "" };
const STORAGE_KEY = "audelle:mix:state";

const TIPS = [
  ["Waveform edit", "Trim, cut, fade and speed — the same editor as Manual Editor."],
  ["4-band mixer", "Pull vocals, drums, bass and air up or down with live preview."],
  ["Import a stem", "Add your singer's take on top of the track at any level."],
];

function db(value: number): string {
  if (value <= 1e-6) return "−∞";
  return `${(20 * Math.log10(value)).toFixed(1)} dB`;
}

export default function MixPage() {
  const project = useProject();
  const localMode = !isBackendConfigured;
  const [history, setHistory] = useState<string[]>([]);
  const [edits, setEdits] = useState(0);
  const [lastAction, setLastAction] = useState("");
  const [lastResult, setLastResult] = useState<ChatResult | null>(null);
  const [resultBlob, setResultBlob] = useState<Blob | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [notice, setNotice] = useState("");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [editorBusy, setEditorBusy] = useState(false);
  const [progress, setProgress] = useState<{ fraction: number; label: string } | null>(null);
  const [stats, setStats] = useState<LocalAnalysis | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const resultUrlRef = useRef<string | null>(null);
  const restoredRef = useRef(false);
  const busyRef = useRef(false);

  useEffect(() => {
    const saved = loadJson<SavedMix>(STORAGE_KEY, EMPTY);
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setHistory(Array.isArray(saved.history) ? saved.history : []);
    setEdits(Number.isFinite(saved.edits) ? saved.edits : 0);
    setLastAction(saved.lastAction ?? "");
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    saveJson(STORAGE_KEY, { history, edits, lastAction } satisfies SavedMix);
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

  // Local "mix doctor" measurements. Deferred so the page paints first.
  useEffect(() => {
    if (!project.audio) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setStats(null);
      return;
    }
    let cancelled = false;
    const audio = project.audio;
    const timer = window.setTimeout(() => {
      void analyzeLocalAsync(audio).then((measured) => {
        if (!cancelled) setStats(measured);
      }).catch(() => {
        /* measurements are optional */
      });
    }, 40);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [project.audio]);

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
    setPrompt("");
    setStats(null);
    restoredRef.current = false;
    removeJson(STORAGE_KEY);
  }, [clearResultUrl]);

  const applyLocalResult = useCallback(
    (payload: { audio: Parameters<typeof project.replaceAudio>[0]; blob: Blob; summary: string }, label: string) => {
      clearResultUrl();
      const url = URL.createObjectURL(payload.blob);
      resultUrlRef.current = url;
      setResultBlob(payload.blob);
      setLastResult({
        intent: "local_mix",
        params: {},
        raw_prompt: label,
        metadata: { local: true },
        download_key: null,
        local_url: url,
        summary: payload.summary,
      });
      setLastAction(payload.summary);
      setEdits((value) => value + 1);
      restoredRef.current = true;
      project.replaceAudio(
        payload.audio,
        payload.blob,
        safeFilename(`${project.fileName ?? "audelle"}-mix`, "wav"),
      );
      const entry = payload.summary.toLowerCase().startsWith(label.toLowerCase())
        ? payload.summary
        : `${label} — ${payload.summary}`;
      setHistory((prev) => [entry, ...prev].slice(0, 30));
      setErrorMsg("");
    },
    [clearResultUrl, project],
  );

  const onLocalEdit = (result: LocalEditResult, label: string) => {
    applyLocalResult(
      { audio: result.audio, blob: result.blob, summary: result.summary },
      label,
    );
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

  const applyLocalPrompt = async (text: string) => {
    const raw = text.trim();
    if (!raw || busyRef.current) return;
    const base = project.audio ?? project.original;
    if (!base) {
      setNotice("The file is still loading.");
      return;
    }
    const plan = parseLocalPrompt(raw, durationOf(base));
    if (plan.unsupported || plan.intent === "unknown") {
      setNotice(plan.unsupported ?? "That edit is not available in browser mode.");
      return;
    }
    busyRef.current = true;
    setBusy(true);
    setErrorMsg("");
    setNotice("");
    setPrompt("");
    setProgress({ fraction: 0, label: "Starting" });
    try {
      const result = await runLocalPlanAsync(base, plan, (fraction, label) => {
        setProgress({ fraction, label });
      });
      if (result.metadata.info_only) {
        setHistory((prev) => [`${result.plan.summary}`, ...prev].slice(0, 30));
      } else {
        applyLocalResult(
          { audio: result.audio, blob: result.blob, summary: result.plan.summary },
          `“${raw}”`,
        );
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Processing failed");
    } finally {
      busyRef.current = false;
      setBusy(false);
      setProgress(null);
    }
  };

  const runServerPrompt = async (text: string) => {
    const raw = text.trim();
    const source = lastResult?.download_key ?? project.upload?.audio_path;
    if (!raw || !source) return;
    setBusy(true);
    setErrorMsg("");
    try {
      const result = await processAudio(source, raw);
      onServerResult(result, `“${raw}”`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Processing failed");
    } finally {
      setBusy(false);
    }
  };

  const activePath = lastResult?.download_key ?? project.upload?.audio_path;
  const playSrc = localMode
    ? project.objectUrl
    : lastResult?.download_key
      ? downloadUrl(lastResult.download_key)
      : project.objectUrl;

  const handleDownload = () => {
    if (resultBlob) {
      downloadBlob(resultBlob, safeFilename(`${project.fileName ?? "audelle"}-mix`, "wav"));
      return;
    }
    if (project.currentBlob && edits > 0) {
      downloadBlob(project.currentBlob, safeFilename(`${project.fileName ?? "audelle"}-mix`, "wav"));
      return;
    }
    if (lastResult?.download_key) {
      window.open(downloadUrl(lastResult.download_key), "_blank", "noopener");
      return;
    }
    setNotice("No rendered file yet. Move a fader or make a cut first.");
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
            <div className="mx-auto w-full max-w-4xl">
              <div className="mb-6 border-l-2 border-[#ef7fa6] pl-3">
                <p className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-[#f4a9c3]">
                  Workspace · mixing
                </p>
                <h1 className="mt-0.5 font-display text-3xl font-bold lowercase tracking-[-0.03em] text-[#f2f1ec]">
                  mix lab
                </h1>
                <p className="mt-1 max-w-2xl text-[0.9375rem] text-white/55">
                  A hands-on mixer: edit the waveform, move four band faders, import your own vocal,
                  and render a WAV. No upload leaves this device.
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
                  {project.error || errorMsg || "Decoding and measuring on this device…"}
                </p>
              )}
              <div className="mt-8 grid gap-3 sm:grid-cols-3">
                {TIPS.map(([title, body]) => (
                  <div key={title} className="border border-white/10 bg-white/[0.02] p-3">
                    <p className="text-sm text-white/85">{title}</p>
                    <p className="mt-1 text-xs leading-relaxed text-white/50">{body}</p>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-xs leading-relaxed text-white/40">
                4-band mixing is frequency splitting, not AI stem isolation. True Demucs stems and
                tempo/key matching need the backend.
              </p>
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
              <div className="border-l-2 border-[#ef7fa6] pl-3">
                <p className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-[#f4a9c3]">
                  Workspace · mixing
                </p>
                <h1 className="mt-0.5 font-display text-2xl font-bold lowercase tracking-[-0.03em] text-[#f2f1ec] sm:text-3xl">
                  mix lab
                </h1>
                <p className="text-[0.9375rem] text-white/55">
                  Wave edit · 4-band faders · imported stem · local render.
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
                <button
                  onClick={handleDownload}
                  disabled={edits === 0}
                  className="rounded-lg bg-[#f2f1ec] px-3 py-1.5 text-xs font-semibold text-black hover:bg-white disabled:cursor-not-allowed disabled:opacity-35"
                >
                  Download WAV
                </button>
              </div>
            </div>
            {(busy || editorBusy) && (
              <div className="mt-2 flex items-center gap-2 border border-white/10 bg-white/[0.025] px-2.5 py-1.5 text-[0.72rem] text-white/55">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-white/70" />
                Rendering on this device — the sidebar still works.
              </div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-4 sm:p-6">
            <div className="mx-auto grid w-full max-w-6xl gap-4 lg:grid-cols-[1.45fr_0.8fr]">
              <div className="space-y-4">
                <div className="flex items-center gap-3 border border-white/8 bg-white/[0.02] p-3">
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-white">{project.file.name}</p>
                    <p className="text-xs text-white/55">
                      {edits === 0 ? "Original" : `Render ${edits} · ${lastAction || "mixed"}`}
                      {project.loading ? " · decoding…" : ""}
                    </p>
                  </div>
                  {stats && (
                    <span className="hidden font-mono text-[0.65rem] text-white/40 sm:inline">
                      {stats.bpm ? `${stats.bpm.toFixed(0)} BPM · ` : ""}
                      {Math.round(stats.brightnessHz)} Hz bright
                    </span>
                  )}
                </div>

                {playSrc ? (
                  <>
                    <div>
                      <div className="mb-2 flex items-end justify-between">
                        <p className="eyebrow">1 · Wave edit</p>
                        <span className="font-mono text-[0.58rem] uppercase tracking-[0.16em] text-white/35">
                          drag / trim / fade
                        </span>
                      </div>
                      <ManualEditor
                        src={playSrc}
                        audioPath={activePath}
                        localAudio={localMode ? project.audio : null}
                        onResult={onServerResult}
                        onLocalResult={onLocalEdit}
                        onBusyChange={setEditorBusy}
                        disabled={project.loading}
                      />
                    </div>

                    <div>
                      <div className="mb-2 flex items-end justify-between">
                        <p className="eyebrow">2 · Balance</p>
                        <span className="font-mono text-[0.58rem] uppercase tracking-[0.16em] text-white/35">
                          live preview
                        </span>
                      </div>
                      {localMode ? (
                        project.audio ? (
                          <TrackMixer
                            src={playSrc}
                            audio={project.audio}
                            onApply={(payload) => applyLocalResult(payload, "4-band mix")}
                            onBusyChange={setBusy}
                            disabled={project.loading || busy}
                          />
                        ) : null
                      ) : activePath ? (
                        <StemMixer audioPath={activePath} onResult={onServerResult} disabled={busy} />
                      ) : null}
                    </div>
                  </>
                ) : (
                  <div className="border border-white/10 p-6 text-sm text-white/55">
                    The audio is still loading. If this stays empty, upload the file again.
                  </div>
                )}

                {history.length > 0 && (
                  <div className="border border-white/8 bg-white/[0.01] p-3">
                    <p className="eyebrow mb-1.5">Mix history</p>
                    <ul className="space-y-1">
                      {history.map((item, index) => (
                        <li key={`${item}-${index}`} className="text-xs text-white/70">
                          ✓ {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <aside className="space-y-4">
                {lastResult && playSrc && (
                  <div className="border border-white/12 bg-white/[0.025] p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <p className="eyebrow">Latest render</p>
                      <span className="font-mono text-[0.55rem] uppercase tracking-[0.14em] text-white/35">
                        WAV
                      </span>
                    </div>
                    <AudioPreview src={playSrc} height={48} />
                    <button
                      onClick={handleDownload}
                      className="mt-2 w-full rounded-none border border-white/25 bg-white/[0.07] px-3 py-2.5 text-sm font-medium text-white/85 hover:border-white/50"
                    >
                      Download WAV
                    </button>
                    <p className="mt-2 text-xs leading-relaxed text-white/50">
                      {lastResult.summary || lastResult.raw_prompt || "Rendered mix."}
                    </p>
                  </div>
                )}

                <div>
                  <div className="mb-2 flex items-end justify-between">
                    <p className="eyebrow">3 · Add a voice</p>
                    <span className="font-mono text-[0.58rem] uppercase tracking-[0.16em] text-white/35">
                      your stem
                    </span>
                  </div>
                  {localMode ? (
                    project.audio ? (
                      <StemImport
                        base={project.audio}
                        onApply={(payload) => applyLocalResult(payload, "Stem import")}
                        onBusyChange={setBusy}
                        disabled={project.loading || busy}
                      />
                    ) : null
                  ) : activePath ? (
                    <JamMixer audioPath={activePath} onResult={onServerResult} disabled={busy} />
                  ) : null}
                </div>

                {localMode && project.audio && (
                  <div className="border border-white/10 bg-white/[0.02] p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <p className="eyebrow">4 · Mix doctor</p>
                      <span className="font-mono text-[0.55rem] uppercase tracking-[0.14em] text-white/35">
                        local
                      </span>
                    </div>
                    {stats ? (
                      <dl className="mb-3 grid grid-cols-2 gap-1.5 text-[0.7rem]">
                        <div className="border border-white/8 bg-black/25 px-2 py-1.5">
                          <dt className="text-white/40">Peak</dt>
                          <dd className="font-mono text-white/80">{db(stats.peak)}</dd>
                        </div>
                        <div className="border border-white/8 bg-black/25 px-2 py-1.5">
                          <dt className="text-white/40">RMS</dt>
                          <dd className="font-mono text-white/80">{db(stats.rms)}</dd>
                        </div>
                        <div className="border border-white/8 bg-black/25 px-2 py-1.5">
                          <dt className="text-white/40">Brightness</dt>
                          <dd className="font-mono text-white/80">{Math.round(stats.brightnessHz)} Hz</dd>
                        </div>
                        <div className="border border-white/8 bg-black/25 px-2 py-1.5">
                          <dt className="text-white/40">Tempo</dt>
                          <dd className="font-mono text-white/80">{stats.bpm ? `${stats.bpm.toFixed(0)} BPM` : "—"}</dd>
                        </div>
                      </dl>
                    ) : (
                      <p className="mb-3 text-xs text-white/45">Measuring the current render…</p>
                    )}
                    <div className="flex flex-wrap gap-1.5">
                      {["Normalize the volume", "Boost the vocals", "Make the drums louder", "Make it darker"].map((tip) => (
                        <button
                          key={tip}
                          type="button"
                          onClick={() => void applyLocalPrompt(tip)}
                          disabled={busy}
                          className="rounded-full border border-white/12 bg-white/[0.03] px-2.5 py-1 text-[0.7rem] text-white/70 hover:border-white/30 hover:text-white disabled:opacity-40"
                        >
                          {tip}
                        </button>
                      ))}
                    </div>
                    <div className="mt-2 flex gap-2">
                      <input
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            void applyLocalPrompt(prompt);
                          }
                        }}
                        placeholder="Ask for a local edit…"
                        className="min-w-0 flex-1 border border-white/15 bg-black/40 px-2.5 py-2 text-xs text-white placeholder-white/30 outline-none focus:border-white/40"
                        disabled={busy}
                      />
                      <button
                        type="button"
                        onClick={() => void applyLocalPrompt(prompt)}
                        disabled={!prompt.trim() || busy}
                        className="shrink-0 border border-white/20 bg-white/[0.06] px-3 py-2 text-xs text-white/80 hover:border-white/40 disabled:opacity-40"
                      >
                        {busy ? "…" : "Go"}
                      </button>
                    </div>
                  </div>
                )}

                {!localMode && activePath && (
                  <OptimizePanel
                    audioPath={activePath}
                    onApplyPrompt={(value) => void runServerPrompt(value)}
                  />
                )}

                {progress && (
                  <div className="border border-white/12 bg-white/[0.03] p-3">
                    <div className="flex justify-between text-[0.7rem] text-white/60">
                      <span>{progress.label}</span>
                      <span>{Math.round(progress.fraction * 100)}%</span>
                    </div>
                    <div className="mt-1 h-1 overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-white/70 transition-[width]"
                        style={{ width: `${Math.round(progress.fraction * 100)}%` }}
                      />
                    </div>
                  </div>
                )}

                {(project.error || errorMsg) && <p className="text-xs text-red-400">{project.error || errorMsg}</p>}

                <p className="text-[0.7rem] leading-relaxed text-white/35">
                  The 4-band mixer is frequency splitting, not AI stem isolation. Imported stems are
                  summed locally without tempo/key matching. Both are real audio renders, not previews.
                </p>
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
