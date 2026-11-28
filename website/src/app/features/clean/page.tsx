"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Nav from "@/components/Nav";
import Sidebar from "@/components/Sidebar";
import FileUpload from "@/components/FileUpload";
import {
  TARGET_PEAK_DB,
  applyCleanPlan,
  diagnose,
  planClean,
  type CleanDiagnosis,
  type CleanResult,
} from "@/lib/audio/clean";
import { decodeFile, durationOf, encodeWav } from "@/lib/audio";
import type { LocalAudio } from "@/lib/audio/types";

const MODELS = [
  {
    name: "Diagnoser",
    job: "Predicts noise level (SNR) and the clipped fraction",
    data: "VoiceBank-DEMAND clean speech + DEMAND noise, mixed at known SNRs",
    metric: "SNR error (dB), clipped-% error",
    where: "CPU",
  },
  {
    name: "Denoiser",
    job: "Conv-GRU mask that removes stationary noise",
    data: "The same mixtures, cleaned against their clean reference",
    metric: "SI-SDR, STOI, PESQ",
    where: "GPU",
  },
  {
    name: "Declipper",
    job: "Small U-Net that restores flat-topped peaks",
    data: "Clean speech hard-clipped at known levels",
    metric: "SI-SDR vs cubic-spline baseline",
    where: "GPU",
  },
  {
    name: "Router",
    job: "Decision tree over the diagnoser that picks the repair chain",
    data: "Labelled by oracle search over the candidate pipelines",
    metric: "Agreement with the oracle",
    where: "CPU, seconds",
  },
];

function formatDb(value: number): string {
  if (!Number.isFinite(value) || value <= -120) return "silent";
  return `${value.toFixed(1)} dB`;
}

function formatPercent(value: number): string {
  if (value <= 0) return "none";
  if (value < 0.01) return `${value.toFixed(3)}%`;
  return `${value.toFixed(2)}%`;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** Peak-envelope waveform. Peaks are computed once per audio object. */
function Waveform({ audio, tone = "before" }: { audio: LocalAudio; tone?: "before" | "after" }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    const cssWidth = Math.max(240, parent?.clientWidth ?? 640);
    const cssHeight = 96;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(cssWidth * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    canvas.style.width = `${cssWidth}px`;
    canvas.style.height = `${cssHeight}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const channels = audio.channels;
    const length = channels[0]?.length ?? 0;
    if (length === 0) return;

    const columns = Math.max(1, Math.floor(cssWidth));
    const step = length / columns;
    const mid = cssHeight / 2;
    const color = tone === "after" ? "242, 241, 236" : "255, 255, 255";

    ctx.strokeStyle = `rgba(${color}, 0.10)`;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(cssWidth, mid);
    ctx.stroke();

    ctx.fillStyle = `rgba(${color}, ${tone === "after" ? 0.85 : 0.45})`;
    for (let c = 0; c < columns; c++) {
      const start = Math.floor(c * step);
      const end = Math.min(length, Math.floor((c + 1) * step));
      let peak = 0;
      for (let i = start; i < end; i++) {
        for (let ch = 0; ch < channels.length; ch++) {
          const v = Math.abs(channels[ch][i]);
          if (v > peak) peak = v;
        }
      }
      const h = Math.max(1, Math.min(mid, peak * mid));
      ctx.fillRect(c, mid - h, 1, h * 2);
    }
  }, [audio, tone]);

  return <canvas ref={canvasRef} className="block w-full" aria-hidden="true" />;
}

function Meter({
  label,
  severity,
  value,
  note,
}: {
  label: string;
  severity: number;
  value: string;
  note: string;
}) {
  const percent = Math.round(Math.max(0, Math.min(1, severity)) * 100);
  return (
    <div className="border border-white/10 bg-white/[0.02] p-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-[0.6rem] uppercase tracking-[0.18em] text-white/45">{label}</span>
        <span className="text-sm text-white/85">{value}</span>
      </div>
      <div className="mt-2 h-1 overflow-hidden bg-white/10">
        <div className="h-full bg-[#f2f1ec]" style={{ width: `${percent}%` }} />
      </div>
      <p className="mt-2 text-[0.75rem] leading-snug text-white/45">{note}</p>
    </div>
  );
}

function Metric({ label, before, after }: { label: string; before: string; after: string | null }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/8 py-2 text-sm last:border-b-0">
      <span className="text-white/55">{label}</span>
      <span className="flex items-center gap-2 font-mono text-[0.75rem] text-white/60">
        {before}
        {after !== null && (
          <>
            <span aria-hidden className="text-white/25">→</span>
            <span className="text-white/90">{after}</span>
          </>
        )}
      </span>
    </div>
  );
}

export default function CleanPage() {
  const [file, setFile] = useState<File | null>(null);
  const [original, setOriginal] = useState<LocalAudio | null>(null);
  const [diagnosis, setDiagnosis] = useState<CleanDiagnosis | null>(null);
  const [result, setResult] = useState<CleanResult | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [view, setView] = useState<"before" | "after">("after");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{ fraction: number; label: string } | null>(null);
  const [error, setError] = useState("");

  const urlsRef = useRef<string[]>([]);

  const trackUrl = useCallback((url: string) => {
    urlsRef.current.push(url);
    return url;
  }, []);

  const clearUrls = useCallback(() => {
    for (const url of urlsRef.current) URL.revokeObjectURL(url);
    urlsRef.current = [];
    setOriginalUrl(null);
    setResultUrl(null);
  }, []);

  useEffect(() => {
    return () => {
      for (const url of urlsRef.current) URL.revokeObjectURL(url);
      urlsRef.current = [];
    };
  }, []);

  const reset = useCallback(() => {
    clearUrls();
    setFile(null);
    setOriginal(null);
    setDiagnosis(null);
    setResult(null);
    setProgress(null);
    setError("");
    setView("after");
  }, [clearUrls]);

  const handleFile = useCallback(
    async (picked: File) => {
      clearUrls();
      setError("");
      setResult(null);
      setProgress(null);
      setBusy(true);
      try {
        const decoded = await decodeFile(picked);
        setFile(picked);
        setOriginal(decoded);
        setDiagnosis(diagnose(decoded));
        setOriginalUrl(trackUrl(URL.createObjectURL(picked)));
        setView("before");
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "That file could not be decoded.");
      } finally {
        setBusy(false);
      }
    },
    [clearUrls, trackUrl],
  );

  const handleRepair = useCallback(() => {
    if (!original || !diagnosis || busy) return;
    setBusy(true);
    setError("");
    setProgress({ fraction: 0, label: "Starting" });
    // Yield once so the progress bar paints before the CPU-bound chain starts.
    window.setTimeout(() => {
      try {
        const plan = planClean(diagnosis);
        const repaired = applyCleanPlan(original, diagnosis, plan, (fraction, label) => {
          setProgress({ fraction, label });
        });
        const blob = encodeWav(repaired.audio);
        setResult(repaired);
        setResultUrl(trackUrl(URL.createObjectURL(blob)));
        setView("after");
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "The repair failed.");
      } finally {
        setProgress(null);
        setBusy(false);
      }
    }, 30);
  }, [original, diagnosis, busy, trackUrl]);

  const handleDownload = useCallback(() => {
    if (!result || !file) return;
    const blob = encodeWav(result.audio);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${file.name.replace(/\.[^.]+$/, "")}-repaired.wav`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 2000);
  }, [result, file]);

  const plan = diagnosis ? planClean(diagnosis) : null;
  const shown = view === "after" && result ? result.audio : original;
  const shownUrl = view === "after" && resultUrl ? resultUrl : originalUrl;
  const changed = Boolean(result);

  return (
    <>
      <Nav />
      <div className="flex min-h-screen pt-[var(--nav-h)]">
        <Sidebar onPromptSelect={() => {}} />
        <main className="min-w-0 flex-1 overflow-x-hidden">
          <header className="border-b border-white/10 px-4 py-4 sm:px-6">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-white/40">
                  Workspace · clean audio
                </p>
                <h1 className="mt-0.5 font-display text-3xl font-bold lowercase tracking-[-0.03em] text-[#f2f1ec]">
                  fix my recording.
                </h1>
                <p className="mt-1 max-w-2xl text-[0.9375rem] text-white/55">
                  Audelle measures the file first, names what is actually wrong, and repairs only the
                  faults it found. Everything runs on your device.
                </p>
              </div>
              {file && (
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    onClick={reset}
                    className="rounded-none border border-white/12 px-3 py-2 text-xs text-white/65 transition-colors hover:border-white/30 hover:text-white"
                  >
                    New file
                  </button>
                  {result && (
                    <button
                      onClick={handleDownload}
                      className="rounded-none bg-[#f2f1ec] px-4 py-2 text-xs font-semibold text-black transition-colors hover:bg-white"
                    >
                      Download WAV
                    </button>
                  )}
                </div>
              )}
            </div>
          </header>

          <div className="px-4 py-5 sm:px-6">
            {!file ? (
              <div className="mx-auto w-full max-w-3xl">
                <FileUpload onFileSelected={(picked) => void handleFile(picked)} onInvalid={setError} />
                {busy && <p className="mt-3 text-center text-sm text-white/50">Decoding and measuring…</p>}
                {error && <p className="mt-3 text-center text-sm text-red-400">{error}</p>}

                <div className="mt-10 grid gap-6 border-t border-white/8 pt-6 sm:grid-cols-3">
                  {[
                    {
                      title: "Noise",
                      body: "Measures the quiet frames to find the noise floor, and reports the gap to the loud parts.",
                    },
                    {
                      title: "Clipping",
                      body: "Finds flat-topped plateaus at full scale — the signature of an input that overloaded.",
                    },
                    {
                      title: "Level",
                      body: "Reports peak and RMS, then sets the export to a known ceiling instead of guessing.",
                    },
                  ].map((item) => (
                    <div key={item.title} className="border-t border-white/15 pt-4">
                      <h3 className="font-display text-xl font-bold text-[#f2f1ec]">{item.title}</h3>
                      <p className="mt-2 text-sm leading-relaxed text-white/55">{item.body}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
                <div className="min-w-0 space-y-5">
                  <section className="border border-white/10 bg-white/[0.02] p-4">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-sm text-white">{file.name}</p>
                        <p className="mt-0.5 font-mono text-[0.65rem] uppercase tracking-[0.14em] text-white/45">
                          {original && `${formatTime(durationOf(original))} · ${(original.sampleRate / 1000).toFixed(1)} kHz · ${original.channels.length === 1 ? "mono" : "stereo"}`}
                        </p>
                      </div>
                      <div className="flex items-center gap-1 border border-white/12 p-0.5">
                        <button
                          onClick={() => setView("before")}
                          aria-pressed={view === "before"}
                          className={`px-3 py-1.5 font-mono text-[0.65rem] uppercase tracking-[0.12em] transition-colors ${
                            view === "before" ? "bg-white/12 text-white" : "text-white/50 hover:text-white/80"
                          }`}
                        >
                          Before
                        </button>
                        <button
                          onClick={() => setView("after")}
                          disabled={!result}
                          aria-pressed={view === "after"}
                          title={result ? "Show the repaired file" : "Repair the file first"}
                          className={`px-3 py-1.5 font-mono text-[0.65rem] uppercase tracking-[0.12em] transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                            view === "after" ? "bg-white/12 text-white" : "text-white/50 hover:text-white/80"
                          }`}
                        >
                          After
                        </button>
                      </div>
                    </div>

                    {shown && (
                      <div className="mt-3 border border-white/8 bg-black/40">
                        <Waveform audio={shown} tone={view === "after" ? "after" : "before"} />
                      </div>
                    )}

                    {shownUrl && (
                      <audio key={shownUrl} src={shownUrl} controls className="mt-3 w-full" preload="metadata" />
                    )}

                    <p className="mt-2 font-mono text-[0.6rem] uppercase tracking-[0.14em] text-white/35">
                      {view === "after" && result ? "repaired render" : "original file"}
                      {view === "after" && result ? " · WAV, 16-bit" : ""}
                    </p>
                  </section>

                  {diagnosis && (
                    <section>
                      <div className="mb-2 flex items-end justify-between gap-3">
                        <h2 className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-white/45">
                          Diagnosis
                        </h2>
                        <span className="font-mono text-[0.58rem] uppercase tracking-[0.14em] text-white/30">
                          measured, not predicted
                        </span>
                      </div>
                      <div className="grid gap-2 sm:grid-cols-3">
                        <Meter
                          label="Noise"
                          severity={diagnosis.severity.noise}
                          value={diagnosis.noiseReliable ? `${diagnosis.snrDb.toFixed(0)} dB SNR` : "unmeasured"}
                          note={
                            diagnosis.noiseReliable
                              ? `Noise floor ${formatDb(diagnosis.noiseFloorDb)} at the quiet frames.`
                              : "No pauses in this file, so the noise floor cannot be separated from the signal."
                          }
                        />
                        <Meter
                          label="Clipping"
                          severity={diagnosis.severity.clipping}
                          value={formatPercent(diagnosis.clipping.percent)}
                          note={
                            diagnosis.clipping.runs > 0
                              ? `${diagnosis.clipping.runs.toLocaleString()} flat-topped stretches, longest ${diagnosis.clipping.longestRun} samples.`
                              : "No flat-topped samples found at full scale."
                          }
                        />
                        <Meter
                          label="Level"
                          severity={diagnosis.severity.level}
                          value={formatDb(diagnosis.peakDb)}
                          note={`RMS ${formatDb(diagnosis.rmsDb)} · target peak ${TARGET_PEAK_DB} dBFS.`}
                        />
                      </div>
                    </section>
                  )}

                  {plan && (
                    <section className="border border-white/10 bg-white/[0.02] p-4">
                      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
                        <h2 className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-white/45">
                          Repair chain
                        </h2>
                        <span className="font-mono text-[0.58rem] uppercase tracking-[0.14em] text-white/30">
                          chosen from this file&apos;s measurements
                        </span>
                      </div>
                      <ol className="space-y-2">
                        {plan.steps.map((step, index) => (
                          <li key={step.key} className="flex gap-3">
                            <span className="mt-0.5 font-mono text-[0.65rem] text-white/30">
                              {String(index + 1).padStart(2, "0")}
                            </span>
                            <div>
                              <p className="text-sm font-medium text-white/90">{step.label}</p>
                              <p className="text-[0.8125rem] leading-snug text-white/50">{step.detail}</p>
                            </div>
                          </li>
                        ))}
                      </ol>
                      {plan.skipped.length > 0 && (
                        <ul className="mt-3 space-y-1 border-t border-white/8 pt-3">
                          {plan.skipped.map((note) => (
                            <li key={note} className="text-[0.75rem] leading-snug text-white/35">
                              Skipped · {note}
                            </li>
                          ))}
                        </ul>
                      )}

                      <div className="mt-4 flex flex-wrap items-center gap-2">
                        <button
                          onClick={handleRepair}
                          disabled={busy}
                          className="rounded-none bg-[#f2f1ec] px-5 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-white disabled:cursor-not-allowed disabled:opacity-45"
                        >
                          {busy ? "Repairing…" : changed ? "Repair again" : "Repair this file"}
                        </button>
                        {changed && (
                          <span className="text-[0.8125rem] text-white/45">
                            {plan.steps.map((step) => step.label).join(" → ")}
                          </span>
                        )}
                      </div>

                      {progress && (
                        <div className="mt-3">
                          <div className="flex justify-between text-[0.7rem] text-white/55">
                            <span>{progress.label}</span>
                            <span>{Math.round(progress.fraction * 100)}%</span>
                          </div>
                          <div className="mt-1 h-1 overflow-hidden bg-white/10">
                            <div
                              className="h-full bg-white/70 transition-[width]"
                              style={{ width: `${Math.round(progress.fraction * 100)}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </section>
                  )}

                  {result && (
                    <section className="border border-white/10 bg-white/[0.02] p-4">
                      <h2 className="mb-2 font-mono text-[0.62rem] uppercase tracking-[0.18em] text-white/45">
                        What changed
                      </h2>
                      <Metric label="Peak" before={formatDb(result.before.peakDb)} after={formatDb(result.after.peakDb)} />
                      <Metric label="RMS" before={formatDb(result.before.rmsDb)} after={formatDb(result.after.rmsDb)} />
                      <Metric
                        label="SNR"
                        before={result.before.noiseReliable ? `${result.before.snrDb.toFixed(0)} dB` : "—"}
                        after={result.after.noiseReliable ? `${result.after.snrDb.toFixed(0)} dB` : "—"}
                      />
                      <Metric
                        label="Clipped samples"
                        before={formatPercent(result.before.clipping.percent)}
                        after={formatPercent(result.after.clipping.percent)}
                      />
                      <Metric label="Sample rate" before={`${(result.audio.sampleRate / 1000).toFixed(1)} kHz`} after={null} />
                      <p className="mt-3 text-[0.75rem] leading-snug text-white/35">
                        These are measurements of the file, not a listening score. A repair that
                        changes nothing will say so.
                      </p>
                    </section>
                  )}

                  {error && <p className="text-sm text-red-400">{error}</p>}
                </div>

                <aside className="min-w-0 space-y-4">
                  <section className="border border-white/10 bg-white/[0.02] p-4">
                    <h2 className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-white/45">
                      What this build does not do
                    </h2>
                    <ul className="mt-2 space-y-2 text-[0.8125rem] leading-snug text-white/50">
                      <li>Room echo is not measured, because there is no dereverberator to fix it.</li>
                      <li>Noise removal assumes steady noise (hiss, fan, hum), not a second voice.</li>
                      <li>Declipping interpolates the lost peaks; it cannot recover what was not recorded.</li>
                    </ul>
                  </section>

                  <section className="border border-white/10 bg-white/[0.02] p-4">
                    <h2 className="font-mono text-[0.62rem] uppercase tracking-[0.18em] text-white/45">
                      On this page
                    </h2>
                    <ul className="mt-2 space-y-1.5 text-[0.8125rem] text-white/55">
                      <li className="flex justify-between gap-2">
                        <span>Diagnosis</span>
                        <span className="font-mono text-[0.7rem] text-white/40">signal processing</span>
                      </li>
                      <li className="flex justify-between gap-2">
                        <span>Declip</span>
                        <span className="font-mono text-[0.7rem] text-white/40">Hermite</span>
                      </li>
                      <li className="flex justify-between gap-2">
                        <span>Denoise</span>
                        <span className="font-mono text-[0.7rem] text-white/40">spectral gate</span>
                      </li>
                      <li className="flex justify-between gap-2">
                        <span>Normalise</span>
                        <span className="font-mono text-[0.7rem] text-white/40">peak, {TARGET_PEAK_DB} dBFS</span>
                      </li>
                    </ul>
                  </section>
                </aside>
              </div>
            )}

            <section className="mt-12 border-t border-white/10 pt-8">
              <p className="font-mono text-[0.62rem] uppercase tracking-[0.2em] text-white/40">
                Research track
              </p>
              <h2 className="mt-2 max-w-2xl font-display text-3xl font-black lowercase leading-[0.95] tracking-[-0.03em] text-[#f2f1ec]">
                the trained version.
              </h2>
              <p className="mt-3 max-w-2xl text-[0.9375rem] leading-relaxed text-white/55">
                This page runs a signal-processing baseline that works offline. Alongside it, we are
                training four small models from scratch on VoiceBank-DEMAND and evaluating each one
                against this baseline, so the comparison is measured rather than claimed.
              </p>

              <div className="mt-6 overflow-x-auto">
                <table className="w-full min-w-[640px] border-collapse text-left">
                  <thead>
                    <tr className="border-b border-white/15">
                      {["Model", "Job", "Trained on", "Metric", "Status"].map((head) => (
                        <th
                          key={head}
                          className="pb-2 pr-4 font-mono text-[0.58rem] uppercase tracking-[0.16em] text-white/35"
                        >
                          {head}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {MODELS.map((model) => (
                      <tr key={model.name} className="border-b border-white/8 align-top">
                        <td className="py-3 pr-4 text-sm text-white/90">{model.name}</td>
                        <td className="py-3 pr-4 text-[0.8125rem] text-white/55">{model.job}</td>
                        <td className="py-3 pr-4 text-[0.8125rem] text-white/55">{model.data}</td>
                        <td className="py-3 pr-4 text-[0.8125rem] text-white/55">{model.metric}</td>
                        <td className="py-3 pr-4 font-mono text-[0.62rem] uppercase tracking-[0.12em] text-white/35">
                          not trained yet
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <p className="mt-4 max-w-2xl text-[0.8125rem] leading-snug text-white/40">
                No model card is shown until a model has numbers. When a run finishes, its score goes
                in this table next to the baseline it has to beat. Until then, the baseline above is
                what runs.
              </p>
            </section>
          </div>
        </main>
      </div>
    </>
  );
}
