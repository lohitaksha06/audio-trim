"use client";

import { useState } from "react";
import { optimizeAudio, type OptimizeResponse } from "@/services/api";

interface OptimizePanelProps {
  /** Server audio_path / storage key to analyze. */
  audioPath: string;
  /** User tapped a tip's Apply — parent fills the prompt box / runs it. */
  onApplyPrompt: (prompt: string) => void;
}

const SEV_STYLE: Record<string, string> = {
  high: "border-red-400/20 bg-red-500/[0.06] text-red-300",
  medium: "border-amber-400/20 bg-amber-500/[0.06] text-amber-300",
  low: "border-white/10 bg-white/[0.03] text-white/60",
  good: "border-green-400/20 bg-green-500/[0.06] text-green-300",
};

export default function OptimizePanel({ audioPath, onApplyPrompt }: OptimizePanelProps) {
  const [report, setReport] = useState<OptimizeResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    setBusy(true);
    setError("");
    try {
      setReport(await optimizeAudio(audioPath));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-4">
      <div className="mb-1 flex items-center justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-white/50">Optimize my song</h4>
        {report && (
          <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 font-mono text-[11px] text-white/60">
            score {report.score}/100
          </span>
        )}
      </div>
      <p className="mb-3 text-[11px] leading-relaxed text-white/35">
        Mix Doctor measures loudness, clipping, dynamics and spectral balance, then tells you exactly
        what to fix — tap Apply on any tip to run its one-click prompt.
      </p>

      {!report ? (
        <button
          type="button"
          onClick={run}
          disabled={busy}
          className="w-full rounded-lg bg-neon-blue/15 px-3 py-2 text-xs font-medium text-neon-blue transition-colors hover:bg-neon-blue/25 disabled:opacity-40"
        >
          {busy ? "Listening…" : "Check my mix"}
        </button>
      ) : (
        <div className="space-y-2">
          <div className="grid grid-cols-4 gap-1.5 text-center">
            {[
              [`Peak ${report.summary.peak.toFixed(2)}`, "peak"],
              [`RMS ${report.summary.rms.toFixed(3)}`, "rms"],
              [`Crest ${report.summary.crest_db}dB`, "dyn"],
              [`Dyn ${report.summary.dynamics_db}dB`, "range"],
            ].map(([v, k]) => (
              <div key={k} className="rounded-lg bg-black/30 px-1 py-1.5 font-mono text-[10px] text-white/50">{v}</div>
            ))}
          </div>
          {report.tips.map((t, i) => (
            <div key={i} className={`rounded-xl border p-3 ${SEV_STYLE[t.severity] ?? SEV_STYLE.low}`}>
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold">{t.title}</span>
                <span className="shrink-0 rounded-full bg-black/30 px-2 py-0.5 text-[10px] uppercase tracking-wider opacity-70">{t.severity}</span>
              </div>
              <p className="mt-1 text-[11px] leading-relaxed opacity-80">{t.detail}</p>
              <button
                type="button"
                onClick={() => onApplyPrompt(t.fix_prompt)}
                className="mt-2 w-full rounded-lg bg-black/30 px-2 py-1.5 text-left text-[11px] hover:bg-black/50 transition-colors"
              >
                Apply: <span className="underline">“{t.fix_prompt}”</span>
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={run}
            disabled={busy}
            className="w-full rounded-lg border border-white/10 px-3 py-1.5 text-[11px] text-white/40 transition-colors hover:border-white/20 disabled:opacity-40"
          >
            {busy ? "Listening…" : "Re-check after my fix"}
          </button>
        </div>
      )}
      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}
