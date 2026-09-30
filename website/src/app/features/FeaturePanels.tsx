"use client";

import type { ReactNode } from "react";
import { useFeaturePrompt } from "./FeaturePromptContext";

/** Section wrapper: one eyebrow label + consistent rhythm. */
export function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="eyebrow mb-3">{title}</h3>
      <div className="space-y-2">{children}</div>
    </section>
  );
}

/**
 * Honest empty/loading state. Feature pages used to render invented numbers
 * (random stem bars, fixed structure timestamps) — this replaces that with a
 * truthful prompt to upload or analyse.
 */
export function NeedsData({
  title,
  body,
  actionLabel,
}: {
  title: string;
  body: string;
  actionLabel?: string;
}) {
  const ctx = useFeaturePrompt();
  if (!ctx?.hasFile) {
    return (
      <div className="rounded-xl border border-white/12 bg-white/[0.02] p-4">
        <p className="text-sm text-white/84">{title}</p>
        <p className="mt-1 text-sm text-white/58">{body}</p>
      </div>
    );
  }
  return (
    <div className="rounded-xl border border-white/12 bg-white/[0.02] p-4">
      <p className="text-sm text-white/84">{title}</p>
      <p className="mt-1 text-sm text-white/58">{body}</p>
      {actionLabel && (
        <button
          onClick={ctx.runAnalyze}
          disabled={ctx.analyzing}
          className="mt-3 rounded-lg bg-neon-blue/15 px-4 py-2 text-sm font-medium text-neon-blue transition-colors hover:bg-neon-blue/25 disabled:opacity-50"
        >
          {ctx.analyzing ? "Analysing…" : actionLabel}
        </button>
      )}
    </div>
  );
}

/** A button that runs a real prompt through the engine when a file is loaded. */
export function ActionButton({
  label,
  prompt,
  hint,
  disabled,
  tone = "default",
}: {
  label: string;
  prompt: string;
  hint?: string;
  disabled?: boolean;
  tone?: "default" | "accent";
}) {
  const ctx = useFeaturePrompt();
  const blocked = disabled || ctx?.busy || !ctx?.hasFile;
  return (
    <button
      onClick={() => ctx?.triggerPrompt?.(prompt)}
      disabled={blocked}
      title={!ctx?.hasFile ? "Upload a file first" : hint}
      className={`w-full rounded-xl border p-3 text-left text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${
        tone === "accent"
          ? "border-neon-purple/25 bg-neon-purple/[0.06] text-white/90 hover:border-neon-purple/45 hover:bg-neon-purple/10"
          : "border-white/10 bg-white/[0.02] text-white/84 hover:border-neon-blue/35 hover:bg-neon-blue/5 hover:text-neon-blue"
      }`}
    >
      {label}
      {hint && <span className="mt-0.5 block text-xs text-white/58">{hint}</span>}
    </button>
  );
}

export function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-white/[0.02] px-3.5 py-2.5">
      <span className="text-sm text-white/78">{label}</span>
      <span className="text-sm text-white/90">{value}</span>
    </div>
  );
}

export function fmtDuration(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * The reply log. Rendered above the action list so a result is visible the
 * moment it lands, instead of below a screenful of buttons.
 */
export function Conversation({
  history,
  busy,
}: {
  history: { role: string; text: string }[];
  busy: boolean;
}) {
  if (history.length === 0 && !busy) return null;
  return (
    <div className="mb-5 space-y-2.5">
      {history.map((h, i) => (
        <div key={i} className={`flex ${h.role === "user" ? "justify-end" : "justify-start"}`}>
          <div
            className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
              h.role === "user"
                ? "bg-gradient-to-r from-neon-blue/20 to-neon-purple/20 text-white/90"
                : "border border-white/10 bg-white/[0.04] text-white/84"
            }`}
          >
            {h.text}
          </div>
        </div>
      ))}
      {busy && (
        <div className="flex items-center gap-2 rounded-xl border border-neon-blue/25 bg-neon-blue/10 px-4 py-2.5 text-sm text-neon-blue">
          <span className="h-3 w-3 animate-spin rounded-full border border-neon-blue border-t-transparent" />
          Processing…
        </div>
      )}
    </div>
  );
}
