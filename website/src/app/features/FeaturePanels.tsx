"use client";

import type { ReactNode } from "react";
import { isBackendConfigured } from "@/services/api";
import { localSupport } from "@/lib/audio";
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
}: {
  title: string;
  body: string;
}) {
  const ctx = useFeaturePrompt();
  if (!ctx?.hasFile) {
    return (
      <div className="rounded-xl border border-white/12 bg-white/[0.02] p-4">
        <p className="text-[0.9375rem] text-white/84">{title}</p>
        <p className="mt-1 text-[0.9375rem] leading-relaxed text-white/58">{body}</p>
      </div>
    );
  }
  return (
    <div className="rounded-xl border border-white/12 bg-white/[0.02] p-4">
      <p className="text-[0.9375rem] text-white/84">{title}</p>
      <p className="mt-1 text-[0.9375rem] leading-relaxed text-white/58">{body}</p>
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
  // In browser-only builds, mark the buttons that cannot run here. They stay
  // clickable so the chat can explain the local alternative instead of a dead
  // control with no feedback.
  const support = !isBackendConfigured ? localSupport(prompt) : null;
  const serverOnly = support !== null && !support.supported;
  return (
    <button
      onClick={() => ctx?.triggerPrompt?.(prompt)}
      disabled={blocked}
      title={
        !ctx?.hasFile
          ? "Upload a file first"
          : serverOnly
            ? support?.reason ?? "This one needs the backend"
            : hint
      }
      className={`w-full rounded-xl border p-3.5 text-left text-[0.9375rem] transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${
        tone === "accent"
          ? "border-neon-purple/25 bg-neon-purple/[0.06] text-white/90 hover:border-neon-purple/45 hover:bg-neon-purple/10"
          : "border-white/10 bg-white/[0.02] text-white/84 hover:border-neon-blue/35 hover:bg-neon-blue/5 hover:text-neon-blue"
      }`}
    >
      <span className="flex items-center justify-between gap-2">
        <span>{label}</span>
        {serverOnly && (
          <span className="shrink-0 rounded-full border border-amber-300/30 bg-amber-300/10 px-2 py-0.5 text-[0.6875rem] font-medium uppercase tracking-wide text-amber-200">
            Server
          </span>
        )}
      </span>
      {hint && <span className="mt-1 block text-[0.8125rem] leading-snug text-white/58">{hint}</span>}
      {serverOnly && (
        <span className="mt-1 block text-[0.75rem] leading-snug text-amber-200/70">
          Needs the backend — click to see the local alternative.
        </span>
      )}
    </button>
  );
}

export function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-white/[0.02] px-3.5 py-2.5">
      <span className="text-[0.9375rem] text-white/78">{label}</span>
      <span className="text-[0.9375rem] text-white/90">{value}</span>
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
            className={`max-w-[85%] rounded-xl px-4 py-3 text-[0.9375rem] leading-relaxed ${
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
        <div className="flex items-center gap-2 rounded-xl border border-neon-blue/25 bg-neon-blue/10 px-4 py-3 text-[0.9375rem] text-neon-blue">
          <span className="h-3 w-3 animate-spin rounded-full border border-neon-blue border-t-transparent" />
          Processing…
        </div>
      )}
    </div>
  );
}
