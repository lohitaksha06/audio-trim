"use client";

import { useState } from "react";
import { processAudio, type ProcessResponse } from "@/services/api";

interface StemMixerProps {
  /** Server audio_path / storage key that the mix chains onto. */
  audioPath: string;
  /** Page updates its chaining state + output panel from the result. */
  onResult: (res: ProcessResponse, label: string) => void;
  disabled?: boolean;
}

const STEMS = [
  { id: "vocals", label: "Vocals", hint: "Presence 1–6 kHz" },
  { id: "drums", label: "Drums", hint: "Punch <200 Hz + snap" },
  { id: "bass", label: "Bass", hint: "Weight <250 Hz" },
  { id: "other", label: "Other", hint: "Everything else" },
] as const;

type StemId = (typeof STEMS)[number]["id"];

const PRESETS: { name: string; gains: Record<StemId, number> }[] = [
  { name: "Balanced", gains: { vocals: 0, drums: 0, bass: 0, other: 0 } },
  { name: "Drums punch", gains: { vocals: -2, drums: 6, bass: 2, other: 0 } },
  { name: "Vocal up", gains: { vocals: 6, drums: -2, bass: 0, other: -2 } },
  { name: "Bass weight", gains: { vocals: 0, drums: 2, bass: 6, other: -2 } },
  { name: "Karaoke", gains: { vocals: -12, drums: 2, bass: 2, other: 2 } },
  { name: "Acapella-ish", gains: { vocals: 6, drums: -8, bass: -8, other: -6 } },
];

export default function StemMixer({ audioPath, onResult, disabled }: StemMixerProps) {
  const [gains, setGains] = useState<Record<StemId, number>>({ vocals: 0, drums: 0, bass: 0, other: 0 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const set = (id: StemId, v: number) => setGains((g) => ({ ...g, [id]: v }));

  const apply = async () => {
    setBusy(true);
    setError("");
    try {
      const parts = STEMS.map((s) => `${s.id} ${gains[s.id] >= 0 ? "+" : ""}${gains[s.id].toFixed(0)}dB`);
      const prompt = `balance the mix: ${parts.join(", ")}`;
      const res = await processAudio(audioPath, prompt);
      const desc = STEMS.filter((s) => gains[s.id] !== 0)
        .map((s) => `${s.label} ${gains[s.id] > 0 ? "+" : ""}${gains[s.id]}dB`)
        .join(", ");
      onResult(res, desc ? `Mix (${desc})` : "Mix (balanced)");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Mix failed");
    } finally {
      setBusy(false);
    }
  };

  const touched = STEMS.some((s) => gains[s.id] !== 0);

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-4">
      <div className="mb-1 flex items-center justify-between">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-white/50">Prioritize sounds</h4>
        <span className="text-[10px] text-white/25">Demucs stems when possible, EQ-balance fallback</span>
      </div>
      <p className="mb-3 text-[11px] leading-relaxed text-white/35">
        Drums louder, vocals lower? Move the faders, then Apply. Each mix builds on your latest output.
      </p>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {STEMS.map((s) => (
          <div key={s.id} className="rounded-xl border border-white/5 bg-black/30 p-3">
            <div className="text-xs font-medium text-white/80">{s.label}</div>
            <div className="text-[10px] text-white/25">{s.hint}</div>
            <input
              type="range"
              min={-12}
              max={12}
              step={1}
              value={gains[s.id]}
              onChange={(e) => set(s.id, Number(e.target.value))}
              className="mt-2 w-full accent-neon-purple"
            />
            <div className="mt-1 text-center font-mono text-xs text-white/70">
              {gains[s.id] > 0 ? `+${gains[s.id]}` : gains[s.id]} dB
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {PRESETS.map((p) => (
          <button
            key={p.name}
            type="button"
            onClick={() => setGains({ ...p.gains })}
            className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-white/50 transition-colors hover:border-neon-purple/40 hover:text-neon-purple"
          >
            {p.name}
          </button>
        ))}
      </div>

      <button
        type="button"
        onClick={apply}
        disabled={!touched || busy || disabled}
        className="mt-3 w-full rounded-lg bg-neon-purple/15 px-3 py-2 text-xs font-medium text-neon-purple transition-colors hover:bg-neon-purple/25 disabled:opacity-40"
      >
        {busy ? "Mixing…" : "Apply mix"}
      </button>
      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
    </div>
  );
}
