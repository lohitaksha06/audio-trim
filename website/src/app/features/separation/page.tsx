"use client";

import FeatureLayout from "../FeatureLayout";

const STEMS = ["Vocals", "Drums", "Bass", "Guitar", "Keys", "Other"];

export default function SeparationPage() {
  return (
    <FeatureLayout title="Source Separation" subtitle="Isolate, remove, or extract individual instruments">
      <div className="space-y-5">
        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Detected Stems</h3>
          <div className="grid grid-cols-2 gap-2">
            {STEMS.map((stem) => (
              <div key={stem} className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] p-3">
                <span className="text-sm text-white/70">{stem}</span>
                <div className="h-2 w-16 rounded-full bg-white/10 overflow-hidden">
                  <div className="h-full rounded-full bg-gradient-to-r from-neon-blue to-neon-purple" style={{ width: `${50 + Math.random() * 40}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Quick Actions</h3>
          <div className="space-y-2">
            {[
              { label: "Keep vocals only", prompt: "Remove everything except the vocals" },
              { label: "Extract drums", prompt: "Give me just the drums as a stem" },
              { label: "Remove bass", prompt: "Remove the bass from this track" },
              { label: "Separate all stems", prompt: "Separate vocals, drums, bass, and other instruments into individual stems" },
            ].map((action) => (
              <button
                key={action.label}
                className="w-full text-left rounded-xl border border-white/5 bg-white/[0.02] p-3 text-sm text-white/60 hover:border-neon-blue/20 hover:text-neon-blue/80 hover:bg-neon-blue/5 transition-all"
              >
                {action.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </FeatureLayout>
  );
}
