"use client";

import FeatureLayout from "../FeatureLayout";

export default function FilmPage() {
  return (
    <FeatureLayout title="Film & Video" subtitle="Clean dialogue, match room tone, separate stems for post-production">
      <div className="space-y-5">
        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Audio Layers</h3>
          <div className="space-y-2">
            {[
              { label: "Dialogue", level: 85, color: "from-blue-500 to-blue-400" },
              { label: "Music", level: 60, color: "from-purple-500 to-purple-400" },
              { label: "Sound Effects", level: 45, color: "from-amber-500 to-amber-400" },
              { label: "Ambience", level: 30, color: "from-green-500 to-green-400" },
            ].map((layer) => (
              <div key={layer.label} className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
                <div className="flex justify-between mb-2">
                  <span className="text-sm text-white/70">{layer.label}</span>
                  <span className="text-xs text-white/30">{layer.level}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                  <div className={`h-full rounded-full bg-gradient-to-r ${layer.color}`} style={{ width: `${layer.level}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Quick Actions</h3>
          <div className="space-y-2">
            {[
              { label: "Clean dialogue", prompt: "Clean up the dialogue and remove background noise" },
              { label: "Match room tone", prompt: "Match the room tone across all dialogue clips" },
              { label: "Separate stems", prompt: "Separate dialogue, music, and sound effects into individual stems" },
              { label: "ADR sync", prompt: "Sync the replacement dialogue to match lip movements" },
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
