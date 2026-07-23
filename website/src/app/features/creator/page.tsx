"use client";

import FeatureLayout from "../FeatureLayout";

export default function CreatorPage() {
  return (
    <FeatureLayout title="Content Creator" subtitle="Optimize for TikTok, Reels, Shorts, and social platforms">
      <div className="space-y-5">
        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Format Presets</h3>
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: "TikTok", ratio: "9:16", duration: "60s" },
              { label: "Reels", ratio: "9:16", duration: "30s" },
              { label: "Shorts", ratio: "9:16", duration: "60s" },
              { label: "Story", ratio: "9:16", duration: "15s" },
              { label: "Landscape", ratio: "16:9", duration: "Custom" },
              { label: "Square", ratio: "1:1", duration: "Custom" },
            ].map((preset) => (
              <button key={preset.label} className="rounded-xl border border-white/5 bg-white/[0.02] p-3 text-left hover:border-neon-blue/20 hover:bg-neon-blue/5 transition-all">
                <span className="text-sm text-white/70 block">{preset.label}</span>
                <span className="text-xs text-white/30">{preset.ratio} · {preset.duration}</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Enhancements</h3>
          <div className="space-y-2">
            {[
              { label: "Auto-captions", desc: "Generate and burn in subtitles" },
              { label: "Background music", desc: "Add music matching the energy" },
              { label: "Noise removal", desc: "Clean up background noise" },
              { label: "Voice enhance", desc: "Make voice clearer and louder" },
            ].map((item) => (
              <div key={item.label} className="flex items-center justify-between rounded-xl border border-white/5 bg-white/[0.02] p-3">
                <div>
                  <span className="text-sm text-white/70 block">{item.label}</span>
                  <span className="text-xs text-white/30">{item.desc}</span>
                </div>
                <div className="h-5 w-9 rounded-full bg-white/10 relative cursor-pointer">
                  <div className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white/40" />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Quick Actions</h3>
          <div className="space-y-2">
            {[
              { label: "Vertical version", prompt: "Create a 30-second vertical version optimized for TikTok" },
              { label: "Add captions", prompt: "Generate captions and burn them into the video" },
              { label: "Speed ramp", prompt: "Speed up the slow parts and keep the exciting parts normal" },
              { label: "Add background music", prompt: "Add background music that matches the energy of this clip" },
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
