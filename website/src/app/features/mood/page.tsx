"use client";

import FeatureLayout from "../FeatureLayout";

export default function MoodPage() {
  return (
    <FeatureLayout title="Mood & Style" subtitle="Transform the feel, tone, and character of your audio">
      <div className="space-y-5">
        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Mood Controls</h3>
          <div className="space-y-3">
            {[
              { label: "Brightness", left: "Dark", right: "Bright" },
              { label: "Energy", left: "Calm", right: "Intense" },
              { label: "Warmth", left: "Cold", right: "Warm" },
              { label: "Space", left: "Dry", right: "Reverb" },
            ].map((control) => (
              <div key={control.label}>
                <div className="flex justify-between mb-1.5">
                  <span className="text-xs text-white/40">{control.left}</span>
                  <span className="text-xs text-white/60 font-medium">{control.label}</span>
                  <span className="text-xs text-white/40">{control.right}</span>
                </div>
                <div className="h-2 rounded-full bg-white/10 relative">
                  <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-4 w-4 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple border-2 border-black cursor-pointer" />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Quick Actions</h3>
          <div className="space-y-2">
            {[
              { label: "Make darker", prompt: "Make this section sound darker and more moody" },
              { label: "More energetic", prompt: "Make the chorus more energetic and powerful" },
              { label: "Add reverb", prompt: "Add reverb to the vocals to make it sound like a cathedral" },
              { label: "Add fade in/out", prompt: "Add a smooth fade in at the beginning and fade out at the end" },
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
