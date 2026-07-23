"use client";

import FeatureLayout from "../FeatureLayout";

export default function EditingPage() {
  return (
    <FeatureLayout title="Smart Editing" subtitle="Trim, cut, and rearrange sections with natural language">
      <div className="space-y-5">
        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Structure</h3>
          <div className="space-y-1.5">
            {[
              { label: "Intro", time: "0:00", color: "from-blue-500/20 to-blue-500/5" },
              { label: "Verse 1", time: "0:32", color: "from-purple-500/20 to-purple-500/5" },
              { label: "Chorus", time: "1:15", color: "from-neon-blue/20 to-neon-blue/5" },
              { label: "Verse 2", time: "1:50", color: "from-purple-500/20 to-purple-500/5" },
              { label: "Chorus", time: "2:35", color: "from-neon-blue/20 to-neon-blue/5" },
              { label: "Bridge", time: "3:10", color: "from-amber-500/20 to-amber-500/5" },
              { label: "Outro", time: "3:45", color: "from-green-500/20 to-green-500/5" },
            ].map((section, i) => (
              <div key={i} className={`flex items-center justify-between rounded-lg bg-gradient-to-r ${section.color} border border-white/5 px-3 py-2`}>
                <span className="text-sm text-white/70">{section.label}</span>
                <span className="text-xs text-white/30 font-mono">{section.time}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Quick Actions</h3>
          <div className="space-y-2">
            {[
              { label: "Trim section", prompt: "Trim the track from 1:00 to 2:30" },
              { label: "Remove section", prompt: "Remove the section from 2:30 to 3:15" },
              { label: "Cut intro", prompt: "Remove the intro and start from the first verse" },
              { label: "Make highlight reel", prompt: "Create a 30-second highlight reel from the best parts" },
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
