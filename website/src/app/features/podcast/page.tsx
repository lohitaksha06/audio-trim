"use client";

import FeatureLayout from "../FeatureLayout";

export default function PodcastPage() {
  return (
    <FeatureLayout title="Podcast Tools" subtitle="Clean up dialogue, normalize speakers, and generate chapters">
      <div className="space-y-5">
        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Speakers Detected</h3>
          <div className="space-y-2">
            {[
              { name: "Speaker 1", segments: 12, color: "bg-neon-blue" },
              { name: "Speaker 2", segments: 8, color: "bg-neon-purple" },
            ].map((speaker) => (
              <div key={speaker.name} className="flex items-center gap-3 rounded-xl border border-white/5 bg-white/[0.02] p-3">
                <div className={`h-3 w-3 rounded-full ${speaker.color}`} />
                <span className="text-sm text-white/70 flex-1">{speaker.name}</span>
                <span className="text-xs text-white/30">{speaker.segments} segments</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Transcript Preview</h3>
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 space-y-2">
            {[
              { speaker: "Speaker 1", text: "Hey, welcome to the show.", time: "0:05" },
              { speaker: "Speaker 2", text: "Thanks for having me!", time: "0:08" },
              { speaker: "Speaker 1", text: "So tell us about your new project.", time: "0:12" },
            ].map((t, i) => (
              <div key={i} className="flex gap-2 text-sm">
                <span className="shrink-0 text-neon-blue text-xs">{t.speaker}</span>
                <span className="text-white/50">{t.text}</span>
                <span className="ml-auto shrink-0 text-xs text-white/20">{t.time}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Quick Actions</h3>
          <div className="space-y-2">
            {[
              { label: "Remove ums and ahs", prompt: "Remove all ums, ahs, and filler words from this podcast" },
              { label: "Normalize volume", prompt: "Normalize all speakers to the same volume level" },
              { label: "Split by speaker", prompt: "Split this recording by speaker into separate tracks" },
              { label: "Generate chapters", prompt: "Generate chapter markers with timestamps from this transcript" },
              { label: "Remove silence", prompt: "Remove all long pauses and silences longer than 0.5 seconds" },
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
