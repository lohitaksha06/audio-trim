"use client";

import { useState } from "react";
import FeatureLayout from "../FeatureLayout";
import { useFeaturePrompt } from "../FeaturePromptContext";

export default function EditingPage() {
  const promptCtx = useFeaturePrompt();
  const [bpm, setBpm] = useState("128");
  return (
    <FeatureLayout title="Smart Editing" subtitle="Trim, cut, and rearrange sections with natural language">
      <div className="space-y-5">
        <div>
          <h3 className="text-sm font-semibold text-white/78 uppercase tracking-wider mb-3">Structure</h3>
          <div className="space-y-1.5">
            {[
              { label: "Intro", time: "0:00", color: "bg-blue-500/10" },
              { label: "Verse 1", time: "0:32", color: "bg-purple-500/10" },
              { label: "Chorus", time: "1:15", color: "bg-neon-blue/10" },
              { label: "Verse 2", time: "1:50", color: "bg-purple-500/10" },
              { label: "Chorus", time: "2:35", color: "bg-neon-blue/10" },
              { label: "Bridge", time: "3:10", color: "bg-amber-500/10" },
              { label: "Outro", time: "3:45", color: "bg-green-500/10" },
            ].map((section, i) => (
              <div key={i} className={`flex items-center justify-between rounded-lg ${section.color} border border-white/10 px-3.5 py-2.5`}>
                <span className="text-sm text-white/90">{section.label}</span>
                <span className="text-xs text-white/58 font-mono">{section.time}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/78 uppercase tracking-wider mb-3">Tempo / BPM</h3>
          <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3 space-y-2">
            <p className="text-xs text-white/68">Time-stretch to an exact tempo. If you ask to change tempo without a BPM, the AI will ask you which BPM you want.</p>
            <div className="flex gap-2">
              <input
                type="number"
                min={40}
                max={220}
                value={bpm}
                onChange={(e) => setBpm(e.target.value)}
                placeholder="128"
                className="w-24 rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white outline-none focus:border-neon-blue/50"
              />
              <button
                onClick={() => promptCtx?.setPrompt(`Set tempo to ${bpm || 128} BPM`)}
                className="flex-1 rounded-lg bg-neon-blue/15 px-3 py-2 text-xs font-medium text-neon-blue hover:bg-neon-blue/25 transition-colors"
              >
                Set tempo to {bpm || 128} BPM
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => promptCtx?.setPrompt(`Speed up to ${bpm || 128} BPM`)}
                className="rounded-lg border border-white/5 bg-white/[0.02] p-2 text-xs text-white/84 hover:border-neon-blue/20 hover:text-neon-blue/80 transition-all"
              >
                Speed up to {bpm || 128} BPM
              </button>
              <button
                onClick={() => promptCtx?.setPrompt(`Slow down to ${bpm || 128} BPM`)}
                className="rounded-lg border border-white/5 bg-white/[0.02] p-2 text-xs text-white/84 hover:border-neon-blue/20 hover:text-neon-blue/80 transition-all"
              >
                Slow down to {bpm || 128} BPM
              </button>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/78 uppercase tracking-wider mb-3">Genre / Style</h3>
          <div className="space-y-2">
            {[
              { label: "What genre is this?", prompt: "What genre is this?" },
              { label: "What EDM style is this?", prompt: "What EDM style is this?" },
              { label: "What BPM is this?", prompt: "What BPM is this?" },
            ].map((action) => (
              <button
                key={action.label}
                onClick={() => promptCtx?.setPrompt(action.prompt)}
                className="w-full text-left rounded-xl border border-white/5 bg-white/[0.02] p-3 text-sm text-white/84 hover:border-neon-blue/20 hover:text-neon-blue/80 hover:bg-neon-blue/5 transition-all"
              >
                {action.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/78 uppercase tracking-wider mb-3">Quick Actions</h3>
          <div className="space-y-2">
            {[
              { label: "Trim section", prompt: "Trim the track from 1:00 to 2:30" },
              { label: "Remove section", prompt: "Remove the section from 2:30 to 3:15" },
              { label: "Cut intro", prompt: "Remove the intro and start from the first verse" },
              { label: "Make highlight reel", prompt: "Create a 30-second highlight reel from the best parts" },
            ].map((action) => (
              <button
                key={action.label}
                onClick={() => promptCtx?.setPrompt(action.prompt)}
                className="w-full text-left rounded-xl border border-white/5 bg-white/[0.02] p-3 text-sm text-white/84 hover:border-neon-blue/20 hover:text-neon-blue/80 hover:bg-neon-blue/5 transition-all"
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
