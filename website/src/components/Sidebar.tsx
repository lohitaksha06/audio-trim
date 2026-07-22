"use client";

import { useState } from "react";

interface SidebarProps {
  onPromptSelect: (prompt: string) => void;
}

const FEATURE_CATEGORIES = [
  {
    name: "Source Separation",
    icon: "🎵",
    features: [
      { label: "Isolate vocals only", prompt: "Remove everything except the vocals" },
      { label: "Extract drums", prompt: "Give me just the drums as a stem" },
      { label: "Remove bass", prompt: "Remove the bass from this track" },
      { label: "Separate all stems", prompt: "Separate vocals, drums, bass, and other instruments into individual stems" },
      { label: "Remove specific instrument", prompt: "Remove the guitar from this track" },
    ],
  },
  {
    name: "Smart Editing",
    icon: "✂️",
    features: [
      { label: "Trim section", prompt: "Trim the track from 1:00 to 2:30" },
      { label: "Remove section", prompt: "Remove the section from 2:30 to 3:15" },
      { label: "Cut intro", prompt: "Remove the intro and start from the first verse" },
      { label: "Make highlight reel", prompt: "Create a 30-second highlight reel from the best parts" },
    ],
  },
  {
    name: "Mood & Style",
    icon: "🎨",
    features: [
      { label: "Make darker", prompt: "Make this section sound darker and more moody" },
      { label: "More energetic", prompt: "Make the chorus more energetic and powerful" },
      { label: "Add reverb", prompt: "Add reverb to the vocals to make it sound like a cathedral" },
      { label: "Add fade in/out", prompt: "Add a smooth fade in at the beginning and fade out at the end" },
    ],
  },
  {
    name: "Podcast",
    icon: "🎙️",
    features: [
      { label: "Remove ums and ahs", prompt: "Remove all ums, ahs, and filler words from this podcast" },
      { label: "Normalize volume", prompt: "Normalize all speakers to the same volume level" },
      { label: "Split by speaker", prompt: "Split this recording by speaker into separate tracks" },
      { label: "Generate chapters", prompt: "Generate chapter markers with timestamps from this transcript" },
      { label: "Remove silence", prompt: "Remove all long pauses and silences longer than 0.5 seconds" },
    ],
  },
  {
    name: "Content Creator",
    icon: "🎬",
    features: [
      { label: "Vertical version", prompt: "Create a 30-second vertical version optimized for TikTok" },
      { label: "Add captions", prompt: "Generate captions and burn them into the video" },
      { label: "Speed ramp", prompt: "Speed up the slow parts and keep the exciting parts normal" },
      { label: "Add background music", prompt: "Add background music that matches the energy of this clip" },
    ],
  },
  {
    name: "Film & Video",
    icon: "🎥",
    features: [
      { label: "Clean dialogue", prompt: "Clean up the dialogue and remove background noise" },
      { label: "Match room tone", prompt: "Match the room tone across all dialogue clips" },
      { label: "Separate stems", prompt: "Separate dialogue, music, and sound effects into individual stems" },
    ],
  },
];

export default function Sidebar({ onPromptSelect }: SidebarProps) {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <aside className="w-72 lg:w-80 shrink-0 border-r border-white/5 bg-white/[0.01] flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="p-4 sm:p-5 border-b border-white/5">
        <h2 className="text-base sm:text-lg font-semibold text-white/80">Features</h2>
        <p className="mt-1 text-xs sm:text-sm text-white/30">Click to auto-fill prompt</p>
      </div>

      {/* Feature list */}
      <div className="flex-1 overflow-y-auto sidebar-scroll p-3 sm:p-4 space-y-1">
        {FEATURE_CATEGORIES.map((cat) => (
          <div key={cat.name}>
            <button
              onClick={() => setExpanded(expanded === cat.name ? null : cat.name)}
              className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-left transition-colors hover:bg-white/5"
            >
              <span className="text-lg">{cat.icon}</span>
              <span className="flex-1 text-sm sm:text-base font-medium text-white/70">{cat.name}</span>
              <svg
                className={`h-4 w-4 text-white/30 transition-transform ${expanded === cat.name ? "rotate-180" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {expanded === cat.name && (
              <div className="ml-6 mt-1 space-y-0.5 pb-2">
                {cat.features.map((f) => (
                  <button
                    key={f.label}
                    onClick={() => onPromptSelect(f.prompt)}
                    className="w-full text-left px-3 py-2 rounded-md text-xs sm:text-sm text-white/50 transition-colors hover:bg-neon-blue/10 hover:text-neon-blue"
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Settings */}
      <div className="border-t border-white/5 p-4 sm:p-5">
        <div className="flex items-center gap-2 mb-3">
          <svg className="h-4 w-4 text-white/30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          <span className="text-sm sm:text-base font-medium text-white/50">Settings</span>
        </div>
        <div className="space-y-2.5">
          <label className="flex items-center justify-between">
            <span className="text-xs sm:text-sm text-white/40">Auto-analyze</span>
            <div className="h-5 w-9 rounded-full bg-neon-blue/30 relative cursor-pointer">
              <div className="absolute left-4.5 top-0.5 h-4 w-4 rounded-full bg-neon-blue" />
            </div>
          </label>
          <label className="flex items-center justify-between">
            <span className="text-xs sm:text-sm text-white/40">High quality</span>
            <div className="h-5 w-9 rounded-full bg-white/10 relative cursor-pointer">
              <div className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-white/40" />
            </div>
          </label>
          <div>
            <span className="text-xs sm:text-sm text-white/40 block mb-1">Output format</span>
            <select className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs sm:text-sm text-white/60 outline-none">
              <option value="wav">WAV (Lossless)</option>
              <option value="flac">FLAC (Compressed)</option>
              <option value="mp3">MP3 (Smaller)</option>
              <option value="aac">AAC</option>
            </select>
          </div>
        </div>
      </div>
    </aside>
  );
}
