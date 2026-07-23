"use client";

import FeatureLayout from "../FeatureLayout";

const FORMATS = [
  { ext: "WAV", desc: "Lossless, large file size", icon: "W" },
  { ext: "FLAC", desc: "Lossless, compressed", icon: "F" },
  { ext: "MP3", desc: "Lossy, small file size", icon: "M" },
  { ext: "AAC", desc: "Lossy, better quality than MP3", icon: "A" },
  { ext: "OGG", desc: "Lossy, open format", icon: "O" },
];

export default function ConvertPage() {
  return (
    <FeatureLayout title="Format Conversion" subtitle="Convert between audio formats with quality control">
      <div className="space-y-5">
        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Output Format</h3>
          <div className="space-y-2">
            {FORMATS.map((fmt) => (
              <button
                key={fmt.ext}
                className="w-full flex items-center gap-3 rounded-xl border border-white/5 bg-white/[0.02] p-3 text-left hover:border-neon-blue/20 hover:bg-neon-blue/5 transition-all group"
              >
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white/5 text-sm font-bold text-white/50 group-hover:bg-neon-blue/15 group-hover:text-neon-blue transition-colors">
                  {fmt.icon}
                </div>
                <div>
                  <span className="text-sm text-white/70 block">{fmt.ext}</span>
                  <span className="text-xs text-white/30">{fmt.desc}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Quality Settings</h3>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-1.5">
                <span className="text-xs text-white/40">Bitrate</span>
                <span className="text-xs text-white/60 font-medium">320 kbps</span>
              </div>
              <div className="h-2 rounded-full bg-white/10 relative">
                <div className="absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple border-2 border-black cursor-pointer" />
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1.5">
                <span className="text-xs text-white/40">Sample Rate</span>
                <span className="text-xs text-white/60 font-medium">44.1 kHz</span>
              </div>
              <div className="h-2 rounded-full bg-white/10 relative">
                <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-4 w-4 rounded-full bg-gradient-to-r from-neon-blue to-neon-purple border-2 border-black cursor-pointer" />
              </div>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-white/50 uppercase tracking-wider mb-3">Quick Actions</h3>
          <div className="space-y-2">
            {[
              { label: "WAV to MP3", prompt: "Convert this file from WAV to MP3 format" },
              { label: "WAV to FLAC", prompt: "Convert this file from WAV to FLAC format" },
              { label: "MP3 to WAV", prompt: "Convert this file from MP3 to WAV format" },
              { label: "Any to OGG", prompt: "Convert this file to OGG format" },
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
