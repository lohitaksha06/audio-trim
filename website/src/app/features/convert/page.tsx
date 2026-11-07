"use client";

import FeatureLayout from "../FeatureLayout";
import { isBackendConfigured } from "@/services/api";
import { ActionButton, Panel, Row } from "../FeaturePanels";
import { useFeaturePrompt } from "../FeaturePromptContext";

const FORMATS = [
  { ext: "WAV", desc: "Lossless, large file", prompt: "Convert this file to wav format" },
  { ext: "FLAC", desc: "Lossless, compressed", prompt: "Convert this file to flac format" },
  { ext: "MP3", desc: "Lossy, small file", prompt: "Convert this file to mp3 format" },
  { ext: "AAC", desc: "Lossy, better than MP3", prompt: "Convert this file to aac format" },
  { ext: "OGG", desc: "Lossy, open format", prompt: "Convert this file to ogg format" },
];

function SourceInfo() {
  const ctx = useFeaturePrompt();
  const a = ctx?.analysis;
  if (!a) return null;
  return (
    <Panel title="Your source file">
      <Row label="Sample rate" value={`${(a.sample_rate / 1000).toFixed(1)} kHz`} />
      <Row label="Channels" value={a.channels === 1 ? "Mono" : "Stereo"} />
      <Row label="Duration" value={`${Math.floor(a.duration_seconds / 60)}:${String(Math.floor(a.duration_seconds % 60)).padStart(2, "0")}`} />
    </Panel>
  );
}

export default function ConvertPage() {
  return (
    <FeatureLayout
      title="Format Conversion"
      subtitle="Convert between audio formats"
      featureKey="convert"
      eyebrow="format conversion"
      accent="#d8b46a"
      intro="Export the current render in the format your platform needs."
    >
      <div className="space-y-6">
        <SourceInfo />

        <Panel title="Convert to">
          {(isBackendConfigured ? FORMATS : FORMATS.filter((fmt) => fmt.ext === "WAV")).map((fmt) => (
            <ActionButton key={fmt.ext} label={fmt.ext} hint={fmt.desc} prompt={fmt.prompt} />
          ))}
        </Panel>

        <Panel title="Quality">
          <p className="rounded-xl border border-white/12 bg-white/[0.02] p-4 text-sm text-white/58">
            {isBackendConfigured
              ? "The converter uses each codec's default settings — there is no bitrate or sample-rate control yet."
              : "This build exports WAV. FLAC, MP3, AAC and OGG need the backend; use the WAV download above in the meantime."}
          </p>
        </Panel>
      </div>
    </FeatureLayout>
  );
}
