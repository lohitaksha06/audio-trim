"use client";

import FeatureLayout from "../FeatureLayout";
import { ActionButton, Panel } from "../FeaturePanels";
import { useFeaturePrompt } from "../FeaturePromptContext";
import { fmtDuration } from "../FeaturePanels";

function StructureTimeline() {
  const ctx = useFeaturePrompt();
  const sections = ctx?.understand?.structure?.sections ?? [];
  if (!sections.length) return null;
  return (
    <Panel title="Detected structure">
      <div className="space-y-1.5">
        {sections.map((s, i) => (
          <div
            key={i}
            className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] px-3.5 py-2.5"
          >
            <span className="text-sm capitalize text-white/84">{s.label}</span>
            <span className="font-mono text-xs text-white/58">
              {fmtDuration(s.start)}–{fmtDuration(s.end)}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

export default function EditingPage() {
  return (
    <FeatureLayout title="Smart Editing" subtitle="Trim, cut, rearrange and change tempo with natural language">
      <div className="space-y-6">
        {/* Only shown once real analysis exists — never invented timestamps. */}
        <StructureTimeline />

        <Panel title="Tempo / BPM">
          <p className="rounded-xl border border-white/12 bg-white/[0.02] p-4 text-sm text-white/58">
            Ask for an exact tempo and the engine time-stretches to it. Leave the BPM out and the
            AI asks you which one you want.
          </p>
          <ActionButton
            label="Set tempo to 128 BPM"
            hint="Stretch the audio to exactly 128 BPM"
            prompt="Set tempo to 128 BPM"
          />
          <ActionButton label="Speed up to 140 BPM" prompt="Speed up to 140 BPM" />
          <ActionButton label="Slow down to 90 BPM" prompt="Slow down to 90 BPM" />
          <ActionButton
            label="What BPM is this?"
            hint="Detect the tempo without changing anything"
            prompt="What BPM is this?"
          />
        </Panel>

        <Panel title="Genre / style">
          <ActionButton label="What genre is this?" prompt="What genre is this?" />
          <ActionButton label="What EDM style is this?" prompt="What EDM style is this?" />
        </Panel>

        <Panel title="Trim & arrange">
          <ActionButton label="Trim to first 30 seconds" prompt="Trim from 0:00 to 0:30" />
          <ActionButton label="Remove the intro" prompt="Remove the section from 0:00 to 0:20" />
          <ActionButton
            label="Cut a 30s highlight reel"
            prompt="Create a 30-second highlight reel from the best parts"
          />
        </Panel>

        <Panel title="Pitch & playback">
          <ActionButton label="Pitch up 2 semitones" prompt="Pitch it up 2 semitones" />
          <ActionButton label="Transpose down 3 semitones" prompt="Transpose down 3 semitones" />
          <ActionButton label="Reverse it" prompt="Reverse it" />
          <ActionButton label="Loop a section 3×" prompt="Loop from 0:15 to 0:30 3 times" />
        </Panel>
      </div>
    </FeatureLayout>
  );
}
