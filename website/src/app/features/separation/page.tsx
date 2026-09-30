"use client";

import FeatureLayout from "../FeatureLayout";
import { ActionButton, Panel } from "../FeaturePanels";
import { useFeaturePrompt } from "../FeaturePromptContext";

const STEM_LABELS: Record<string, string> = {
  vocals: "Vocals",
  drums: "Drums",
  bass: "Bass",
  guitar: "Guitar",
  piano: "Piano",
  keys: "Keys",
  other: "Other",
};

function DetectedStems() {
  const ctx = useFeaturePrompt();
  const found = ctx?.understand?.instruments?.instruments ?? [];
  if (!found.length) return null;
  return (
    <Panel title="Instruments found in your file">
      <div className="space-y-1.5">
        {found.map((el) => (
          <div
            key={el.instrument}
            className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-white/[0.02] px-3.5 py-2.5"
          >
            <span className="text-sm text-white/84">{STEM_LABELS[el.instrument] ?? el.instrument}</span>
            <span className="flex items-center gap-2">
              <span className="h-1.5 w-24 overflow-hidden rounded-full bg-white/10">
                <span
                  className="block h-full rounded-full bg-neon-blue"
                  style={{ width: `${Math.round(el.confidence * 100)}%` }}
                />
              </span>
              <span className="font-mono text-xs text-white/58">
                {(el.confidence * 100).toFixed(0)}%
              </span>
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

export default function SeparationPage() {
  return (
    <FeatureLayout title="Source Separation" subtitle="Isolate, remove, or extract individual instruments">
      <div className="space-y-6">
        <DetectedStems />

        <Panel title="Split into stems">
          <ActionButton
            label="Separate into 4 stems"
            hint="Demucs splits vocals, drums, bass and everything else"
            prompt="Separate into stems"
            tone="accent"
          />
        </Panel>

        <Panel title="Isolate one instrument">
          <ActionButton label="Keep only the vocals" prompt="Keep only the vocals" />
          <ActionButton label="Extract just the drums" prompt="Give me just the drums as a stem" />
          <ActionButton label="Extract just the bass" prompt="Extract just the bass" />
        </Panel>

        <Panel title="Remove one instrument">
          <ActionButton label="Remove the drums" prompt="Remove the drums" />
          <ActionButton label="Remove the bass" prompt="Remove the bass from this track" />
          <ActionButton label="Remove the kick drum" prompt="Remove the kick drum" />
        </Panel>

        <Panel title="Add instruments back">
          <ActionButton label="Add a bass line" prompt="Add a bass line to this track" tone="accent" />
          <ActionButton label="Add drums" prompt="Add drums to this track" tone="accent" />
          <ActionButton label="Add a synth pad" prompt="Add a synth pad" tone="accent" />
        </Panel>
      </div>
    </FeatureLayout>
  );
}
