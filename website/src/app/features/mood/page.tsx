"use client";

import FeatureLayout from "../FeatureLayout";
import { ActionButton, Panel, Row } from "../FeaturePanels";
import { useFeaturePrompt } from "../FeaturePromptContext";

function RealMood() {
  const ctx = useFeaturePrompt();
  const mood = ctx?.understand?.mood;
  if (!mood) return null;
  return (
    <Panel title="Measured mood">
      <Row label="Overall" value={<span className="capitalize">{mood.mood}</span>} />
      <Row label="Energy" value={mood.energy_mean.toFixed(2)} />
      <Row label="Brightness" value={mood.brightness_mean.toFixed(2)} />
      <Row label="Tension" value={mood.tension_mean.toFixed(2)} />
      <p className="rounded-xl border border-white/12 bg-white/[0.02] p-4 text-sm text-white/68">
        {mood.description}
      </p>
    </Panel>
  );
}

function EnergyCurve() {
  const ctx = useFeaturePrompt();
  const curve = ctx?.understand?.energy_curve?.curve ?? [];
  if (curve.length < 4) return null;
  const max = Math.max(...curve.map((p) => p.energy), 0.0001);
  return (
    <Panel title="Energy over time">
      <div className="flex h-24 items-end gap-px rounded-xl border border-white/12 bg-white/[0.02] p-3">
        {curve.map((p, i) => (
          <span
            key={i}
            title={`${p.t.toFixed(1)}s · energy ${p.energy.toFixed(2)}`}
            className="flex-1 rounded-t bg-neon-blue/60"
            style={{ height: `${Math.max(3, (p.energy / max) * 100)}%` }}
          />
        ))}
      </div>
      <p className="text-xs text-white/58">Measured from your audio — hover a bar for details.</p>
    </Panel>
  );
}

export default function MoodPage() {
  return (
    <FeatureLayout title="Mood & Style" subtitle="Transform the feel, tone and character of your audio">
      <div className="space-y-6">
        <RealMood />
        <EnergyCurve />

        <Panel title="Change the mood">
          <ActionButton label="Make it darker" prompt="Make this sound darker" />
          <ActionButton label="Make it brighter" prompt="Make it brighter" />
          <ActionButton label="More energetic" prompt="Make this more energetic" />
          <ActionButton label="Calm it down" prompt="Calm this down" />
        </Panel>

        <Panel title="Space & polish">
          <ActionButton
            label="Add reverb"
            hint="Cathedral-style room sound"
            prompt="Add reverb to make it sound like a cathedral"
            tone="accent"
          />
          <ActionButton
            label="Fade in and out"
            prompt="Add a fade in at the beginning and fade out at the end"
            tone="accent"
          />
          <ActionButton label="Normalize loudness" prompt="Normalize the volume" tone="accent" />
        </Panel>

        <Panel title="Style presets">
          <ActionButton label="House style" prompt="Convert this song into a house music style" />
          <ActionButton label="Techno style" prompt="Convert to techno style" />
          <ActionButton label="Synthwave style" prompt="Convert to synthwave style" />
          <ActionButton label="Lo-fi" prompt="Make it lofi" />
        </Panel>
      </div>
    </FeatureLayout>
  );
}
