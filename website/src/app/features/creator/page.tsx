"use client";

import FeatureLayout from "../FeatureLayout";
import { ActionButton, Panel } from "../FeaturePanels";
import { AddBeats, AddInstruments } from "../AddLayers";

export default function CreatorPage() {
  return (
    <FeatureLayout title="Content Creator" subtitle="Short-form cuts, clean audio, and platform-ready exports">
      <div className="space-y-6">
        <Panel title="Short-form cuts">
          <ActionButton
            label="30s highlight reel"
            hint="Pulls the best 30 seconds out of your file"
            prompt="Create a 30-second highlight reel from the best parts"
            tone="accent"
          />
          <ActionButton label="15s teaser" prompt="Create a 15-second teaser from the best parts" />
          <ActionButton label="Trim the first 30 seconds" prompt="Trim from 0:00 to 0:30" />
        </Panel>

        <Panel title="Make the voice carry">
          <ActionButton
            label="Clean up background noise"
            prompt="Remove background noise"
            tone="accent"
          />
          <ActionButton
            label="Make the voice clearer and louder"
            prompt="Make voices clearer and remove background noise"
            tone="accent"
          />
          <ActionButton label="Remove filler words" prompt="Remove all ums, ahs and filler words" />
        </Panel>

        <Panel title="Punch it up">
          <ActionButton label="Add energy" prompt="Make this more energetic" />
          <ActionButton label="Speed up the tempo" prompt="Set tempo to 140 BPM" />
          <ActionButton label="Add background music" prompt="Add background music that matches the energy of this clip" />
        </Panel>

        <AddBeats />
        <AddInstruments />

        <Panel title="Export">
          <ActionButton label="Export as MP3" prompt="Convert this file to mp3 format" />
          <ActionButton label="Export as WAV" prompt="Convert this file to wav format" />
        </Panel>
      </div>
    </FeatureLayout>
  );
}
