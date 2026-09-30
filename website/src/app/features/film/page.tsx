"use client";

import FeatureLayout from "../FeatureLayout";
import { ActionButton, Panel } from "../FeaturePanels";

export default function FilmPage() {
  return (
    <FeatureLayout title="Film & Video" subtitle="Clean dialogue and pull stems for post-production">
      <div className="space-y-6">
        <Panel title="Dialogue">
          <ActionButton
            label="Clean dialogue"
            hint="Isolates speech, cuts hiss and rumble, lifts it over the scene"
            prompt="Clean up the dialogue and remove background noise"
            tone="accent"
          />
          <ActionButton
            label="De-noise a noisy scene"
            prompt="Remove all background noise, it is very noisy"
            tone="accent"
          />
        </Panel>

        <Panel title="Stems for post">
          <ActionButton
            label="Separate into stems"
            hint="Vocals / drums / bass / other — download as a ZIP"
            prompt="Separate into stems"
          />
          <ActionButton label="Keep only the speech" prompt="Keep only the vocals" />
          <ActionButton label="Remove the music bed" prompt="Remove the bass and drums" />
        </Panel>

        <Panel title="Levels & timing">
          <ActionButton label="Normalize scene loudness" prompt="Normalize the volume" />
          <ActionButton label="Match levels in the mix" prompt="Make the drums louder and the vocals quieter" />
        </Panel>
      </div>
    </FeatureLayout>
  );
}
