"use client";

import FeatureLayout from "../FeatureLayout";
import { ActionButton, Panel } from "../FeaturePanels";

/**
 * Speaker diarization and transcription run on the backend as one-off jobs
 * (`POST /api/ml/diarize`, `/api/ml/transcribe`) rather than as part of the
 * prompt pipeline, so this page links to those instead of showing invented
 * speakers and transcripts.
 */
export default function PodcastPage() {
  return (
    <FeatureLayout title="Podcast Tools" subtitle="Clean up dialogue, normalize speakers, and cut filler words">
      <div className="space-y-6">
        <Panel title="Dialogue cleanup">
          <ActionButton
            label="Remove ums, ahs and filler words"
            prompt="Remove all ums, ahs, and filler words from this podcast"
            tone="accent"
          />
          <ActionButton
            label="Remove long pauses"
            hint="Cuts silence longer than 0.5s"
            prompt="Remove all long pauses and silences longer than 0.5 seconds"
            tone="accent"
          />
          <ActionButton
            label="Make voices clear and audible"
            hint="Isolates the vocal stem, polishes it, lifts it over the music"
            prompt="Make voices clearer and remove background noise"
            tone="accent"
          />
        </Panel>

        <Panel title="Levels">
          <ActionButton
            label="Normalize speaker volume"
            prompt="Normalize all speakers to the same volume level"
          />
          <ActionButton label="Remove room noise" prompt="Remove background noise" />
        </Panel>

        <Panel title="Deeper analysis">
          <p className="rounded-xl border border-white/12 bg-white/[0.02] p-4 text-sm text-white/58">
            Speaker diarization and word-level transcription run as backend jobs and are not part of
            the prompt pipeline yet. Use the sidebar Guide for how the current engine handles speech.
          </p>
        </Panel>
      </div>
    </FeatureLayout>
  );
}
