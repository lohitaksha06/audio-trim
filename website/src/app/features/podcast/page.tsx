"use client";

import FeatureLayout from "../FeatureLayout";
import { ActionButton, Panel } from "../FeaturePanels";
import { useFeaturePrompt } from "../FeaturePromptContext";

interface SpeakerInfo {
  speakers?: string[];
  talk_time_seconds?: Record<string, number>;
  chapter_count?: number;
  note?: string | null;
}

function SpeakerSummary() {
  const ctx = useFeaturePrompt();
  if (!ctx?.hasFile) return null;

  const meta = (ctx.lastMetadata ?? {}) as unknown as SpeakerInfo;
  const speakers = meta.speakers ?? [];

  if (!speakers.length) {
    return (
      <div className="rounded-xl border border-white/12 bg-white/[0.02] p-4">
        <p className="text-sm text-white/84">Speakers found in your file</p>
        <p className="mt-1 text-sm text-white/58">
          Run &ldquo;Split this interview by speaker&rdquo; — the real detected speakers and their
          talk time will appear here.
        </p>
      </div>
    );
  }

  const chapters = meta.chapter_count;
  return (
    <>
      <Panel title="Speakers detected">
        <div className="space-y-1.5">
          {speakers.map((s, i) => (
            <div
              key={s}
              className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.02] px-3.5 py-2.5"
            >
              <span className="text-sm text-white/84">
                Speaker {i + 1} <span className="text-white/48">({s})</span>
              </span>
              <span className="font-mono text-xs text-white/58">
                {meta.talk_time_seconds?.[s]?.toFixed(1)}s talking
              </span>
            </div>
          ))}
        </div>
        <p className="text-xs text-white/48">
          Anonymous clusters from this recording — not names, and only valid for this file.
        </p>
      </Panel>
      {typeof chapters === "number" && (
        <Panel title="Chapters">
          <div className="rounded-lg border border-white/10 bg-white/[0.02] px-3.5 py-2.5 text-sm text-white/78">
            {chapters} chapter{chapters === 1 ? "" : "s"} marked at real pauses in the conversation.
          </div>
        </Panel>
      )}
    </>
  );
}

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
        <SpeakerSummary />

        <Panel title="Speaker work (real diarization)">
          <ActionButton
            label="Split by speaker"
            hint="Detects speakers, then exports one track per person"
            prompt="Split this interview by speaker into separate tracks"
            tone="accent"
          />
          <ActionButton
            label="Keep only speaker 1"
            hint="Cuts every other speaker's turn"
            prompt="Keep only speaker 1"
            tone="accent"
          />
          <ActionButton
            label="Remove speaker 2"
            prompt="Remove speaker 2"
            tone="accent"
          />
          <ActionButton
            label="Generate chapters"
            hint="Chapter marks at real pauses in the conversation"
            prompt="Generate chapters from this transcript"
          />
        </Panel>

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

        <Panel title="Tighter cuts">
          <ActionButton
            label="Cut pauses over 0.5s"
            prompt="Remove pauses longer than 0.5 seconds"
          />
          <ActionButton
            label="Cut pauses over 1s"
            prompt="Remove fillers with maximum pause of 1.5 seconds"
          />
        </Panel>

        <Panel title="Deeper analysis">
          <p className="rounded-xl border border-white/12 bg-white/[0.02] p-4 text-sm text-white/58">
            Filler removal is acoustic (filled pauses), so it works without a transcript and across
            accents. Speaker labels come from unsupervised clustering, so they separate turns
            reliably on distinct voices but can still confuse similar speakers.
          </p>
        </Panel>
      </div>
    </FeatureLayout>
  );
}
