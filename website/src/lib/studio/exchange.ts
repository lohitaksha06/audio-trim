/**
 * Timeline interchange. EDL is the most boring format that actually imports
 * into Reaper, Premiere and Resolve; FCPXML covers Final Cut. Both are plain
 * text, generated from the same project document that renders the mix.
 */

import type { StudioAsset, StudioProject } from "./types";

function tc(seconds: number, fps = 25): string {
  const total = Math.max(0, Math.round(seconds * fps));
  const frames = total % fps;
  const secs = Math.floor(total / fps) % 60;
  const mins = Math.floor(total / (fps * 60)) % 60;
  const hours = Math.floor(total / (fps * 3600));
  return [hours, mins, secs, frames].map((value) => String(value).padStart(2, "0")).join(":");
}

function safe(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

export function buildEdl(project: StudioProject, assets: Record<string, StudioAsset>): string {
  const lines: string[] = [`TITLE: ${safe(project.name)}`, "FCM: NON-DROP FRAME", ""];
  let number = 1;
  project.tracks.forEach((track, trackIndex) => {
    track.clips.forEach((clip) => {
      const asset = assets[clip.assetId];
      const reel = safe(asset?.name.replace(/\.[^.]+$/, "") ?? "AX").slice(0, 8).toUpperCase() || "AX";
      const channel = trackIndex % 2 === 0 ? "A" : "A2";
      const end = clip.start + clip.duration;
      lines.push(
        `${String(number).padStart(3, "0")}  ${reel.padEnd(8, " ")} ${channel.padEnd(5, " ")} C        ` +
          `${tc(clip.offset)} ${tc(clip.offset + clip.duration)} ${tc(clip.start)} ${tc(end)}`,
      );
      lines.push(`* FROM CLIP NAME: ${safe(track.name)} / ${safe(clip.name)}`);
      if (clip.fadeIn > 0) lines.push(`* FADE IN: ${clip.fadeIn.toFixed(3)}s`);
      if (clip.fadeOut > 0) lines.push(`* FADE OUT: ${clip.fadeOut.toFixed(3)}s`);
      lines.push("");
      number++;
    });
  });
  return lines.join("\n");
}

function xml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function buildFcpxml(project: StudioProject, assets: Record<string, StudioAsset>): string {
  const assetIds = new Map<string, string>();
  let assetIndex = 1;
  for (const track of project.tracks) {
    for (const clip of track.clips) {
      if (!assetIds.has(clip.assetId)) assetIds.set(clip.assetId, `a${assetIndex++}`);
    }
  }
  const total = project.tracks.reduce(
    (max, track) => Math.max(max, ...track.clips.map((clip) => clip.start + clip.duration), 0),
    0,
  );
  const resources = [...assetIds.entries()]
    .map(([id, ref]) => {
      const asset = assets[id];
      return `    <asset id="${ref}" name="${xml(asset?.name ?? id)}" start="0s" duration="${(asset?.duration ?? 0).toFixed(3)}s" hasVideo="0" hasAudio="1" audioSources="1" audioChannels="${asset?.channels ?? 2}" audioRate="${asset?.sampleRate ?? project.sampleRate}"/>`;
    })
    .join("\n");
  const clips = project.tracks
    .map((track, trackIndex) =>
      track.clips
        .map(
          (clip) =>
            `          <asset-clip ref="${assetIds.get(clip.assetId)}" lane="${trackIndex + 1}" offset="${clip.start.toFixed(3)}s" start="${clip.offset.toFixed(3)}s" duration="${clip.duration.toFixed(3)}s" name="${xml(clip.name)}"/>`,
        )
        .join("\n"),
    )
    .filter(Boolean)
    .join("\n");
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.8">
  <resources>
    <format id="r1" name="FFVideoFormat1080p25" frameDuration="1/25s" width="1920" height="1080"/>
${resources}
  </resources>
  <library>
    <event name="Audelle">
      <project name="${xml(project.name)}">
        <sequence duration="${Math.max(0, total).toFixed(3)}s" format="r1" tcStart="0s" tcFormat="NDF">
          <spine>
${clips}
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
`;
}
