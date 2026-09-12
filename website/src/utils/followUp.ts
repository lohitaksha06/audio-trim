/** Conversational follow-ups + human result summaries for the prompt chat. */

const LEAD = /^(please\s+|now\s+|then\s+|and\s+|ok\s+|okay\s+|so\s+)+/;

const EDIT_START =
  /^(add|remove|delete|drop|trim|cut|crop|shorten|make|convert|turn|change|separate|split|extract|isolate|keep|fade|normaliz|enhance|denoise|clean|generate|create|insert|give it|speed|slow)/;

const SHOW_START =
  /^(show|send|give|play|download|export|share|where|let me (hear|see|have)|i want (to hear|to see|the))/;

const OUTPUT_NOUN =
  /output|result|file|track|song|mix|stem|download|version|drums|bass|vocal/;

/** True when the user refers to the existing output, not a new edit. */
export function isOutputFollowUp(prompt: string): boolean {
  const core = prompt.toLowerCase().trim().replace(LEAD, "");
  if (EDIT_START.test(core)) return false;
  return SHOW_START.test(core) && OUTPUT_NOUN.test(core);
}

export interface ResultMeta {
  added_instrument?: string;
  groove?: string;
  tempo_bpm?: number;
  beat_count?: number;
  hits?: number;
  style?: string;
  enhanced?: string;
  removed_stem?: string;
  isolated_stem?: string;
}

/** "Added drums · funky groove · 99 BPM · 32 hits" — proof the AI did work. */
export function describeResult(intent: string, meta?: ResultMeta | null): string | null {
  if (!meta) return null;
  const bits: string[] = [];
  if (meta.added_instrument) {
    bits.push(`added ${meta.added_instrument}`);
    if (meta.groove && meta.groove !== "default") bits.push(`${meta.groove.replace(/_/g, " ")} groove`);
    if (meta.tempo_bpm) bits.push(`${Math.round(meta.tempo_bpm)} BPM`);
    if (meta.hits) bits.push(`${meta.hits} hits`);
  } else if (meta.style) {
    bits.push(`${meta.style} style`);
    if (meta.tempo_bpm) bits.push(`${Math.round(meta.tempo_bpm)} BPM`);
  } else if (meta.enhanced) {
    bits.push("vocals enhanced + denoised");
  } else if (meta.removed_stem) {
    bits.push(`removed ${meta.removed_stem}`);
  } else if (meta.isolated_stem) {
    bits.push(`isolated ${meta.isolated_stem}`);
  } else {
    return null;
  }
  void intent;
  return bits.join(" · ");
}
