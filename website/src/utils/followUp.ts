/** Conversational follow-ups + human result summaries for the prompt chat. */

const LEAD = /^(please\s+|now\s+|then\s+|and\s+|ok\s+|okay\s+|so\s+)+/;

const EDIT_START =
  /^(add|remove|delete|drop|trim|cut|crop|shorten|make|convert|turn|change|separate|split|extract|isolate|keep|fade|normaliz|enhance|denoise|clean|generate|create|insert|give it|speed|slow|mix|balance|rebalance|prioritiz|prioritise|feature)/;

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
  combined?: string[];
  groove?: string;
  wave?: string;
  tempo_bpm?: number;
  beat_count?: number;
  hits?: number;
  style?: string;
  enhanced?: string;
  boosted?: string;
  removed_stem?: string;
  isolated_stem?: string;
  gain_db?: number;
  gains_db?: Record<string, number>;
  method?: string;
  snr_before_db?: number;
  snr_after_db?: number;
  snr_gain_db?: number;
  vocal_boost_db?: number;
  fillers_removed?: number;
  seconds_saved?: number;
  note?: string;
  song_bpm?: number;
  stem_bpm?: number;
  target_bpm?: number;
  speed_factor?: number;
  stretch_factor?: number;
  genre?: string;
  genre_confidence?: number;
  edm_style?: string;
  groove_confidence?: number;
  beat_offset_sec?: number;
  stem_level?: number;
  /** Provenance of added material — shown so nothing is oversold. */
  timbre_source?: string;
  timbre_family?: string;
  timbre_requested?: string;
  notes?: number;
  root_midi?: number;
  drum_source?: string;
  speakers?: string[];
  talk_time_seconds?: Record<string, number>;
  chapter_count?: number;
  split_count?: number;
  selected_speaker?: string;
  action?: "keep" | "remove";
  spans_cut?: number;
  seconds_removed?: number;
}

function grooveBits(meta: ResultMeta, bits: string[]) {
  if (meta.groove && meta.groove !== "default") bits.push(`${meta.groove.replace(/_/g, " ")} groove`);
  if (meta.tempo_bpm) bits.push(`${Math.round(meta.tempo_bpm)} BPM`);
  if (meta.hits) bits.push(`${meta.hits} hits`);
  // Say where the sound actually came from, rather than implying it was invented.
  if (meta.drum_source) bits.push("real recorded hits");
  if (meta.timbre_family) bits.push(`${meta.timbre_family.replace("/", " ")} timbre from trained model`);
  else if (meta.timbre_requested) bits.push(`${meta.timbre_requested} (rule-based synth)`);
}

/** "Added drums · funky groove · 99 BPM · 32 hits" — proof the AI did work. */
export function describeResult(intent: string, meta?: ResultMeta | null): string | null {
  if (!meta) return null;
  const bits: string[] = [];
  if (intent === "split_speakers" || typeof meta.split_count === "number") {
    const n = meta.speakers?.length ?? meta.split_count ?? 0;
    if (!n) {
      return meta.note ?? null;
    }
    bits.push(`split into ${n} speaker track${n === 1 ? "" : "s"}`);
    if (meta.talk_time_seconds) {
      const talk = Object.entries(meta.talk_time_seconds)
        .map(([s, t]) => `${s} ${Number(t).toFixed(1)}s`)
        .join(" · ");
      if (talk) bits.push(talk);
    }
  } else if (intent === "keep_speaker" || intent === "remove_speaker") {
    if (!meta.selected_speaker) {
      return meta.note ?? null;
    }
    const name = meta.selected_speaker;
    if (meta.action === "remove") {
      bits.push(`removed ${name}`);
      if (typeof meta.seconds_removed === "number") {
        bits.push(`${meta.seconds_removed}s cut`);
      }
      if (meta.spans_cut) bits.push(`${meta.spans_cut} turn${meta.spans_cut === 1 ? "" : "s"}`);
    } else {
      bits.push(`kept only ${name}`);
      if (typeof meta.spans_cut === "number" && meta.spans_cut > 0) {
        bits.push(`${meta.spans_cut} other segment${meta.spans_cut === 1 ? "" : "s"} removed`);
      }
    }
  } else if (intent === "chapters" || typeof meta.chapter_count === "number") {
    const n = meta.chapter_count ?? 0;
    if (!n) return meta.note ?? null;
    bits.push(`marked ${n} chapter${n === 1 ? "" : "s"} at real pauses`);
    if (meta.speakers?.length) bits.push(`${meta.speakers.length} speaker${meta.speakers.length === 1 ? "" : "s"} detected`);
  } else if (intent === "mix_stem" || (meta.song_bpm && meta.stem_bpm)) {
    bits.push(`mixed your stem (${meta.stem_bpm ?? "?"} → ${meta.song_bpm ?? "?"} BPM`);
    if (meta.stretch_factor && meta.stretch_factor !== 1) bits.push(`stretched ×${meta.stretch_factor}`);
    if (typeof meta.beat_offset_sec === "number") bits.push(`aligned +${meta.beat_offset_sec}s`);
    bits.push(`${Math.round((meta.stem_level ?? 0.6) * 100)}% level)`);
  } else if (meta.combined && meta.combined.length > 0) {
    bits.push(`added ${meta.combined.join(" + ")}`);
    grooveBits(meta, bits);
  } else if (meta.added_instrument) {
    bits.push(`added ${meta.added_instrument}${meta.wave ? ` (${meta.wave} wave)` : ""}`);
    grooveBits(meta, bits);
  } else if (meta.gains_db && Object.keys(meta.gains_db).length > 0) {
    bits.push(
      `rebalanced ${Object.entries(meta.gains_db)
        .map(([k, v]) => `${k} ${v > 0 ? "+" : ""}${v}dB`)
        .join(", ")}${meta.method === "eq_balance" ? " (EQ balance)" : " (stem remix)"}`
    );
  } else if (meta.boosted) {
    bits.push(`turned up ${meta.boosted}`);
  } else if (typeof meta.gain_db === "number") {
    bits.push(`gain ${meta.gain_db > 0 ? "+" : ""}${meta.gain_db} dB`);
  } else if (typeof meta.fillers_removed === "number") {
    if (meta.fillers_removed === 0) {
      bits.push("no hesitation detected — audio unchanged");
    } else {
      bits.push(
        `removed ${meta.fillers_removed} filler${meta.fillers_removed === 1 ? "" : "s"}` +
          (meta.seconds_saved ? ` · ${meta.seconds_saved}s saved` : "")
      );
    }
  } else if (typeof meta.target_bpm === "number" || typeof meta.speed_factor === "number" || typeof meta.stretch_factor === "number") {
    const from = typeof meta.song_bpm === "number" ? `${Math.round(meta.song_bpm)} BPM → ` : "";
    const to = typeof meta.target_bpm === "number" ? `${Math.round(meta.target_bpm)} BPM` : "";
    const factor = typeof (meta.stretch_factor ?? meta.speed_factor) === "number"
      ? ` (×${(meta.stretch_factor ?? meta.speed_factor) as number})` : "";
    bits.push(`tempo ${from}${to}${factor}`.trim());
  } else if (meta.genre || meta.edm_style || typeof meta.song_bpm === "number") {
    if (meta.genre) bits.push(`genre: ${meta.genre}${typeof meta.genre_confidence === "number" ? ` (${Math.round(meta.genre_confidence * 100)}%)` : ""}`);
    if (meta.edm_style) bits.push(`style: ${String(meta.edm_style).replace(/_/g, " ")}`);
    if (typeof meta.song_bpm === "number") bits.push(`${Math.round(meta.song_bpm)} BPM`);
  } else if (meta.style) {
    bits.push(`${meta.style} style`);
    if (meta.tempo_bpm) bits.push(`${Math.round(meta.tempo_bpm)} BPM`);
  } else if (meta.enhanced) {
    const target = meta.enhanced === "mix" ? "background noise" : "voice";
    if (typeof meta.snr_after_db === "number") {
      const gain = typeof meta.snr_gain_db === "number" ? meta.snr_gain_db : 0;
      bits.push(
        `${target} cleaned · speech-to-noise ${Math.round(meta.snr_after_db)} dB` +
          (gain > 0 ? ` (+${Math.round(gain)} dB better)` : " (already clean)")
      );
    } else if (meta.method === "stem_remix" && typeof meta.vocal_boost_db === "number") {
      bits.push(`vocals lifted +${meta.vocal_boost_db} dB over the mix`);
    } else {
      bits.push(`vocals enhanced + denoised`);
    }
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
