/**
 * Prompt parser for browser-only mode.
 *
 * The server has a large grammar (instruments, grooves, stems, speakers). This
 * covers only the operations the browser can actually perform, and says so
 * plainly for everything else instead of silently doing nothing. Keeping the
 * grammar small is deliberate: a wrong local guess is worse than an honest
 * "that one needs the server".
 */

import type { LocalPlan } from "./types";

/** Time token: `1:02`, `1:02.5`, `1:02:03`, `90`, `90s`, `2 min`, `1.5 minutes`. */
const T = String.raw`\d+(?::\d{1,2}){0,2}(?:\.\d+)?(?:\s*(?:h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds))?`;

export function normalizePrompt(prompt: string): string {
  return prompt
    .toLowerCase()
    .replace(/[’‘]/g, "'")
    .replace(/\bcant\b/g, "can't")
    .replace(/\bcannot\b/g, "can't")
    .replace(/\bu\b/g, "you")
    .replace(/\bpls\b|\bplease\b/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function parseTime(raw: string): number | null {
  const s = raw.trim().toLowerCase().replace(",", ".");
  if (!s) return null;

  const mixed = s.match(
    /^(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\s*(?:and\s*)?(?:(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes))?\s*(?:and\s*)?(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds))?$/,
  );
  if (mixed) {
    return Number(mixed[1]) * 3600 + Number(mixed[2] ?? 0) * 60 + Number(mixed[3] ?? 0);
  }

  const minSec = s.match(
    /^(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\s*(?:and\s*)?(?:(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?)?$/,
  );
  if (minSec) return Number(minSec[1]) * 60 + Number(minSec[2] ?? 0);

  const colon = s.match(/^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$/);
  if (colon) {
    return Number(colon[1] ?? 0) * 3600 + Number(colon[2]) * 60 + Number(colon[3]);
  }

  const single = s.match(
    /^(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)?$/,
  );
  if (!single) return null;
  const value = Number(single[1]);
  const unit = single[2] ?? "";
  if (/^h/.test(unit)) return value * 3600;
  if (/^m/.test(unit)) return value * 60;
  return value;
}

function matchTime(input: string): number | null {
  const value = parseTime(input);
  return value === null || !Number.isFinite(value) ? null : value;
}

const TARGETS: { target: "vocals" | "drums" | "bass"; re: RegExp }[] = [
  { target: "vocals", re: /\b(vocals?|voices?|speech|singer|singing|vox)\b/ },
  { target: "drums", re: /\b(drums?|kick|snare|hats?|hi-?hats?|percussion|beat)\b/ },
  { target: "bass", re: /\b(bass|bassline|sub bass|808)\b/ },
];

function formatSeconds(value: number): string {
  const m = Math.floor(value / 60);
  const s = value - m * 60;
  const text = s % 1 === 0 ? s.toFixed(0) : s.toFixed(1);
  return `${m}:${text.padStart(2, "0")}`;
}

/** Short durations read better in seconds ("5.0 s") than as "0:05". */
function shortDuration(value: number): string {
  return value < 60 ? `${value.toFixed(1)} s` : formatSeconds(value);
}

function trimPlan(lower: string, duration: number): LocalPlan | null {
  // `from A to B`, `between A and B`, `A to B`
  const range = new RegExp(
    String.raw`(?:from\s+|between\s+)?(${T})\s*(?:to|until|through|and|->|-)\s*(${T})`,
  ).exec(lower);
  if (range) {
    const start = matchTime(range[1]);
    const end = matchTime(range[2]);
    if (start !== null && end !== null && end > start) {
      return {
        intent: "trim",
        params: { start, end: Math.min(end, duration) },
        summary: `Trimmed to ${formatSeconds(start)}–${formatSeconds(Math.min(end, duration))}`,
      };
    }
  }

  const first = new RegExp(String.raw`\bfirst\s+(${T})`).exec(lower);
  if (first) {
    const cut = matchTime(first[1]);
    if (cut !== null) {
      return {
        intent: "trim",
        params: { start: cut, end: duration },
        summary: `Cut the first ${shortDuration(cut)}`,
      };
    }
  }

  const last = new RegExp(String.raw`\blast\s+(${T})`).exec(lower);
  if (last) {
    const cut = matchTime(last[1]);
    if (cut !== null) {
      return {
        intent: "trim",
        params: { start: 0, end: Math.max(0, duration - cut) },
        summary: `Cut the last ${shortDuration(cut)}`,
      };
    }
  }

  const offStart = new RegExp(String.raw`(${T})\s*(?:off|from)\s+the\s+(?:start|beginning|top)`).exec(lower);
  if (offStart) {
    const cut = matchTime(offStart[1]);
    if (cut !== null) {
      return {
        intent: "trim",
        params: { start: cut, end: duration },
        summary: `Cut the first ${shortDuration(cut)}`,
      };
    }
  }

  const toEnd = new RegExp(String.raw`\bto\s+(${T})\s*$`).exec(lower);
  if (toEnd) {
    const end = matchTime(toEnd[1]);
    if (end !== null) {
      return {
        intent: "trim",
        params: { start: 0, end: Math.min(end, duration) },
        summary: `Shortened to ${formatSeconds(Math.min(end, duration))}`,
      };
    }
  }

  // "trim 30 seconds" with no other qualifier reads as "make it 30 seconds".
  const bare = new RegExp(String.raw`\b(?:trim|shorten|crop)\s+(${T})\s*$`).exec(lower);
  if (bare) {
    const end = matchTime(bare[1]);
    if (end !== null) {
      return {
        intent: "trim",
        params: { start: 0, end: Math.min(end, duration) },
        summary: `Shortened to ${formatSeconds(Math.min(end, duration))}`,
      };
    }
  }

  return {
    intent: "trim",
    params: {},
    summary: "",
    unsupported: 'Tell me the range: "Trim from 0:30 to 1:15", "Cut the first 5 seconds", or "Cut the last 10 seconds".',
  };
}

function fadePlan(lower: string, duration: number): LocalPlan {
  const hasIn = /\bfade\s*in\b|\bfade\s+the\s+(?:start|beginning)\b/.test(lower);
  const hasOut = /\bfade\s*out\b|\bfade\s+the\s+end\b/.test(lower);
  const durationMatch = new RegExp(String.raw`\b(?:over|for|of)\s+(${T})`).exec(lower);
  const requested = durationMatch ? matchTime(durationMatch[1]) : null;
  const length = requested && requested > 0 ? requested : 1.5;

  if (!hasIn && !hasOut) {
    return {
      intent: "fade",
      params: { fade_in_seconds: length, fade_out_seconds: length },
      summary: `Faded in and out over ${length.toFixed(1)} s`,
    };
  }
  const inSec = hasIn ? Math.min(length, duration / 2) : 0;
  const outSec = hasOut ? Math.min(length, duration / 2) : 0;
  const parts: string[] = [];
  if (inSec) parts.push(`in over ${inSec.toFixed(1)} s`);
  if (outSec) parts.push(`out over ${outSec.toFixed(1)} s`);
  return {
    intent: "fade",
    params: { fade_in_seconds: inSec, fade_out_seconds: outSec },
    summary: `Faded ${parts.join(" and ")}`,
  };
}

function speedPlan(lower: string): LocalPlan {
  // An exact BPM target beats every vague direction word. Without this branch,
  // "Set tempo to 128 BPM" fell through to the default 1.25x and silently
  // stretched the file to the wrong tempo.
  const bpmMatch =
    /(\d{2,3}(?:\.\d+)?)\s*bpm\b/.exec(lower) ??
    /\b(?:tempo|bpm)\s*(?:to|at|of|:|=)?\s*(\d{2,3}(?:\.\d+)?)\b/.exec(lower);
  if (bpmMatch) {
    const targetBpm = Number(bpmMatch[1]);
    if (Number.isFinite(targetBpm) && targetBpm >= 40 && targetBpm <= 220) {
      return {
        intent: "speed",
        params: { targetBpm },
        summary: "",
      };
    }
    return {
      intent: "speed",
      params: {},
      summary: "",
      unsupported: "Exact tempo targets must be between 40 and 220 BPM.",
    };
  }

  let factor: number | null = null;
  const slower = /\bslow(?:er|\s*down)?\b|\bdown\b/.test(lower);

  const times = /(\d+(?:\.\d+)?)\s*(?:x|×|times)\b/.exec(lower);
  if (times) factor = Number(times[1]);

  if (factor === null) {
    const percent = /(\d+(?:\.\d+)?)\s*%/.exec(lower);
    if (percent) {
      const p = Number(percent[1]);
      // "speed up by 20%" is 1.2x, while "150% speed" is 1.5x. Treating both
      // as p/100 made the first one 0.2x and fall out of range.
      factor = /\bby\b/.test(lower) ? (slower ? 1 - p / 100 : 1 + p / 100) : p / 100;
    }
  }
  if (factor === null && /\bdouble\b/.test(lower)) factor = 2;
  if (factor === null && /\bhalf\b/.test(lower)) factor = 0.5;
  if (factor === null) factor = slower ? 0.8 : 1.25;

  if (!Number.isFinite(factor) || factor <= 0.25 || factor >= 4) {
    return {
      intent: "speed",
      params: {},
      summary: "",
      unsupported: "Speed changes are limited to 0.25x–4x. Try \"Speed up 1.25x\".",
    };
  }

  const direction = factor > 1 ? "Sped up" : "Slowed down";
  return {
    intent: "speed",
    params: { factor },
    summary: `${direction} ×${factor.toFixed(factor % 1 === 0 ? 0 : 2)}, pitch preserved`,
  };
}

/**
 * Map a prompt to a local operation. Unsupported prompts carry an
 * `unsupported` message so the UI can say what only the server can do.
 */
export function parseLocalPrompt(prompt: string, duration: number): LocalPlan {
  const lower = normalizePrompt(prompt);

  // --- tempo question -----------------------------------------------------
  // "What BPM is this?" changes nothing, so it must not fall into the speed
  // branch. The engine measures the file and returns the number.
  if (
    (/\b(what|which|detect|tell\s+me|how\s+many)\b/.test(lower) && /\b(bpm|tempo)\b/.test(lower)) ||
    /\bhow\s+fast\s+is\b/.test(lower)
  ) {
    return { intent: "info", params: { kind: "tempo" }, summary: "" };
  }

  // --- style conversion ---------------------------------------------------
  // "Convert to house style" is not a file-format request. The browser can
  // reshape tone, but re-rendering a genre needs the server's generator.
  if (
    /\b(house|techno|synthwave|edm|hip[- ]?hop|jazz|rock|classical|country|reggae|metal)\b/.test(lower) &&
    /\b(style|genre|convert|make it|turn it)\b/.test(lower) &&
    !/\b(format|file|mp3|wav|flac|aac|ogg|m4a)\b/.test(lower)
  ) {
    return {
      intent: "unknown",
      params: {},
      summary: "",
      unsupported:
        "Style conversion needs the server: it re-renders the audio with a drum and instrument generator. In the browser I can trim, remove a section, loop, change pitch or speed, shape tone/EQ, add reverb, remove pauses, and change level or noise.",
    };
  }

  // --- format -------------------------------------------------------------
  if (/\b(convert|export|change format)\b/.test(lower) || /\bto\s+(mp3|flac|aac|ogg|m4a|wav)\b/.test(lower)) {
    const target = /\b(mp3|flac|aac|ogg|m4a|wav)\b/.exec(lower)?.[1] ?? "wav";
    if (target !== "wav") {
      return {
        intent: "convert",
        params: { format: target },
        summary: "",
        unsupported: `Browsers can only encode WAV. Download the WAV here, or use the server for ${target.toUpperCase()}.`,
      };
    }
    return {
      intent: "convert",
      params: { format: "wav" },
      summary: "Exported as WAV (already the browser format)",
    };
  }

  // --- pauses / silence ----------------------------------------------------
  // This has to run before `trim`: "cut pauses over 0.5s" used to fall into
  // trimPlan and return "tell me the range", which was the wrong answer.
  const pauseWord = /\b(pause|pauses|silence|silences|silent|gap|gaps|quiet\s+parts?)\b/.test(lower);
  const fillerWord = /\b(filler|fillers|um|ums|ah|ahs|uh|uhs|hesitation|stutter)\b/.test(lower);
  const removeWord = /\b(remove|cut|delete|trim|strip|shorten|get\s+rid\s+of)\b/.test(lower);
  if (pauseWord && !fillerWord && removeWord) {
    const bound =
      /(?:longer\s+than|over|above|more\s+than)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b/.exec(lower) ??
      /(?:max(?:imum)?|up\s+to)\s*(?:a\s*)?(?:pause|gap)?\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b/.exec(lower);
    const minSeconds = bound ? Number(bound[1]) : 0.5;
    if (!Number.isFinite(minSeconds) || minSeconds < 0.05 || minSeconds > 20) {
      return {
        intent: "silence",
        params: {},
        summary: "",
        unsupported: "Pause removal works on pauses between 0.05 and 20 seconds.",
      };
    }
    return {
      intent: "silence",
      params: { minSeconds },
      summary: "",
    };
  }

  // --- remove a time range ------------------------------------------------
  const removeRange = new RegExp(
    String.raw`\b(?:remove|delete|cut\s+out)\b[^.]*?\b(?:from|between)\s+(${T})\s*(?:to|and|until|through|-)\s*(${T})`,
  ).exec(lower);
  if (removeRange) {
    const start = matchTime(removeRange[1]);
    const end = matchTime(removeRange[2]);
    if (start !== null && end !== null && end > start) {
      return {
        intent: "remove",
        params: { start, end: Math.min(end, duration) },
        summary: "",
      };
    }
    return {
      intent: "remove",
      params: {},
      summary: "",
      unsupported: 'Tell me the range: "Remove the section from 0:02 to 0:04".',
    };
  }

  // --- noise --------------------------------------------------------------
  if (
    /\b(noise|noisy|hiss|hum|static|denoise|de-noise)\b/.test(lower) ||
    /\bbackground\s+(noise|sound|hum)\b/.test(lower) ||
    /\bclean\s+up\s+the\s+(audio|voice|speech|track|recording)\b/.test(lower) ||
    /\b(remove|reduce|kill)\s+(the\s+)?(background|noise|hiss|hum)\b/.test(lower)
  ) {
    return {
      intent: "denoise",
      params: { strength: 2 },
      summary: "Denoised with a Wiener filter (strength 2.0)",
    };
  }

  // --- trim ---------------------------------------------------------------
  if (/\b(trim|cut|crop|shorten)\b/.test(lower)) {
    const plan = trimPlan(lower, duration);
    if (plan) return plan;
  }

  // --- fade ---------------------------------------------------------------
  if (/\bfade\b/.test(lower)) return fadePlan(lower, duration);

  // --- mood / tone --------------------------------------------------------
  if (/\breverb\b|\bcathedral\b|\broom\s+sound\b|\bhall\b/.test(lower)) {
    return {
      intent: "reverb",
      params: { seconds: 2.2, mix: 0.25 },
      summary: "Reverb added (2.2 s tail, 25% wet)",
    };
  }
  if (/\blo-?fi\b|\blow[- ]fidelity\b/.test(lower)) {
    return {
      intent: "tone",
      params: { kind: "lofi" },
      summary: "Lo-fi tone: highs rolled off above ~3 kHz, low end warmed",
    };
  }
  if (/\b(darker|darken)\b/.test(lower) || /\bdark\s+(sound|mood|tone|mix|version)\b/.test(lower)) {
    return {
      intent: "tone",
      params: { kind: "darker" },
      summary: "Tone tilted darker: highs above ~2 kHz rolled off",
    };
  }
  if (/\b(brighter|brighten)\b/.test(lower) || /\bbright\s+(sound|mood|tone|mix|version)\b/.test(lower)) {
    return {
      intent: "tone",
      params: { kind: "brighter" },
      summary: "Tone tilted brighter: presence above ~2.5 kHz lifted",
    };
  }
  if (/\b(more\s+energetic|energetic|punchier|punchy|more\s+energy)\b/.test(lower)) {
    return {
      intent: "tone",
      params: { kind: "energetic" },
      summary: "Level +2.3 dB with a presence lift above ~2.5 kHz",
    };
  }
  if (/\b(calm|calmer|soothing|soothe)\b/.test(lower)) {
    return {
      intent: "tone",
      params: { kind: "calm" },
      summary: "Softened: highs eased, level −2.2 dB",
    };
  }

  // --- normalize ----------------------------------------------------------
  if (/\bnormaliz|\bnormalis|\bloudness\b/.test(lower)) {
    return { intent: "normalize", params: {}, summary: "" };
  }

  // --- reverse ------------------------------------------------------------
  if (/\breverse[sd]?\b|\bbackwards\b/.test(lower)) {
    return { intent: "reverse", params: {}, summary: "Reversed" };
  }

  // --- pitch / transpose --------------------------------------------------
  const pitchMatch = /\b(?:pitch|transpose)\b[^.]*?\b(up|down)\s*(\d+(?:\.\d+)?)\s*semitones?\b/.exec(lower);
  if (pitchMatch) {
    const semitones = (pitchMatch[1] === "down" ? -1 : 1) * Number(pitchMatch[2]);
    if (Number.isFinite(semitones) && Math.abs(semitones) <= 12) {
      return { intent: "pitch", params: { semitones }, summary: "" };
    }
    return {
      intent: "pitch",
      params: {},
      summary: "",
      unsupported: "Pitch shifts are limited to ±12 semitones.",
    };
  }

  // --- loop / repeat ------------------------------------------------------
  if (/\b(loop|repeat)\b/.test(lower)) {
    const range = new RegExp(
      String.raw`\b(?:from|between)\s+(${T})\s*(?:to|and|until|through|-)\s*(${T})`,
    ).exec(lower);
    const start = range ? matchTime(range[1]) : 0;
    const end = range ? matchTime(range[2]) : duration;
    const timesMatch = /\b(\d+)\s*(?:x|times)\b/.exec(lower);
    const times = Math.max(2, Math.min(8, timesMatch ? Number(timesMatch[1]) : 2));
    if (start !== null && end !== null && end > start) {
      return {
        intent: "loop",
        params: {
          start: Math.min(start, duration),
          end: Math.min(end, duration),
          times,
        },
        summary: "",
      };
    }
    return {
      intent: "loop",
      params: {},
      summary: "",
      unsupported: 'Tell me the loop range: "Loop from 0:15 to 0:30 3 times".',
    };
  }

  // --- speed --------------------------------------------------------------
  if (/\bspeed\b|\btempo\b|\bfaster\b|\bslower\b|\bslow\b|\bsped\s*up\b/.test(lower)) {
    return speedPlan(lower);
  }

  // --- boost / gain -------------------------------------------------------
  const explicitDb = /([+-]?\d+(?:\.\d+)?)\s*db\b/.exec(lower);
  if (explicitDb) {
    const db = Math.max(-24, Math.min(24, Number(explicitDb[1])));
    return { intent: "gain", params: { db }, summary: `Gain ${db > 0 ? "+" : ""}${db.toFixed(1)} dB` };
  }

  if (/\b(clarity|clearer|muffled|muddy)\b/.test(lower)) {
    return {
      intent: "boost",
      params: { target: "vocals", direction: "up" },
      summary: "Presence lifted +4.6 dB across 1–6 kHz (relative to the rest of the mix)",
    };
  }

  const targets = TARGETS.filter((t) => t.re.test(lower)).map((t) => t.target);
  const boostWord =
    /\bboost|\blift\b|\bamplif|\bbring\s+up\b|\bturn\s+up\b|\bpump\s+up\b|\bcan't\s+hear\b|\btoo\s+(quiet|soft)\b/.test(
      lower,
    );
  const loudWord = /\blouder\b/.test(lower);
  const quietWord = /\bquieter\b|\bsofter\b/.test(lower) || /\b(lower|turn\s+down)\s+(the\s+)?(vocals?|drums?|bass)\b/.test(lower);
  const target = targets[0] ?? null;

  if (targets.length >= 2) {
    return {
      intent: "unknown",
      params: {},
      summary: "",
      unsupported:
        "Balancing two parts against each other needs stem separation, which only the server does. Locally I can boost or cut one band: vocals, drums, or bass.",
    };
  }

  if ((boostWord || loudWord || quietWord) && target) {
    const down = quietWord && !loudWord && !boostWord;
    const amount =
      target === "vocals"
        ? down
          ? "Presence cut −6.0 dB across 1–6 kHz"
          : "Presence lifted +4.6 dB across 1–6 kHz"
        : target === "drums"
          ? down
            ? "Sub cut −5.0 dB below 200 Hz, snap −3.0 dB at 2–4 kHz"
            : "Sub +5.6 dB below 200 Hz, snap +3.5 dB at 2–4 kHz"
          : down
            ? "Low end cut −6.0 dB below 250 Hz"
            : "Low end +5.1 dB below 250 Hz";
    return {
      intent: "boost",
      params: { target, direction: down ? "down" : "up" },
      summary: `${amount} (relative to the rest of the mix)`,
    };
  }

  if (boostWord || loudWord) {
    return { intent: "gain", params: { db: 4 }, summary: "Gain +4.0 dB" };
  }
  if (quietWord) {
    return { intent: "gain", params: { db: -4 }, summary: "Gain −4.0 dB" };
  }

  return {
    intent: "unknown",
    params: {},
    summary: "",
    unsupported:
      "That one needs the server. In the browser I can trim, remove a section, loop, fade, change pitch or speed, normalise, reverse, denoise, shape tone/EQ, remove pauses, and boost vocals/drums/bass.",
  };
}

/** Prompt chips shown when no backend is configured. */
export const LOCAL_PROMPTS = [
  "Make it darker",
  "Make it brighter",
  "More energetic",
  "Calm it down",
  "Add reverb",
  "Trim from 0:30 to 1:15",
  "Cut the first 5 seconds",
  "Fade in over 2 seconds",
  "Fade out over 3 seconds",
  "Make it 3 dB louder",
  "Normalize the audio",
  "Reverse it",
  "Pitch up 2 semitones",
  "Loop from 0:15 to 0:30 3 times",
  "Speed up 1.25x",
  "What BPM is this?",
  "Remove long pauses",
  "Remove the background noise",
  "Boost the vocals",
  "Make the drums louder",
  "Boost the bass",
];

/**
 * Whether a feature-page button can run in browser-only mode.
 *
 * The duration is a placeholder: support depends on the intent, not on the
 * file length (trim clamps to the real duration when it runs).
 */
export function localSupport(prompt: string): { supported: boolean; reason?: string } {
  const plan = parseLocalPrompt(prompt, 120);
  return {
    supported: plan.intent !== "unknown" && !plan.unsupported,
    reason: plan.unsupported,
  };
}
