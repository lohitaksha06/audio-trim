"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import Nav from "@/components/Nav";
import { getCatalog, type CatalogResponse } from "@/services/api";

const SECTIONS: { title: string; hint: string; prompts: string[] }[] = [
  {
    title: "Voice clarity & noise removal",
    hint: "The AI isolates your vocal stem, polishes it and lifts it over the music (+4 to +6 dB). Name background noise to strip it.",
    prompts: [
      "Make her voice clear and audible",
      "Make voices clearer and remove background noise",
      "Remove background noise",
      "Remove ALL background noise, it is very noisy",
      "Enhance vocals and denoise",
      "Add drums and make the voice more clear",
    ],
  },
  {
    title: "Add instruments (beat-synced)",
    hint: "AI detects your song's BPM, beats and key, then plays along. Say 'add drums' with no style and it matches your song's feel automatically.",
    prompts: [
      "Add drums",
      "Add drums that match the song",
      "Add only snare",
      "Add a kick drum",
      "Add kick and snare without hats",
      "Add funky drums following the groove",
      "Add four-on-the-floor drums",
      "Add swing drums",
      "Add bass guitar",
      "Add strings",
      "Add trumpet",
      "Add choir pad",
      "Add piano chords",
      "Add drums at 100 BPM",
    ],
  },
  {
    title: "EDM kits (full menu)",
    hint: "Every kit gets its own drum groove + synth timbre. Say 'add X drums' for just the groove, 'add X' for the full kit.",
    prompts: [
      "Add house drums",
      "Add techno",
      "Add trance",
      "Add trap drums and 808",
      "Add dnb break and reese bass",
      "Add hardstyle kick",
      "Add phonk cowbell",
      "Add synthwave pad",
      "Add deep house",
      "Add dubstep wobble",
      "Add futuristic synth",
      "Add tropical synth",
      "Add uk garage drums",
      "Add a 2-step beat",
      "Add amapiano drums",
      "Add an afro house groove",
      "Add a jungle break",
      "Add a grime beat",
    ],
  },
  {
    title: "Synth waves & bass",
    hint: "Name the oscillator: supersaw, square, saw, triangle, sine, FM, acid, reese, 808, pluck, pad, arp.",
    prompts: [
      "Add a supersaw lead",
      "Add a square wave lead",
      "Add an acid bassline",
      "Add a reese bass",
      "Add an 808 bass",
      "Add a pluck",
      "Add an arp",
    ],
  },
  {
    title: "Separated stems",
    hint: "Demucs splits any song into 4 stems. Isolate one, remove one, or rebalance them.",
    prompts: [
      "Separate into stems",
      "Keep only the vocals",
      "Extract just the bass",
      "Remove the drums",
      "Remove the kick drum from 2:30 to 3:45",
    ],
  },
  {
    title: "Mix Lab — prioritize sounds",
    hint: "Drums louder, vocals lower? Open Mix Lab for faders — or type it. True stem remix when separation works, EQ-balance otherwise.",
    prompts: [
      "Make the drums louder and the vocals quieter",
      "Prioritize drums over vocals",
      "Balance the mix: vocals -3dB, drums +6dB",
      "Turn up the bass and turn down the background",
    ],
  },
  {
    title: "New edits: reverse, loop, transpose",
    hint: "Flip it backwards, loop a section, or change pitch without changing tempo (±12 semitones).",
    prompts: [
      "Reverse it",
      "Reverse from 0:05 to 0:20",
      "Loop the chorus 3 times",
      "Repeat the intro",
      "Pitch it up 2 semitones",
      "Transpose down 3 semitones",
    ],
  },
  {
    title: "Optimize my song",
    hint: "Mix Doctor measures clipping, dynamics and spectral balance, then gives one-click fixes. Also in Mix Lab → 'Check my mix'.",
    prompts: [
      "Normalize the volume",
      "Make it brighter",
      "Make the bass louder",
      "Give the chorus more energy",
    ],
  },
  {
    title: "Combine layers",
    hint: "Stack several instruments in one go — or chain edits: each prompt builds on your latest output.",
    prompts: [
      "Combine drums and bass",
      "Add both drums and bass",
      "Layer drums, bass and strings together",
      "Add trumpet and sax",
    ],
  },
  {
    title: "Fix & clean",
    hint: "Presence lift, hiss/rumble cut and spectral-gate denoise. Say what you can't hear to turn it up.",
    prompts: [
      "I cant hear the drums",
      "Make the bass louder",
      "Remove that cymbal crash at 1:23 and fill smoothly",
    ],
  },
  {
    title: "Style presets",
    hint: "Honest DSP restyles (drums + groove + tone) — not generative, but instant.",
    prompts: [
      "Convert this song into a house music style",
      "Convert to techno style",
      "Convert to trance style",
      "Convert to hardstyle style",
      "Convert to synthwave style",
      "Convert to garage style",
      "Convert to amapiano style",
      "Convert to afro house style",
      "Convert to jungle style",
      "Convert to grime style",
      "Make it tropical edm style",
      "Make it lofi",
    ],
  },
  {
    title: "Trim & arrange",
    hint: "Timestamps in m:ss, seconds, or song parts after AI analysis.",
    prompts: [
      "Trim from 0:05 to 0:10",
      "Remove the section from 1:00 to 1:30",
      "Remove the silence",
      "Remove ums and ahs",
    ],
  },
  {
    title: "Polish & space",
    hint: "Fades, loudness, mood lighting and room sound.",
    prompts: [
      "Fade in and out",
      "Normalize the volume",
      "Add reverb",
      "Make it darker",
      "Speed it up",
    ],
  },
  {
    title: "Talk to your output",
    hint: "Follow-ups refer to your latest result — no re-processing.",
    prompts: ["Show me the output file", "Send me the download", "Play the result"],
  },
];

const FALLBACK_GROOVES = ["default", "four_on_floor", "funky", "swing", "half_time", "double_time", "tropical", "future", "dubstep", "big_room", "deep_house", "tech_house", "techno", "trance", "trap", "dnb", "hardstyle", "phonk", "synthwave", "garage", "amapiano", "afro_house", "jungle", "grime"];
const STYLES = ["house", "deep_house", "techno", "trance", "trap", "dnb", "hardstyle", "phonk", "synthwave", "tropical", "edm", "futuristic", "dubstep", "lofi", "acoustic", "garage", "amapiano", "afro_house", "jungle", "grime"];

const STEM_INFO: { name: string; contains: string }[] = [
  { name: "vocals", contains: "Lead vocal, speech, choir lead — isolated with Demucs" },
  { name: "drums", contains: "Kick, snare, hats, cymbals, full kit and breaks" },
  { name: "bass", contains: "Bass guitar, 808s, sub and low synth bass" },
  { name: "other", contains: "Everything else: keys, guitars, synths, pads, FX" },
];

const FAMILY_ORDER = ["voice", "rhythm", "bass", "percussion", "strings", "keys", "orchestral", "synth", "edm", "fx"];

export default function GuidePage() {
  const [copied, setCopied] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const copy = async (p: string) => {
    try {
      await navigator.clipboard.writeText(p);
      setCopied(p);
      setTimeout(() => setCopied(null), 1200);
    } catch { /* clipboard unavailable */ }
  };

  useEffect(() => {
    let live = true;
    getCatalog().then((c) => { if (live) setCatalog(c); }).catch(() => { /* offline: static lists */ });
    return () => { live = false; };
  }, []);

  const grouped = useMemo(() => {
    if (!catalog) return null;
    const groups = new Map<string, CatalogResponse["instruments"]>();
    for (const inst of catalog.instruments) {
      if (inst.id === "vocals") continue; // vocals can't be synthesized; listed under stems
      if (!groups.has(inst.family)) groups.set(inst.family, []);
      groups.get(inst.family)!.push(inst);
    }
    return FAMILY_ORDER.filter((f) => groups.has(f)).map((f) => ({ family: f, items: groups.get(f)! }));
  }, [catalog]);

  const grooves = catalog?.grooves?.length ? catalog.grooves : FALLBACK_GROOVES;

  return (
    <div className="flex min-h-screen flex-col bg-black">
      <Nav />
      <div className="pt-14 sm:pt-16 flex-1">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 py-8 sm:py-12">
          <h1 className="text-2xl sm:text-4xl font-bold text-white">What can I ask Audelle?</h1>
          <p className="mt-2 text-sm sm:text-base text-white/40">
            Everything below is real and covered by the backend test-suite.
            Click any prompt to copy it, then paste it in the <Link href="/prompt" className="text-neon-blue hover:underline">Editor</Link> — or open
            the <Link href="/features/mix" className="text-neon-purple hover:underline">Mix Lab</Link> for hands-on faders and mix tips.
          </p>

          <div className="mt-6 grid gap-2 rounded-2xl border border-white/10 bg-white/[0.02] p-4 sm:p-5">
            <h2 className="text-xs font-semibold tracking-widest text-white/50 uppercase">How layering works</h2>
            <ul className="space-y-1.5 text-xs sm:text-sm leading-relaxed text-white/50 list-disc list-inside">
              <li>Each edit builds on your <b className="text-white/70">latest output</b> — add drums, then say “add bass too” and you get piano + drums + bass.</li>
              <li>“New chat” clears the conversation but keeps your file; “New file” starts over completely.</li>
              <li>Every “add” also saves the instrument-only stem — download it solo from the Output panel.</li>
            </ul>
          </div>

          <div className="mt-8 rounded-2xl border border-neon-purple/20 bg-neon-purple/[0.04] p-4 sm:p-5">
            <h2 className="text-base sm:text-lg font-semibold text-white">Separated stems — what&apos;s inside your song</h2>
            <p className="mt-1 text-xs sm:text-sm text-white/40">
              Upload anything and Demucs splits it into 4 stems. Isolate one, remove one, or rebalance them with Mix Lab.
            </p>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {STEM_INFO.map((s) => (
                <div key={s.name} className="rounded-xl border border-white/10 bg-black/40 p-3">
                  <div className="text-sm font-semibold text-neon-purple">{s.name}</div>
                  <div className="mt-0.5 text-xs text-white/50">{s.contains}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.02] p-4 sm:p-5">
            <h2 className="text-base sm:text-lg font-semibold text-white">
              Instruments you can add {catalog ? <span className="text-xs font-normal text-white/30">· live from your backend</span> : <span className="text-xs font-normal text-white/30">· connecting…</span>}
            </h2>
            <p className="mt-1 text-xs sm:text-sm text-white/40">
              Beat-, key- and tempo-matched to your track. Click one to copy its prompt.
            </p>
            {grouped ? (
              <div className="mt-3 space-y-4">
                {grouped.map((g) => (
                  <div key={g.family}>
                    <div className="text-xs font-semibold tracking-widest text-white/40 uppercase">{g.family}</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {g.items.map((inst) => {
                        const prompt = `Add ${inst.name}`;
                        return (
                          <button
                            key={inst.id}
                            onClick={() => copy(prompt)}
                            title={inst.blurb}
                            className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs sm:text-sm text-white/60 transition-colors hover:border-neon-blue/40 hover:text-neon-blue"
                          >
                            {copied === prompt ? "Copied!" : `${inst.name} — ${inst.blurb}`}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-xs text-white/30">Start the backend to browse all {`45+`} instruments live — the prompt lists below work regardless.</p>
            )}
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <span className="text-xs text-white/30 py-1">Grooves:</span>
            {grooves.map((g) => (
              <span key={g} className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-white/50">{g}</span>
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            <span className="text-xs text-white/30 py-1">Styles:</span>
            {STYLES.map((s) => (
              <span key={s} className="rounded-full border border-neon-blue/20 bg-neon-blue/[0.05] px-2.5 py-1 text-xs text-neon-blue/80">{s}</span>
            ))}
          </div>

          <div className="mt-8 space-y-6">
            {SECTIONS.map((sec) => (
              <div key={sec.title} className="rounded-2xl border border-white/10 bg-white/[0.02] p-4 sm:p-5">
                <h2 className="text-base sm:text-lg font-semibold text-white">{sec.title}</h2>
                <p className="mt-1 text-xs sm:text-sm text-white/40">{sec.hint}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {sec.prompts.map((p) => (
                    <button
                      key={p}
                      onClick={() => copy(p)}
                      title="Click to copy"
                      className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs sm:text-sm text-white/60 transition-colors hover:border-neon-blue/40 hover:text-neon-blue"
                    >
                      {copied === p ? "Copied!" : p}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 text-center">
            <Link href="/prompt" className="inline-block rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-8 py-3.5 text-sm sm:text-base font-semibold text-black hover:scale-105 transition-transform">
              Open the Editor
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
