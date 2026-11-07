import Link from "next/link";
import Nav from "@/components/Nav";
import AudioWave from "@/components/AudioWave";
import FeatureCard from "@/components/FeatureCard";

export default function Home() {
  return (
    <>
      <Nav />
      {/* Hero Section — editorial type, monochrome, signal band below. */}
      <section className="relative overflow-hidden px-5 pt-[calc(var(--nav-h)+1.75rem)] pb-14 sm:px-8 lg:px-10">
        <div className="mx-auto w-full max-w-[1500px]">
          <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-white/12 pb-4 font-mono text-[0.65rem] uppercase tracking-[0.22em] text-white/45">
            <span>Audelle — audio editor</span>
            <span>Multitrack · mix · master</span>
          </div>

          <div className="pt-14 sm:pt-20 lg:pt-24">
            <h1 className="animate-fade-in-up animate-delay-1 font-display text-[clamp(2rem,10.4vw,10rem)] font-black lowercase leading-[0.82] tracking-[-0.055em] text-[#f2f1ec]">
              <span className="block whitespace-nowrap">upgrade your sound.</span>
              <span className="block whitespace-nowrap">bit by bit.</span>
            </h1>
          </div>

          <div className="mt-10 flex flex-col gap-8 border-b border-white/10 pb-10 sm:mt-14 sm:flex-row sm:items-end sm:justify-between">
            <p className="max-w-xl text-base leading-relaxed text-white/62 sm:text-lg">
              Upload audio or video, describe the edit in plain language, and Audelle renders the
              mix. Clips, channels, tone, timing and loudness in one workspace.
            </p>
            <div className="flex shrink-0 flex-col gap-2.5 sm:flex-row">
              <Link
                href="/prompt"
                className="inline-flex items-center justify-between gap-8 rounded-none bg-[#f2f1ec] px-6 py-4 text-sm font-medium text-black transition-colors hover:bg-white"
              >
                Start editing
                <span aria-hidden>→</span>
              </Link>
              <Link
                href="/studio"
                className="inline-flex items-center justify-between gap-8 rounded-none border border-white/25 px-6 py-4 text-sm font-medium text-white/75 transition-colors hover:border-white/60 hover:text-white"
              >
                Open Studio
                <span aria-hidden>↗</span>
              </Link>
              <Link
                href="/lab"
                className="inline-flex items-center justify-between gap-8 rounded-none border border-white/15 px-6 py-4 text-sm font-medium text-white/55 transition-colors hover:border-white/40 hover:text-white"
              >
                Lab
                <span aria-hidden>↗</span>
              </Link>
            </div>
          </div>

          <div className="animate-fade-in-up animate-delay-2 mt-10">
            <div className="mb-3 flex items-center justify-between font-mono text-[0.65rem] uppercase tracking-[0.22em] text-white/40">
              <span>Live signal</span>
              <span>signal / canvas</span>
            </div>
            <div className="relative h-[clamp(150px,22vh,260px)] w-full overflow-hidden border-y border-white/10">
              <AudioWave />
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="relative scroll-mt-[var(--nav-h)] border-t border-white/10 px-5 py-20 sm:px-8 sm:py-28 lg:px-10">
        <div className="mx-auto max-w-[1500px]">
          <p className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-white/40">
            Process
          </p>
          <h2 className="mt-4 max-w-3xl font-display text-5xl font-black lowercase leading-[0.9] tracking-[-0.04em] text-[#f2f1ec] sm:text-6xl lg:text-7xl">
            how it works.
          </h2>
          <p className="mt-5 max-w-xl text-base leading-relaxed text-white/58">
            Three steps from raw audio to a finished edit.
          </p>

          <div className="mt-12 grid gap-8 md:grid-cols-3">
            {[
              {
                step: "01",
                title: "Upload",
                desc: "Drop audio or video. The session opens with a measured clip on the timeline.",
              },
              {
                step: "02",
                title: "Describe",
                desc: "Type the change you want. Natural language, no waveforms, no sliders, no learning curve.",
              },
              {
                step: "03",
                title: "Export",
                desc: "Download a WAV mix, stems, EDL or FCPXML. Compressed formats and source separation need the server.",
              },
            ].map((item) => (
              <div key={item.step} className="border-t border-white/15 pt-6">
                <span className="font-display text-4xl font-black leading-none text-white/25 sm:text-5xl">
                  {item.step}
                </span>
                <h3 className="mt-6 font-display text-2xl font-bold text-white sm:text-3xl">{item.title}</h3>
                <p className="mt-3 text-sm leading-relaxed text-white/58 sm:text-base">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative border-t border-white/10 px-5 py-20 sm:px-8 sm:py-28 lg:px-10">
        <div className="mx-auto max-w-[1500px]">
          <p className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-white/40">
            Capabilities
          </p>
          <h2 className="mt-4 max-w-3xl font-display text-5xl font-black lowercase leading-[0.9] tracking-[-0.04em] text-[#f2f1ec] sm:text-6xl lg:text-7xl">
            what you can do.
          </h2>
          <p className="mt-5 max-w-xl text-base leading-relaxed text-white/58">
            Everything is possible through natural language.
          </p>
          <div className="mt-12 grid gap-x-8 gap-y-10 sm:grid-cols-2 lg:grid-cols-3">
            <FeatureCard
              icon={<svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" /></svg>}
              title="Source Separation"
              description="Isolate vocals, drums, bass, guitar, or any instrument. Remove or keep whatever you want."
              note="server"
            />
            <FeatureCard
              icon={<svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M7.848 8.25l1.536.887M7.848 8.25a3 3 0 11-5.196-3 3 3 0 015.196 3zm1.536.887a2.165 2.165 0 011.083 1.839c.005.351.054.695.14 1.024M9.384 9.137l2.077 1.199M7.848 15.75l1.536-.887m-1.536.887a3 3 0 01-5.196 3 3 3 0 015.196-3zm1.536-.887a2.165 2.165 0 001.083-1.838c.005-.352.054-.695.14-1.025m-1.223 2.863l2.077-1.199m0-3.328a4.323 4.323 0 012.068-1.379l5.325-1.628a4.5 4.5 0 012.48-.044l.803.215m-7.6 2.068a4.323 4.323 0 00-2.068-1.379l-5.325-1.628a4.5 4.5 0 00-2.48-.044l-.803.215" /></svg>}
              title="Smart Trimming"
              description="Cut by section name — 'second chorus to outro' — or remove entire parts."
            />
            <FeatureCard
              icon={<svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M4.098 19.902a3.75 3.75 0 005.304 0l6.401-6.402M6.75 21A3.75 3.75 0 013 17.25V4.125C3 3.504 3.504 3 4.125 3h5.25c.621 0 1.125.504 1.125 1.125v4.072M6.75 21a3.75 3.75 0 003.75-3.75V8.197M6.75 21h13.125c.621 0 1.125-.504 1.125-1.125v-5.25c0-.621-.504-1.125-1.125-1.125h-4.072M10.5 8.197l2.88-2.88c.438-.439 1.15-.439 1.59 0l3.712 3.713c.44.44.44 1.152 0 1.59l-2.879 2.88M6.75 17.25h.008v.008H6.75v-.008z" /></svg>}
              title="Mood Transfer"
              description="Make sections darker, brighter, or more energetic. Change the feel with a prompt."
            />
            <FeatureCard
              icon={<svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" /></svg>}
              title="Podcast Tools"
              description="Remove ums and ahs, normalize speakers, split by speaker, generate chapters."
            />
            <FeatureCard
              icon={<svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-1.5A1.125 1.125 0 0118 18.375M20.625 4.5H3.375m17.25 0c.621 0 1.125.504 1.125 1.125M20.625 4.5h-1.5C18.504 4.5 18 5.004 18 5.625m3.75 0v1.5c0 .621-.504 1.125-1.125 1.125M3.375 4.5c-.621 0-1.125.504-1.125 1.125M3.375 4.5h1.5C5.496 4.5 6 5.004 6 5.625m-3.75 0v1.5c0 .621.504 1.125 1.125 1.125m0 0h1.5m-1.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125m1.5-3.75C5.496 8.25 6 7.746 6 7.125v-1.5M4.875 8.25C5.496 8.25 6 8.754 6 9.375v1.5m0-5.25v5.25m0-5.25C6 5.004 6.504 4.5 7.125 4.5h9.75c.621 0 1.125.504 1.125 1.125m1.125 2.625h1.5m-1.5 0A1.125 1.125 0 0118 7.125v-1.5m1.125 2.625c-.621 0-1.125.504-1.125 1.125v1.5m2.625-2.625c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125M18 5.625v5.25M7.125 12h9.75m-9.75 0A1.125 1.125 0 016 10.875M7.125 12C6.504 12 6 12.504 6 13.125m0-2.25C6 11.496 5.496 12 4.875 12M18 10.875c0 .621-.504 1.125-1.125 1.125M18 10.875c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125m-12 5.25v-5.25m0 5.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125m-12 0v-1.5c0-.621-.504-1.125-1.125-1.125M18 18.375v-5.25m0 5.25v-1.5c0-.621.504-1.125 1.125-1.125M18 13.125v1.5c0 .621.504 1.125 1.125 1.125M18 13.125c0-.621.504-1.125 1.125-1.125M6 13.125v1.5c0 .621-.504 1.125-1.125 1.125M6 13.125C6 12.504 5.496 12 4.875 12m-1.5 0h1.5m-1.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125m1.5-3.75c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m0 0h-1.5" /></svg>}
              title="Video Support"
              description="Upload video — we extract the audio and let you edit it the same way."
              note="server"
            />
            <FeatureCard
              icon={<svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12l-3-3m0 0l-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>}
              title="Stem Export"
              description="Export individual stems as ZIP, or full mixes in MP3, WAV, FLAC, and AAC."
              note="server"
            />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="relative border-t border-white/10 px-5 py-20 sm:px-8 sm:py-28 lg:px-10">
        <div className="mx-auto max-w-[1500px]">
          <p className="font-mono text-[0.65rem] uppercase tracking-[0.22em] text-white/40">
            Start
          </p>
          <div className="mt-4 flex flex-col gap-8 sm:flex-row sm:items-end sm:justify-between">
            <h2 className="font-display text-5xl font-black lowercase leading-[0.9] tracking-[-0.04em] text-[#f2f1ec] sm:text-6xl lg:text-7xl">
              ready to try it?
            </h2>
            <Link
              href="/prompt"
              className="inline-flex shrink-0 items-center justify-between gap-8 rounded-none bg-[#f2f1ec] px-6 py-4 text-sm font-medium text-black transition-colors hover:bg-white"
            >
              Go to editor
              <span aria-hidden>→</span>
            </Link>
          </div>
          <p className="mt-6 max-w-xl text-sm leading-relaxed text-white/50">
            Upload a file and start editing.
          </p>
        </div>
      </section>
    </>
  );
}
