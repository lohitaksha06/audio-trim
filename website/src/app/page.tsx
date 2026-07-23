import Link from "next/link";
import AudioWave from "@/components/AudioWave";
import FeatureCard from "@/components/FeatureCard";

export default function Home() {
  return (
    <>
      {/* Hero Section */}
      <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-4 sm:px-8">
        <div className="absolute inset-0 bg-gradient-to-b from-black via-black/95 to-black" />
        <AudioWave />

        <div className="relative z-10 flex max-w-5xl flex-col items-center text-center">
          <div className="animate-fade-in-up mb-6 sm:mb-8">
            <span className="inline-block rounded-full border border-neon-blue/30 bg-neon-blue/10 px-5 py-2 text-xs sm:text-sm font-medium tracking-wider text-neon-blue uppercase">
              AI-Powered Audio Editor
            </span>
          </div>

          <h1 className="animate-fade-in-up animate-delay-1 mb-6 sm:mb-8 text-4xl sm:text-5xl lg:text-7xl font-bold tracking-tight text-white leading-tight">
            Describe what you want.
            <br />
            <span className="bg-gradient-to-r from-neon-blue via-neon-purple-light to-neon-purple bg-clip-text text-transparent">
              Let AI handle the rest.
            </span>
          </h1>

          <p className="animate-fade-in-up animate-delay-2 mb-10 sm:mb-12 max-w-3xl text-lg sm:text-xl lg:text-2xl leading-relaxed text-white/60">
            Upload audio or video. The AI analyzes every element — instruments,
            vocals, beats, structure. Then just type what you want:
            <span className="block mt-3 italic text-white/40 text-base sm:text-lg">
              &ldquo;Remove the kick drum from 2:30 to 3:45&rdquo;
              &nbsp;&bull;&nbsp;
              &ldquo;Make this section sound darker&rdquo;
              &nbsp;&bull;&nbsp;
              &ldquo;Give me just the vocals&rdquo;
            </span>
          </p>

          <div className="animate-fade-in-up animate-delay-3 flex flex-col items-center gap-4 sm:flex-row">
            <Link
              href="/prompt"
              className="animate-pulse-glow inline-flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-8 sm:px-10 py-4 sm:py-5 text-base sm:text-lg font-semibold text-black transition-all duration-300 hover:scale-105 hover:from-neon-blue/90 hover:to-neon-purple/90"
            >
              <svg className="h-5 w-5 sm:h-6 sm:w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
              </svg>
              Start Editing
            </Link>
            <button className="inline-flex items-center gap-2.5 rounded-xl border border-white/20 px-8 sm:px-10 py-4 sm:py-5 text-base sm:text-lg font-medium text-white/80 transition-all duration-300 hover:border-white/40 hover:text-white">
              <svg className="h-5 w-5 sm:h-6 sm:w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Watch Demo
            </button>
          </div>
        </div>

        <div className="animate-fade-in-up animate-delay-4 absolute bottom-8 z-10">
          <div className="flex animate-bounce flex-col items-center gap-1 text-white/30">
            <span className="text-xs sm:text-sm">Scroll</span>
            <svg className="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="relative border-t border-white/5 px-4 sm:px-8 py-20 sm:py-28">
        <div className="mx-auto max-w-6xl">
          <h2 className="mb-4 text-center text-3xl sm:text-4xl lg:text-5xl font-bold text-white">
            How it works
          </h2>
          <p className="mb-12 sm:mb-16 text-center text-base sm:text-lg text-white/40">
            Three steps to go from raw audio to finished edit.
          </p>

          <div className="grid gap-6 sm:gap-8 md:grid-cols-3">
            {[
              {
                step: "01",
                title: "Upload",
                desc: "Drop any audio or video file. The AI analyzes everything — instruments, structure, speakers, mood — in seconds.",
              },
              {
                step: "02",
                title: "Describe",
                desc: "Just type what you want. Natural language only. No waveforms, no sliders, no learning curve.",
              },
              {
                step: "03",
                title: "Export",
                desc: "Download your edit as MP3, WAV, FLAC, or stems. Or export XML for Premiere / DaVinci / Final Cut.",
              },
            ].map((item) => (
              <div key={item.step} className="group relative rounded-2xl border border-white/5 bg-white/[0.02] p-6 sm:p-8 transition-all duration-300 hover:border-neon-purple/20 hover:bg-white/[0.05]">
                <span className="mb-4 block text-5xl sm:text-6xl font-bold text-white/10 transition-colors duration-300 group-hover:text-neon-purple/30">
                  {item.step}
                </span>
                <h3 className="mb-3 text-xl sm:text-2xl font-semibold text-white">{item.title}</h3>
                <p className="text-sm sm:text-base leading-relaxed text-white/50">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative border-t border-white/5 px-4 sm:px-8 py-20 sm:py-28">
        <div className="mx-auto max-w-6xl">
          <h2 className="mb-4 text-center text-3xl sm:text-4xl lg:text-5xl font-bold text-white">
            What you can do
          </h2>
          <p className="mb-12 sm:mb-16 text-center text-base sm:text-lg text-white/40">
            Everything is possible through natural language.
          </p>

          <div className="grid gap-5 sm:gap-6 sm:grid-cols-2 lg:grid-cols-3">
            <FeatureCard
              icon={<svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" /></svg>}
              title="Source Separation"
              description="Isolate vocals, drums, bass, guitar, or any instrument. Remove or keep whatever you want."
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
            />
            <FeatureCard
              icon={<svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12l-3-3m0 0l-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>}
              title="Stem Export"
              description="Export individual stems as ZIP, or full mixes in MP3, WAV, FLAC, and AAC."
            />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="relative border-t border-white/5 px-4 sm:px-8 py-20 sm:py-28">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="mb-6 text-3xl sm:text-4xl lg:text-5xl font-bold text-white">
            Ready to try it?
          </h2>
          <p className="mb-10 text-lg sm:text-xl text-white/40">
            No accounts. No sign-up. Just upload and start editing.
          </p>
          <Link
            href="/prompt"
            className="inline-flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-neon-blue to-neon-purple px-8 sm:px-10 py-4 sm:py-5 text-lg sm:text-xl font-semibold text-black transition-all duration-300 hover:scale-105"
          >
            Go to Editor
            <svg className="h-5 w-5 sm:h-6 sm:w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>
      </section>
    </>
  );
}
