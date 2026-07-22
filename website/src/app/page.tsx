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
              icon="🎵"
              title="Source Separation"
              description="Isolate vocals, drums, bass, guitar, or any instrument. Remove or keep whatever you want."
            />
            <FeatureCard
              icon="✂️"
              title="Smart Trimming"
              description="Cut by section name — 'second chorus to outro' — or remove entire parts."
            />
            <FeatureCard
              icon="🎨"
              title="Mood Transfer"
              description="Make sections darker, brighter, or more energetic. Change the feel with a prompt."
            />
            <FeatureCard
              icon="🎙️"
              title="Podcast Tools"
              description="Remove ums and ahs, normalize speakers, split by speaker, generate chapters."
            />
            <FeatureCard
              icon="🎬"
              title="Video Support"
              description="Upload video — we extract the audio and let you edit it the same way."
            />
            <FeatureCard
              icon="📦"
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
