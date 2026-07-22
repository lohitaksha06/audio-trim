import Link from "next/link";

export default function Nav() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-white/5 bg-black/80 backdrop-blur-xl">
      <div className="mx-auto flex h-14 sm:h-16 max-w-[1600px] items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 sm:h-9 sm:w-9 items-center justify-center rounded-lg bg-gradient-to-br from-neon-blue to-neon-purple text-sm font-bold text-black">
            A
          </div>
          <span className="text-base sm:text-lg font-semibold text-white">Audelle</span>
        </Link>

        <div className="flex items-center gap-4 sm:gap-6">
          <Link
            href="/prompt"
            className="text-sm sm:text-base text-white/50 transition-colors hover:text-white"
          >
            Editor
          </Link>
          <span className="text-white/20">|</span>
          <span className="text-xs sm:text-sm text-white/30">No account needed</span>
        </div>
      </div>
    </nav>
  );
}
