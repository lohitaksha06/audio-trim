"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FEATURE_LINKS } from "./featureNav";

const LINKS = [
  { name: "Editor", href: "/prompt" },
  { name: "Lab", href: "/lab" },
  { name: "Guide", href: "/guide" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-white/10 bg-black/90 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-[1600px] items-center justify-between gap-6 px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex shrink-0 items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-neon-blue to-neon-purple text-base font-bold text-black">
            A
          </span>
          <span className="text-lg font-semibold tracking-tight text-white">Audelle</span>
        </Link>

        <nav className="flex items-center gap-1.5" aria-label="Main">
          {LINKS.map((link) => {
            const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
            return (
              <Link
                key={link.name}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={`rounded-lg px-3.5 py-2 text-[15px] transition-colors ${
                  active
                    ? "bg-white/[0.08] font-medium text-white"
                    : "text-white/72 hover:bg-white/[0.05] hover:text-white"
                }`}
              >
                {link.name}
              </Link>
            );
          })}
          <span className="ml-2 hidden border-l border-white/15 pl-4 text-sm text-white/58 xl:inline">
            No account needed
          </span>
        </nav>
      </div>

      {/* Below lg the sidebar is hidden, so the feature rail becomes a single
          horizontal scroller instead of leaving features unreachable. */}
      <nav
        className="sidebar-scroll flex gap-2 overflow-x-auto border-t border-white/10 px-4 py-2 lg:hidden"
        aria-label="Features"
      >
        {FEATURE_LINKS.map((f) => {
          const active = pathname === f.href;
          return (
            <Link
              key={f.href}
              href={f.href}
              aria-current={active ? "page" : undefined}
              className={`shrink-0 whitespace-nowrap rounded-full border px-3.5 py-1.5 text-sm transition-colors ${
                active
                  ? "border-neon-blue/50 bg-neon-blue/15 font-medium text-white"
                  : "border-white/12 text-white/72 hover:border-white/30 hover:text-white"
              }`}
            >
              {f.name}
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
