"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSettings } from "@/utils/settings";
import { FEATURE_LINKS } from "./featureNav";

interface SidebarProps {
  onPromptSelect: (prompt: string) => void;
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={`h-6 w-11 shrink-0 rounded-full relative transition-colors focus:outline-none focus-visible:outline-2 focus-visible:outline-neon-blue ${
        checked ? "bg-neon-blue/80" : "bg-white/15"
      }`}
    >
      <span
        className={`absolute top-0.5 h-5 w-5 rounded-full transition-all ${
          checked ? "left-5 bg-black" : "left-0.5 bg-white/70"
        }`}
      />
    </button>
  );
}

/** Small uppercase label that opens a group of controls. */
function GroupLabel({ children }: { children: React.ReactNode }) {
  return <div className="eyebrow px-3 pb-2">{children}</div>;
}

const FEATURE_ICONS: Record<string, React.ReactNode> = {
  "/prompt": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 12l8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25" />
    </svg>
  ),
  "/features/separation": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
    </svg>
  ),
  "/features/editing": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M7.848 8.25l1.536.887M7.848 8.25a3 3 0 11-5.196-3 3 3 0 015.196 3zm1.536.887a2.165 2.165 0 011.083 1.839c.005.351.054.695.14 1.024M9.384 9.137l2.077 1.199M7.848 15.75l1.536-.887m-1.536.887a3 3 0 11-5.196 3 3 3 0 015.196-3zm1.536-.887a2.165 2.165 0 001.083-1.838c.005-.352.054-.695.14-1.025m-1.223 2.863l2.077-1.199m0-3.328a4.323 4.323 0 012.068-1.379l5.325-1.628a4.5 4.5 0 012.48-.044l.803.215m-7.6 2.068a4.323 4.323 0 00-2.068-1.379l-5.325-1.628a4.5 4.5 0 00-2.48-.044l-.803.215" />
    </svg>
  ),
  "/features/manual": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
    </svg>
  ),
  "/features/mix": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 4v16M5 8h4M9 8v4m6-8v16m0-12h4m-4 4v4m-6 0H5m14 0h-4M5 16h4m10 0h-4" />
    </svg>
  ),
  "/features/mood": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4.098 19.902a3.75 3.75 0 005.304 0l6.401-6.402M6.75 21A3.75 3.75 0 013 17.25V4.125C3 3.504 3.504 3 4.125 3h5.25c.621 0 1.125.504 1.125 1.125v4.072M6.75 21a3.75 3.75 0 003.75-3.75V8.197M6.75 21h13.125c.621 0 1.125-.504 1.125-1.125v-5.25c0-.621-.504-1.125-1.125-1.125h-4.072M10.5 8.197l2.88-2.88c.438-.439 1.15-.439 1.59 0l3.712 3.713c.44.44.44 1.152 0 1.59l-2.879 2.88M6.75 17.25h.008v.008H6.75v-.008z" />
    </svg>
  ),
  "/features/podcast": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
    </svg>
  ),
  "/features/creator": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 20.25h12m-7.5-3v3m3-3v3m-10.125-3h17.25c.621 0 1.125-.504 1.125-1.125V4.875c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125z" />
    </svg>
  ),
  "/features/film": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-1.5A1.125 1.125 0 0118 18.375M20.625 4.5H3.375m17.25 0c.621 0 1.125.504 1.125 1.125M20.625 4.5h-1.5C18.504 4.5 18 5.004 18 5.625m3.75 0v1.5m0-1.5a1.125 1.125 0 011.125 1.125M3.375 4.5c-.621 0-1.125.504-1.125 1.125M3.375 4.5h1.5C5.496 4.5 6 5.004 6 5.625m-3.75 0v1.5M6 5.625v1.5m-3.75 0h1.5M6 5.625a1.125 1.125 0 011.125 1.125M6 7.125v1.5m3-1.5v1.5m-3 0h1.5m3-1.5a1.125 1.125 0 011.125 1.125M9.75 7.125v1.5m2.25-1.5v1.5m-2.25 0h1.5m2.25-1.5h1.5m-1.5-1.5v1.5m-3 0v3m3-3h3m-3 0a9 9 0 019-9m-9 9a9 9 0 019 9m9-9a9 9 0 00-9 9m9 9a9 9 0 01-9-9m-9 9h3.375c.621 0 1.125-.504 1.125-1.125V16.5c0-.621-.504-1.125-1.125-1.125H7.125c-.621 0-1.125.504-1.125 1.125v3.375c0 .621.504 1.125 1.125 1.125z" />
    </svg>
  ),
  "/features/convert": (
    <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12l-3-3m0 0l-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  ),
};

const COMING_SOON = [
  { name: "Easy Voiceover Sync", desc: "AI voice generation & sync" },
  { name: "Smart Storyboarding", desc: "AI scene layouts & transitions" },
];

export default function Sidebar({}: SidebarProps) {
  const pathname = usePathname();
  const [settings, updateSettings] = useSettings();
  const guideActive = pathname === "/guide";

  return (
    <aside className="hidden lg:flex w-72 shrink-0 h-full flex-col border-r border-white/10 bg-white/[0.02]">
      {/* One scroll column for the whole rail. A nested scroller used to hide
          the top features behind an inner scroll position, and a pinned
          footer clipped Settings on short viewports — both are gone. */}
      <div className="sidebar-scroll flex-1 overflow-y-auto px-3 pb-8 pt-5">
        <GroupLabel>Features</GroupLabel>
        <nav className="flex flex-col gap-1" aria-label="Features">
          {FEATURE_LINKS.map((cat) => {
            const isActive = pathname === cat.href;
            return (
              <Link
                key={cat.name}
                href={cat.href}
                aria-current={isActive ? "page" : undefined}
                className={`group flex min-h-11 items-center gap-3 rounded-xl px-3 py-2.5 text-[15px] transition-colors ${
                  isActive
                    ? "bg-neon-blue/12 font-medium text-white"
                    : "text-white/72 hover:bg-white/[0.06] hover:text-white"
                }`}
              >
                <span
                  className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-colors ${
                    isActive
                      ? "bg-neon-blue/20 text-neon-blue"
                      : "text-white/60 group-hover:text-white/90"
                  }`}
                >
                  {FEATURE_ICONS[cat.href]}
                </span>
                <span className="truncate">{cat.name}</span>
                {isActive && <span className="ml-auto h-1.5 w-1.5 rounded-full bg-neon-blue" />}
              </Link>
            );
          })}
        </nav>

        <div className="my-4 h-px bg-white/10" />

        <GroupLabel>Reference</GroupLabel>
        <Link
          href="/guide"
          aria-current={guideActive ? "page" : undefined}
          className={`group flex min-h-11 items-center gap-3 rounded-xl px-3 py-2.5 text-[15px] transition-colors ${
            guideActive
              ? "bg-neon-purple/12 font-medium text-white"
              : "text-white/72 hover:bg-white/[0.06] hover:text-white"
          }`}
        >
          <span
            className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-colors ${
              guideActive ? "bg-neon-purple/20 text-neon-purple-light" : "text-neon-purple-light/80"
            }`}
          >
            <svg className="h-[18px] w-[18px]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 012.455 2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
            </svg>
          </span>
          <span className="min-w-0">
            <span className="block truncate">All features &amp; prompts</span>
            <span className="block text-xs text-white/58">Stems, instruments, examples</span>
          </span>
        </Link>

        <div className="my-4 h-px bg-white/10" />

        {/* Coming Soon is dead weight, not navigation — kept to two lines so it
            cannot push Settings out of view. */}
        <div className="px-3">
          <div className="eyebrow pb-2">Coming soon</div>
          <ul className="space-y-1.5 text-xs leading-relaxed text-white/58">
            {COMING_SOON.map((item) => (
              <li key={item.name} className="flex items-baseline gap-2">
                <span className="h-1 w-1 shrink-0 translate-y-[-2px] rounded-full bg-white/30" />
                <span>
                  <span className="text-white/78">{item.name}</span>
                  <span className="block text-white/48">{item.desc}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
        <div className="my-4 h-px bg-white/10" />

        <div className="px-3">
          <div className="eyebrow pb-3">Settings</div>
          <div className="space-y-4">
            <div className="flex items-center justify-between gap-3">
              <label
                htmlFor="set-auto-analyze"
                className="text-sm leading-snug text-white/78"
                title="Run AI analysis (instruments, structure, mood, genre) right after upload"
              >
                Auto-analyze
              </label>
              <Toggle checked={settings.autoAnalyze} onChange={(v) => updateSettings({ autoAnalyze: v })} label="Auto-analyze uploads" />
            </div>
            <div className="flex items-center justify-between gap-3">
              <label
                htmlFor="set-high-quality"
                className="text-sm leading-snug text-white/78"
                title="On: downloads stay lossless WAV. Off: downloads convert to the format below first."
              >
                High quality
              </label>
              <Toggle checked={settings.highQuality} onChange={(v) => updateSettings({ highQuality: v })} label="High quality downloads" />
            </div>
            <div>
              <label htmlFor="set-output-format" className="mb-1.5 block text-sm text-white/78">
                Output format
                {!settings.highQuality && <span className="ml-1 text-xs text-white/58">used on download</span>}
              </label>
              <select
                id="set-output-format"
                value={settings.outputFormat}
                onChange={(e) => updateSettings({ outputFormat: e.target.value })}
                className="w-full rounded-xl border border-white/15 bg-white/[0.06] px-3 py-2.5 text-sm text-white outline-none transition-colors hover:border-white/25 focus:border-neon-blue/60"
              >
                <option value="wav">WAV (Lossless)</option>
                <option value="flac">FLAC (Compressed)</option>
                <option value="mp3">MP3 (Smaller)</option>
                <option value="aac">AAC</option>
                <option value="ogg">OGG</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
