/** Single source of truth for the feature rail (sidebar + mobile chip row). */

export interface FeatureLink {
  name: string;
  href: string;
}

export const FEATURE_LINKS: FeatureLink[] = [
  { name: "Home", href: "/prompt" },
  { name: "Source Separation", href: "/features/separation" },
  { name: "Smart Editing", href: "/features/editing" },
  { name: "Manual Editor", href: "/features/manual" },
  { name: "Mix Lab", href: "/features/mix" },
  { name: "Mood & Style", href: "/features/mood" },
  { name: "Podcast", href: "/features/podcast" },
  { name: "Content Creator", href: "/features/creator" },
  { name: "Film & Video", href: "/features/film" },
  { name: "Format Conversion", href: "/features/convert" },
];
