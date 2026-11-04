import { ProjectProvider } from "@/components/ProjectProvider";

/**
 * The feature workspaces share one on-device project. Keeping the provider in
 * this nested layout (instead of the root) means the marketing home, guide and
 * the older Editor/Lab pages do not decode a saved song just to show text.
 */
export default function FeaturesLayout({ children }: { children: React.ReactNode }) {
  return <ProjectProvider>{children}</ProjectProvider>;
}
