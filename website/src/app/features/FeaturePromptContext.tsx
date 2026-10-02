"use client";
import { createContext, useContext } from "react";
import type { UploadResponse, UnderstandResponse } from "@/services/api";

/**
 * Shared state for the feature pages: the real uploaded file, its real AI
 * analysis, and the prompt composer. Feature pages render from this instead of
 * hardcoded placeholder data.
 */
export const FeaturePromptContext = createContext<{
  /** Fill the composer (does not submit). */
  setPrompt: (p: string) => void;
  /** Fill the composer and send it immediately. */
  triggerPrompt?: (p: string) => void;
  hasFile: boolean;
  analysis?: UploadResponse["analysis"];
  understand: UnderstandResponse | null;
  analyzing: boolean;
  runAnalyze: () => void;
  busy: boolean;
  hasResult: boolean;
  fileName: string | null;
  /** Number of completed edits — lets a page react to a new result. */
  edits: number;
  /** Metadata of the most recent result, for feature-specific panels. */
  lastMetadata: Record<string, unknown> | null;
} | null>(null);

export function useFeaturePrompt() {
  return useContext(FeaturePromptContext);
}
