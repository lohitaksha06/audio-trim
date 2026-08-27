"use client";
import { createContext, useContext } from "react";

export const FeaturePromptContext = createContext<{
  setPrompt: (p: string) => void;
  triggerPrompt?: (p: string) => void;
} | null>(null);

export function useFeaturePrompt() {
  return useContext(FeaturePromptContext);
}
