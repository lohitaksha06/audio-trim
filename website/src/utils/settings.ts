"use client";

import { useCallback, useEffect, useState } from "react";

export interface AudelleSettings {
  /** Run AI analysis automatically right after upload (else the user presses "Run AI Analysis"). */
  autoAnalyze: boolean;
  /** Download results lossless (WAV). Off = convert to the chosen output format first. */
  highQuality: boolean;
  /** Target format when High quality is off. */
  outputFormat: string;
}

const KEY = "audelle:settings";

export const DEFAULT_SETTINGS: AudelleSettings = {
  autoAnalyze: true,
  highQuality: true,
  outputFormat: "mp3",
};

export function getSettings(): AudelleSettings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function useSettings(): [AudelleSettings, (patch: Partial<AudelleSettings>) => void] {
  // Lazy initializer reads persisted settings without a render-loop effect.
  // (getSettings guards localStorage for server prerendering.)
  const [settings, setSettings] = useState<AudelleSettings>(getSettings);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY) setSettings(getSettings());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const update = useCallback((patch: Partial<AudelleSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      try {
        localStorage.setItem(KEY, JSON.stringify(next));
      } catch {}
      return next;
    });
  }, []);

  return [settings, update];
}
