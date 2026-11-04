"use client";

/**
 * One on-device project shared by every feature page.
 *
 * The old feature pages each decoded their own copy of the upload, so moving
 * between pages (or refreshing) threw the work away. This provider owns the
 * file, the decoded audio, the current edited audio and the download blob.
 * IndexedDB keeps the source and the latest render across a hard refresh.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  analyzeLocalAsync,
  decodeFile,
  encodeWav,
  type LocalAnalysis,
  type LocalAudio,
} from "@/lib/audio";
import {
  isBackendConfigured,
  uploadFile,
  type AnalysisResult,
  type UploadResponse,
} from "@/services/api";
import {
  clearProjectFile,
  clearProjectOutput,
  loadProjectFile,
  loadProjectMeta,
  loadProjectOutput,
  saveProjectFile,
  saveProjectMeta,
  saveProjectOutput,
} from "@/lib/storage";

interface ProjectContextValue {
  file: File | null;
  fileName: string | null;
  objectUrl: string | null;
  /** Current editable audio (original, or the latest local render). */
  audio: LocalAudio | null;
  /** The untouched decoded upload. */
  original: LocalAudio | null;
  upload: UploadResponse | null;
  analysis: AnalysisResult | undefined;
  loading: boolean;
  error: string;
  /** Latest local render, ready for a download. */
  currentBlob: Blob | null;
  /** Changes whenever the current audio/URL changes. */
  version: number;
  localMode: boolean;
  selectFile: (file: File, persist?: boolean) => Promise<void>;
  replaceAudio: (audio: LocalAudio, blob?: Blob, name?: string) => void;
  resetToOriginal: () => void;
  clearProject: () => Promise<void>;
  setError: (message: string) => void;
}

const ProjectContext = createContext<ProjectContextValue | null>(null);

function localUpload(file: File, analysis: AnalysisResult): UploadResponse {
  return {
    filename: file.name,
    size_bytes: file.size,
    is_video: false,
    audio_path: `local:${file.name}`,
    analysis,
  };
}

function analysisFromLocal(
  file: File,
  info: LocalAnalysis,
): AnalysisResult {
  return {
    duration_seconds: info.durationSeconds,
    bpm: info.bpm ?? 0,
    key: "—",
    energy: info.rms,
    sample_rate: info.sampleRate,
    channels: info.channels,
  };
}

export function ProjectProvider({ children }: { children: ReactNode }) {
  const localMode = !isBackendConfigured;
  const [file, setFile] = useState<File | null>(null);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);
  const [audio, setAudio] = useState<LocalAudio | null>(null);
  const [original, setOriginal] = useState<LocalAudio | null>(null);
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [currentBlob, setCurrentBlob] = useState<Blob | null>(null);
  const [version, setVersion] = useState(0);
  const urlRef = useRef<string | null>(null);
  const mountedRef = useRef(true);

  const swapUrl = useCallback((blob: Blob | null) => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = blob ? URL.createObjectURL(blob) : null;
    setObjectUrl(urlRef.current);
  }, []);

  const bump = useCallback(() => setVersion((v) => v + 1), []);

  const selectFile = useCallback(
    async (picked: File, persist = true) => {
      setError("");
      setLoading(true);
      setFile(picked);
      swapUrl(picked);
      setCurrentBlob(null);
      setUpload(null);
      setAnalysis(undefined);
      setAudio(null);
      setOriginal(null);
      await clearProjectOutput();

      let serverUpload: UploadResponse | null = null;
      try {
        if (localMode) {
          const decoded = await decodeFile(picked);
          const info = await analyzeLocalAsync(decoded);
          const measured = analysisFromLocal(picked, info);
          const local = localUpload(picked, measured);
          if (!mountedRef.current) return;
          setOriginal(decoded);
          setAudio(decoded);
          setUpload(local);
          setAnalysis(measured);
        } else {
          const uploaded = await uploadFile(picked);
          serverUpload = uploaded;
          if (!mountedRef.current) return;
          setUpload(uploaded);
          setAnalysis(uploaded.analysis);
          // Decode locally too, so the waveform and the browser-side preview
          // work even when the file lives on the server.
          try {
            const decoded = await decodeFile(picked);
            if (!mountedRef.current) return;
            setOriginal(decoded);
            setAudio(decoded);
          } catch {
            /* Video or unsupported codec: server rendering still works. */
          }
        }
        if (persist) {
          await saveProjectFile(picked);
          saveProjectMeta({
            name: picked.name,
            type: picked.type,
            lastModified: picked.lastModified,
            serverUpload: serverUpload ?? undefined,
          });
        }
      } catch (err) {
        if (mountedRef.current) {
          setError(err instanceof Error ? err.message : "Could not read that file.");
        }
      } finally {
        if (mountedRef.current) setLoading(false);
      }
    },
    [localMode, swapUrl],
  );

  // Restore the last project once on mount.
  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;
    (async () => {
      const stored = await loadProjectFile();
      if (cancelled) return;
      if (!stored) {
        setLoading(false);
        return;
      }
      setFile(stored);
      swapUrl(stored);
      setLoading(true);
      const meta = loadProjectMeta();
      try {
        let decoded: LocalAudio | null = null;
        try {
          decoded = await decodeFile(stored);
        } catch {
          decoded = null;
        }
        if (cancelled) return;
        if (decoded) {
          const info = await analyzeLocalAsync(decoded);
          if (cancelled) return;
          const measured = analysisFromLocal(stored, info);
          setOriginal(decoded);
          setAudio(decoded);
          setAnalysis(measured);
          // Browser projects use the local pseudo-path. Server projects wait
          // for the stored upload path (or the re-upload below) so an edit can
          // never be posted to "local:<name>".
          if (localMode) setUpload(localUpload(stored, measured));
        }
        const serverUpload = meta?.serverUpload as UploadResponse | undefined;
        if (!localMode && serverUpload?.audio_path) {
          setUpload(serverUpload);
          setAnalysis(serverUpload.analysis ?? undefined);
        } else if (!localMode && !serverUpload) {
          // A previous server upload has no reusable path: upload once more.
          const uploaded = await uploadFile(stored);
          if (cancelled) return;
          setUpload(uploaded);
          setAnalysis(uploaded.analysis);
        }

        const output = await loadProjectOutput();
        if (cancelled) return;
        if (output?.blob) {
          try {
            const outFile = new File([output.blob], output.name, { type: "audio/wav" });
            const outAudio = await decodeFile(outFile);
            if (cancelled) return;
            setAudio(outAudio);
            setCurrentBlob(output.blob);
            swapUrl(output.blob);
          } catch {
            /* Keep the original if the cached render is unreadable. */
          }
        }
      } catch {
        /* Keep whatever decoded successfully. */
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
      mountedRef.current = false;
    };
  }, [localMode, swapUrl]);

  // Release the object URL on unmount.
  useEffect(() => {
    return () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, []);

  const replaceAudio = useCallback(
    (nextAudio: LocalAudio, blob?: Blob, name = "audelle-render.wav") => {
      const out = blob ?? encodeWav(nextAudio);
      setAudio(nextAudio);
      setCurrentBlob(out);
      swapUrl(out);
      bump();
      void saveProjectOutput(out, name);
      const meta = loadProjectMeta();
      if (meta) saveProjectMeta({ ...meta, lastAction: name });
    },
    [bump, swapUrl],
  );

  const resetToOriginal = useCallback(() => {
    if (!original) return;
    setAudio(original);
    setCurrentBlob(null);
    if (file) swapUrl(file);
    else swapUrl(encodeWav(original));
    void clearProjectOutput();
    bump();
  }, [bump, file, original, swapUrl]);

  const clearProject = useCallback(async () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setObjectUrl(null);
    setFile(null);
    setAudio(null);
    setOriginal(null);
    setUpload(null);
    setAnalysis(undefined);
    setCurrentBlob(null);
    setError("");
    bump();
    await clearProjectFile();
  }, [bump]);

  const value = useMemo<ProjectContextValue>(
    () => ({
      file,
      fileName: file?.name ?? null,
      objectUrl,
      audio,
      original,
      upload,
      analysis,
      loading,
      error,
      currentBlob,
      version,
      localMode,
      selectFile,
      replaceAudio,
      resetToOriginal,
      clearProject,
      setError,
    }),
    [
      file,
      objectUrl,
      audio,
      original,
      upload,
      analysis,
      loading,
      error,
      currentBlob,
      version,
      localMode,
      selectFile,
      replaceAudio,
      resetToOriginal,
      clearProject,
    ],
  );

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>;
}

export function useProject(): ProjectContextValue {
  const ctx = useContext(ProjectContext);
  if (!ctx) throw new Error("useProject must be used inside ProjectProvider");
  return ctx;
}
