// Backend origin.
//
// Locally the FastAPI backend binds 127.0.0.1 (not `localhost`, which resolves
// to ::1 first on this machine and then fails to connect).
//
// In production this must be set via NEXT_PUBLIC_API_URL. A hard-coded
// 127.0.0.1 fallback is actively harmful on a public deploy: every visitor's
// browser would try *their own* machine and fail with an opaque network error.
// So when the variable is absent we report that plainly rather than pretending.
const CONFIGURED_API = process.env.NEXT_PUBLIC_API_URL?.trim();
const API_BASE = CONFIGURED_API || "http://127.0.0.1:8000";

/** True when this build has no backend to talk to. */
export const isBackendConfigured = Boolean(CONFIGURED_API);

export const BACKEND_MISSING_MESSAGE =
  "No backend is configured for this deployment. Set NEXT_PUBLIC_API_URL to a running Audelle backend, " +
  "or run the DSP tools client-side (no backend needed).";

export interface AnalysisResult {
  duration_seconds: number;
  bpm: number;
  key: string;
  energy: number;
  sample_rate: number;
  channels: number;
}

export interface UploadResponse {
  filename: string;
  size_bytes: number;
  is_video: boolean;
  audio_path: string;
  /** Untouched original, present only for video uploads, so an edit can be
   *  muxed back onto it. */
  source_video_path?: string | null;
  analysis: AnalysisResult;
}

export interface ProcessResponse {
  output_path?: string | null;
  download_key?: string | null;
  /** Edited audio muxed back onto the source video (video uploads only). */
  video_download_key?: string | null;
  layer_download_key?: string | null;
  layer_label?: string | null;
  stems?: Record<string, string> | null;
  stems_keys?: Record<string, string> | null;
  intent: string;
  params: Record<string, unknown>;
  raw_prompt: string;
  metadata?: Record<string, unknown> | null;
}

export interface TranscribeResponse {
  text: string;
  duration_seconds: number;
  model: string;
}

export interface SeparateResponse {
  stems: Record<string, string>;
  sources: string[];
}

export interface InstrumentResult {
  instrument: string;
  confidence: number;
}

export interface StructureSection {
  start: number;
  end: number;
  label: string;
}

export interface CurvePoint {
  t: number;
  energy: number;
  tension: number;
  brightness: number;
  loudness: number;
}

export interface GenreInfo {
  genre: string;
  confidence: number;
  suggested_actions: string[];
}

export interface UnderstandResponse {
  instruments: {
    instruments: InstrumentResult[];
    texture: string;
    tempo_bpm: number;
    duration_seconds: number;
  };
  structure: {
    sections: StructureSection[];
    structure: string;
    duration_seconds: number;
  };
  mood: {
    mood: string;
    energy_mean: number;
    brightness_mean: number;
    tension_mean: number;
    description: string;
    duration_seconds: number;
  };
  energy_curve: {
    curve: CurvePoint[];
    sample_rate: number;
    hop_seconds: number;
    duration_seconds: number;
  };
  genre?: GenreInfo | null;
  rhythm?: {
    tempo_bpm: number;
    beat_count: number;
    four_on_floor: number;
    halftime: number;
    swing: number;
    auto_groove: string;
    groove_confidence: number;
    groove_source: string;
  } | null;
}

export interface JobResponse {
  id: string;
  status: string;
  ready: boolean;
  success: boolean;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export interface ZipResponse {
  key: string;
  path: string;
  files: string[];
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  if (!isBackendConfigured) throw new Error(BACKEND_MISSING_MESSAGE);
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Upload failed");
  }

  return res.json();
}

export async function processAudio(audioPath: string, prompt: string, opts?: { stemPath?: string | null; stemLevel?: number | null; sourceVideoPath?: string | null }): Promise<ProcessResponse> {
  if (!isBackendConfigured) throw new Error(BACKEND_MISSING_MESSAGE);
  const res = await fetch(`${API_BASE}/api/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_path: audioPath,
      prompt,
      stem_path: opts?.stemPath ?? null,
      stem_level: opts?.stemLevel ?? null,
      source_video_path: opts?.sourceVideoPath ?? null,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Processing failed");
  }

  return res.json();
}

export async function fetchSuggestions(prompt: string): Promise<string[]> {
  try {
    const res = await fetch(`${API_BASE}/api/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    if (!res.ok) return [];
    const j = await res.json();
    return Array.isArray(j.suggestions) ? j.suggestions : [];
  } catch {
    // Suggestions are a nicety: never let this break the prompt flow.
    return [];
  }
}

export async function transcribeAudio(audioPath: string): Promise<TranscribeResponse> {
  const res = await fetch(`${API_BASE}/api/ml/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_path: audioPath }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Transcription failed");
  }

  return res.json();
}

export async function separateAudio(audioPath: string): Promise<SeparateResponse> {
  const res = await fetch(`${API_BASE}/api/ml/separate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_path: audioPath }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Separation failed");
  }

  return res.json();
}

export async function getSources(): Promise<{ sources: string[] }> {
  const res = await fetch(`${API_BASE}/api/ml/separate/sources`);

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Failed to get sources");
  }

  return res.json();
}

export async function understandAudio(audioPath: string): Promise<UnderstandResponse> {
  const res = await fetch(`${API_BASE}/api/ml/understand`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_path: audioPath }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Understanding failed");
  }

  return res.json();
}

export async function submitJob(type: string, payload: Record<string, unknown>): Promise<{ job_id: string }> {
  const res = await fetch(`${API_BASE}/api/jobs/${type}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Failed to submit job");
  }

  return res.json();
}

export async function getJob(jobId: string): Promise<JobResponse> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error("Failed to get job status");
  return res.json();
}

export async function exportZip(paths: string[]): Promise<ZipResponse> {
  const res = await fetch(`${API_BASE}/api/export/zip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths }),
  });
  if (!res.ok) throw new Error("Export failed");
  return res.json();
}

export interface EvalMetrics {
  genre: { available: boolean; classes?: string[]; feature_size?: number; model_type?: string; artifact_bytes?: number; reason?: string };
  prompt_bench: { accuracy: number; total: number; correct: number; param_accuracy?: number | null; per_intent: Record<string, { accuracy: number; correct: number; total: number }> };
}

export async function getEvalMetrics(): Promise<EvalMetrics> {
  const res = await fetch(`${API_BASE}/api/eval/metrics`);
  if (!res.ok) throw new Error("Failed to get eval metrics");
  return res.json();
}

export function downloadUrl(key: string): string {
  return `${API_BASE}/api/export/download?path=${encodeURIComponent(key)}`;
}

export interface MixTip {
  severity: "high" | "medium" | "low" | "good";
  title: string;
  detail: string;
  fix_prompt: string;
}

export interface OptimizeResponse {
  summary: {
    peak: number;
    rms: number;
    crest_db: number;
    dynamics_db: number;
    clipped_pct: number;
    bands: { bass: number; lowmid: number; presence: number; air: number };
    stereo_width: number;
    channels: number;
    duration_seconds: number;
    sample_rate: number;
  };
  score: number;
  tips: MixTip[];
}

export async function optimizeAudio(audioPath: string): Promise<OptimizeResponse> {
  const res = await fetch(`${API_BASE}/api/ml/optimize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_path: audioPath }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Optimization failed");
  }

  return res.json();
}

export interface CatalogResponse {
  instruments: { id: string; name: string; family: string; blurb: string }[];
  waves: { id: string; aliases: string[] }[];
  grooves: string[];
  mix_stems: string[];
}

export async function getCatalog(): Promise<CatalogResponse> {
  const res = await fetch(`${API_BASE}/api/ml/catalog`);
  if (!res.ok) throw new Error("Failed to get catalog");
  return res.json();
}
