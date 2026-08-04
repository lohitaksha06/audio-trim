const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
  analysis: AnalysisResult;
}

export interface ProcessResponse {
  output_path?: string | null;
  download_key?: string | null;
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

export async function processAudio(audioPath: string, prompt: string): Promise<ProcessResponse> {
  const res = await fetch(`${API_BASE}/api/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_path: audioPath, prompt }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || "Processing failed");
  }

  return res.json();
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

export function downloadUrl(key: string): string {
  return `${API_BASE}/api/export/download?path=${encodeURIComponent(key)}`;
}
