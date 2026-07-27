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
  stems?: Record<string, string> | null;
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
