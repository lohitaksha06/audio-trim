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
  output_path: string;
  intent: string;
  params: Record<string, unknown>;
  raw_prompt: string;
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
