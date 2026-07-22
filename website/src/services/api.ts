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
