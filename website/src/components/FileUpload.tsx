"use client";

import { useCallback, useRef, useState } from "react";

interface FileUploadProps {
  onFileSelected: (file: File) => void;
}

export default function FileUpload({ onFileSelected }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      const validAudio = [
        "audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac",
        "audio/aac", "audio/mp4", "audio/ogg", "audio/x-m4a",
      ];
      const validVideo = [
        "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska",
      ];
      const ext = file.name.split(".").pop()?.toLowerCase();
      const validExt = ["mp3", "wav", "flac", "aac", "m4a", "ogg", "mp4", "mov", "avi", "mkv"];

      if (!validAudio.includes(file.type) && !validVideo.includes(file.type) && !validExt.includes(ext || "")) {
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onClick={() => inputRef.current?.click()}
      className={`group cursor-pointer rounded-2xl border-2 border-dashed p-10 sm:p-14 lg:p-16 transition-all duration-300 ${
        dragging
          ? "border-neon-blue bg-neon-blue/5 shadow-[0_0_40px_rgba(0,212,255,0.15)]"
          : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
      }`}
    >
      <div className="flex flex-col items-center">
        <div className="mb-5 flex h-16 w-16 sm:h-20 sm:w-20 items-center justify-center rounded-full bg-white/5 group-hover:bg-white/10 transition-colors">
          <svg
            className="h-8 w-8 sm:h-10 sm:w-10 text-white/40"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 16V4m0 0L8 8m4-4l4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2"
            />
          </svg>
        </div>
        <p className="mb-1.5 text-lg sm:text-xl font-medium text-white/70">
          Drop your audio or video file here
        </p>
        <p className="mb-5 text-sm sm:text-base text-white/30">
          or click to browse
        </p>
        <div className="flex flex-wrap items-center justify-center gap-2 text-xs sm:text-sm text-white/20">
          {["MP3", "WAV", "FLAC", "M4A", "MP4", "MOV", "AVI", "MKV"].map((fmt) => (
            <span key={fmt} className="rounded-md border border-white/10 px-2.5 py-1">
              {fmt}
            </span>
          ))}
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept=".mp3,.wav,.flac,.aac,.m4a,.ogg,.mp4,.mov,.avi,.mkv,audio/*,video/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}
