"use client";

import { useCallback, useRef, useState } from "react";
import { isBackendConfigured } from "@/services/api";

interface FileUploadProps {
  onFileSelected: (file: File) => void;
  onInvalid?: (msg: string) => void;
}

export default function FileUpload({ onFileSelected, onInvalid }: FileUploadProps) {
  const localMode = !isBackendConfigured;
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
      const audioExt = ["mp3", "wav", "flac", "aac", "m4a", "ogg"];
      const videoExt = ["mp4", "mov", "avi", "mkv"];
      const validExt = localMode ? audioExt : [...audioExt, ...videoExt];

      if (localMode && (validVideo.includes(file.type) || videoExt.includes(ext || ""))) {
        onInvalid?.("Video import needs the server. Upload MP3, WAV, FLAC, M4A, AAC or OGG.");
        return;
      }
      if (!validAudio.includes(file.type) && !validVideo.includes(file.type) && !validExt.includes(ext || "")) {
        onInvalid?.(
          localMode
            ? `Unsupported file type: ${file.name}. Use MP3, WAV, FLAC, M4A, AAC or OGG.`
            : `Unsupported file type: ${file.name}. Use MP3, WAV, FLAC, M4A, MP4, MOV, AVI or MKV.`,
        );
        return;
      }
      onFileSelected(file);
    },
    [localMode, onFileSelected, onInvalid],
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
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inputRef.current?.click(); } }}
      role="button"
      tabIndex={0}
      aria-label={localMode ? "Upload audio file" : "Upload audio or video file"}
      className={`group relative cursor-pointer rounded-none border border-dashed p-10 sm:p-14 lg:p-16 transition-colors duration-300 ${
        dragging
          ? "border-white/60 bg-white/[0.04]"
          : "border-white/20 bg-black hover:border-white/40 hover:bg-white/[0.02]"
      }`}
    >
      <div className="flex flex-col items-center">
        <div className="mb-5 flex h-16 w-16 sm:h-20 sm:w-20 items-center justify-center border border-white/15 bg-white/[0.03] transition-colors group-hover:border-white/30">
          <svg
            className="h-8 w-8 sm:h-10 sm:w-10 text-white/60"
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
        <p className="mb-1.5 font-display text-xl font-bold text-[#f2f1ec] sm:text-3xl">
          {localMode ? "Drop your audio file here" : "Drop your audio or video file here"}
        </p>
        <p className="mb-5 text-sm text-white/50 sm:text-base">
          or click to browse
        </p>
        <div className="flex flex-wrap items-center justify-center gap-2 font-mono text-[0.65rem] uppercase tracking-[0.14em] text-white/45">
          {(localMode ? ["MP3", "WAV", "FLAC", "M4A", "AAC", "OGG"] : ["MP3", "WAV", "FLAC", "M4A", "MP4", "MOV", "AVI", "MKV"]).map((fmt) => (
            <span key={fmt} className="rounded-none border border-white/15 px-2.5 py-1">
              {fmt}
            </span>
          ))}
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={localMode ? ".mp3,.wav,.flac,.aac,.m4a,.ogg,audio/*" : ".mp3,.wav,.flac,.aac,.m4a,.ogg,.mp4,.mov,.avi,.mkv,audio/*,video/*"}
        aria-hidden={false}
        tabIndex={-1}
        className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
        onChange={(e) => {
          const file = e.target.files?.[0];
          // reset so picking the same file twice still fires onChange
          e.target.value = "";
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}
