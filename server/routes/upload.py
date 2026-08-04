import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

import server.services.storage as storage
from server.services.audio_processor import analyze_audio
from server.services.video_extractor import extract_audio

router = APIRouter(prefix="/api", tags=["upload"])

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename or "file").suffix.lower()

    if ext not in AUDIO_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {AUDIO_EXTENSIONS | VIDEO_EXTENSIONS}",
        )

    content = await file.read()
    storage_key = storage.save_upload(file.filename or "file", content)
    audio_path = storage.resolve(storage_key)

    is_video = ext in VIDEO_EXTENSIONS
    if is_video:
        try:
            extracted = extract_audio(audio_path)
            audio_key = storage.store_local_file(extracted, ext=".wav")
            audio_path = storage.resolve(audio_key)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract audio from video: {e}",
            )

    try:
        analysis = analyze_audio(audio_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze audio: {e}",
        )

    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "is_video": is_video,
        "storage_key": storage_key,
        "audio_path": audio_path,
        "analysis": analysis,
    }
