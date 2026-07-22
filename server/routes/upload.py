import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from server.services.audio_processor import analyze_audio
from server.services.video_extractor import extract_audio

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR = Path(tempfile.gettempdir()) / "audio-trim-uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

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

    temp_path = UPLOAD_DIR / f"{os.urandom(8).hex()}{ext}"
    content = await file.read()
    temp_path.write_bytes(content)

    is_video = ext in VIDEO_EXTENSIONS
    audio_path = str(temp_path)

    if is_video:
        try:
            audio_path = extract_audio(temp_path)
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
        "audio_path": audio_path,
        "analysis": analysis,
    }
