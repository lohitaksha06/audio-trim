import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

import server.services.storage as storage
from server.services.audio_processor import analyze_audio
from server.services.video_extractor import extract_audio, normalize_to_wav

router = APIRouter(prefix="/api", tags=["upload"])

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"}

# Formats libsndfile cannot decode. Verified on this machine:
# soundfile.available_formats() has no AAC/M4A entry, and audioread offers only
# RawAudioFile because no ffmpeg/avconv/gstreamer is on PATH. These are
# transcoded to WAV on upload instead of failing the whole request.
NEEDS_FFMPEG_DECODE = {".m4a", ".aac", ".m4b", ".opus", ".wma"}


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
    elif ext in NEEDS_FFMPEG_DECODE:
        # Decode to WAV first. Without this the analysis below raises and the
        # upload 500s, leaving the UI showing a filename with no audio loaded.
        try:
            decoded = normalize_to_wav(audio_path)
            audio_key = storage.store_local_file(decoded, ext=".wav")
            audio_path = storage.resolve(audio_key)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not decode {ext}. This machine has no system ffmpeg, "
                    f"so AAC/M4A is decoded with the bundled imageio-ffmpeg build; "
                    f"that step failed: {e}"
                ),
            )

    try:
        analysis = analyze_audio(audio_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Saved the file but could not read audio from it "
                f"({ext}): {e}. Try WAV, FLAC, OGG or MP3."
            ),
        )

    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "is_video": is_video,
        "storage_key": storage_key,
        "audio_path": audio_path,
        # For a video upload this is the untouched original, kept so an edit
        # can be muxed back onto it and the user gets a video out.
        "source_video_path": storage.resolve(storage_key) if is_video else None,
        "analysis": analysis,
    }
