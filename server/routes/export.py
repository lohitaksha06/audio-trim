"""Export + download endpoints (stems ZIP, FCPXML, EDL, file serving)."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import server.services.storage as storage
from server.services.exporter import export_zip, export_fcpxml, export_edl

router = APIRouter(prefix="/api/export", tags=["export"])


class ZipRequest(BaseModel):
    paths: list[str]


class FCPXMLRequest(BaseModel):
    segments: list[dict]
    duration: float
    project_name: str = "Audelle Export"


class EDLRequest(BaseModel):
    segments: list[dict]
    duration: float
    fps: int = 30


@router.post("/zip")
async def export_stems_zip(req: ZipRequest):
    try:
        return export_zip(req.paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ZIP export failed: {e}")


@router.post("/fcpxml")
async def export_final_cut(req: FCPXMLRequest):
    try:
        return export_fcpxml(req.segments, req.duration, req.project_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FCPXML export failed: {e}")


@router.post("/edl")
async def export_premiere_resolve(req: EDLRequest):
    try:
        return export_edl(req.segments, req.duration, req.fps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDL export failed: {e}")


class DownloadQuery(BaseModel):
    path: str = ""


@router.get("/download")
async def download(path: str):
    """Serve a storage-resolved file. Caller passes a storage key or local path."""
    if not path:
        raise HTTPException(status_code=400, detail="Missing 'path' parameter")
    try:
        local = storage.resolve(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    filename = Path(local).name
    media_type = "application/octet-stream"
    if filename.endswith(".zip"):
        media_type = "application/zip"
    elif filename.endswith((".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a")):
        media_type = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    elif filename.endswith(".fcpxml") or filename.endswith(".xml"):
        media_type = "application/xml"
    elif filename.endswith(".edl"):
        media_type = "text/plain"
    return FileResponse(local, media_type=media_type, filename=filename)