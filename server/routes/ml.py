from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.ml.transcription.transcriber import Transcriber
from server.ml.source_separation.separator import SourceSeparator

router = APIRouter(prefix="/api/ml", tags=["ml"])

_separator: SourceSeparator | None = None
_transcriber: Transcriber | None = None


def get_separator() -> SourceSeparator:
    global _separator
    if _separator is None:
        _separator = SourceSeparator()
    return _separator


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber


class TranscribeRequest(BaseModel):
    audio_path: str


class TranscribeResponse(BaseModel):
    text: str
    duration_seconds: float
    model: str


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(req: TranscribeRequest):
    try:
        transcriber = get_transcriber()
        result = transcriber.transcribe(req.audio_path)
        return TranscribeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


class SeparateRequest(BaseModel):
    audio_path: str
    stems: list[str] | None = None


class SeparateResponse(BaseModel):
    stems: dict[str, str]
    sources: list[str]


@router.post("/separate", response_model=SeparateResponse)
async def separate_audio(req: SeparateRequest):
    try:
        separator = get_separator()
        all_stems = separator.separate(req.audio_path)
        return SeparateResponse(
            stems=all_stems,
            sources=separator.sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Separation failed: {e}")


@router.get("/separate/sources")
async def list_sources():
    separator = get_separator()
    return {"sources": separator.sources}
