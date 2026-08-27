from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

from server.ml.lab_dsp import apply_lab_tweaks, suggest_lab_preset
from server.ml.audio_understanding import describe_mood, predict_genre
import server.services.storage as storage

router = APIRouter(prefix="/api/lab", tags=["lab"])


class TweakRequest(BaseModel):
    audio_path: str
    gains_db: Optional[Dict[str, float]] = None
    lowpass_hz: Optional[float] = None
    highpass_hz: Optional[float] = None
    pitch_semitones: float = 0.0
    speed: float = 1.0
    gain_db: float = 0.0
    reverb_mix: float = 0.0


class TweakResponse(BaseModel):
    output_path: str
    download_key: str
    params: dict


class SuggestRequest(BaseModel):
    audio_path: str


@router.post("/tweak", response_model=TweakResponse)
async def lab_tweak(req: TweakRequest):
    try:
        out = apply_lab_tweaks(
            req.audio_path,
            gains_db=req.gains_db,
            lowpass_hz=req.lowpass_hz,
            highpass_hz=req.highpass_hz,
            pitch_semitones=req.pitch_semitones,
            speed=req.speed,
            gain_db=req.gain_db,
            reverb_mix=req.reverb_mix,
        )
        key = storage.store_local_file(out, ext=".wav")
        return TweakResponse(
            output_path=out,
            download_key=key,
            params=req.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lab tweak failed: {e}")


@router.post("/suggest")
async def lab_suggest(req: SuggestRequest):
    try:
        mood = describe_mood(req.audio_path)
        genre = None
        try:
            genre = predict_genre(req.audio_path)
        except Exception:
            genre = None
        understand = {"mood": mood, "genre": genre}
        preset = suggest_lab_preset(understand)
        return {"preset": preset, "mood": mood, "genre": genre}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggest failed: {e}")
