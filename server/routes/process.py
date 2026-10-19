from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import server.services.storage as storage
from server.ml.prompt_engine import plan_from_prompt, suggest_alternatives
from server.ml.audio_operations import execute_plan

router = APIRouter(prefix="/api", tags=["process"])


class ProcessRequest(BaseModel):
    audio_path: str
    prompt: str
    stem_path: str | None = None
    stem_level: float | None = None
    # Original upload was a video. When set and the plan produces audio, the
    # result is muxed back onto the source video so the user gets a video out.
    source_video_path: str | None = None


class ProcessResponse(BaseModel):
    output_path: str | None = None
    download_key: str | None = None
    video_download_key: str | None = None
    layer_download_key: str | None = None
    layer_label: str | None = None
    stems: dict[str, str] | None = None
    stems_keys: dict[str, str] | None = None
    intent: str
    params: dict
    raw_prompt: str
    metadata: dict | None = None


class SuggestRequest(BaseModel):
    prompt: str


class SuggestResponse(BaseModel):
    suggestions: list[str]


@router.post("/suggest", response_model=SuggestResponse)
async def suggest_prompts(req: SuggestRequest):
    """Concrete follow-up prompts for a vague or unrecognised request.

    Keeps the "what can it actually do?" logic in one place -- the keyword
    tables live in the engine, so duplicating them in the browser would drift.
    """
    try:
        return SuggestResponse(
            suggestions=suggest_alternatives(req.prompt)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggest failed: {e}")


@router.post("/process", response_model=ProcessResponse)
async def process_audio(req: ProcessRequest):
    try:
        plan = plan_from_prompt(req.prompt)
        # imported-stem file supplied alongside the prompt ("mix my stem in")
        if req.stem_path:
            plan.params["stem_path"] = req.stem_path
        if req.stem_level is not None:
            plan.params["stem_level"] = req.stem_level
        # bare "mix my stem" phrasing without the file attached yet
        from server.ml.prompt_engine import Intent as _Intent

        if plan.intent == _Intent.MIX_STEM and not plan.params.get("stem_path"):
            raise HTTPException(
                status_code=400,
                detail="Upload your stem file first, then press Mix — I need the stem audio to BPM-match it.",
            )
        if plan.intent == _Intent.SPEED and plan.params.get("needs_bpm"):
            raise HTTPException(
                status_code=400,
                detail="What BPM do you want? Tell me e.g. 'Set tempo to 128 BPM' or 'Slow down to 90 BPM'.",
            )
        # chaining: accept a previous download_key as readily as a server path
        audio_src = req.audio_path
        if not Path(audio_src).is_file():
            try:
                audio_src = storage.resolve(audio_src)
            except Exception:
                pass
        result = execute_plan(audio_src, plan)

        download_key = None
        layer_download_key = None
        layer_label = None
        stems_keys = None
        if result.get("output_path"):
            download_key = storage.store_local_file(result["output_path"])
        if result.get("layer_path"):
            layer_download_key = storage.store_local_file(result["layer_path"], ext=".wav")
            layer_label = (result.get("metadata") or {}).get("layer_kind", "added layer")
        if result.get("stems"):
            stems_keys = {
                name: storage.store_local_file(p, ext=".wav")
                for name, p in result["stems"].items()
            }

        # Video: if the original upload was a video and this edit produced
        # audio, write the result back into the source video. The audio-only
        # download above still stands, so nothing is lost if muxing fails.
        video_download_key = None
        if req.source_video_path and result.get("output_path"):
            try:
                from server.services.video_extractor import mux_audio_onto_video

                src_video = req.source_video_path
                if not Path(src_video).is_file():
                    src_video = storage.resolve(src_video)
                muxed = mux_audio_onto_video(src_video, result["output_path"])
                video_download_key = storage.store_local_file(muxed, ext=".mp4")
            except Exception as e:
                # Never fail the whole edit over the convenience video.
                result.setdefault("metadata", {})["video_mux_error"] = str(e)[:200]

        return ProcessResponse(
            output_path=result.get("output_path"),
            download_key=download_key,
            video_download_key=video_download_key,
            layer_download_key=layer_download_key,
            layer_label=layer_label,
            stems=result.get("stems"),
            stems_keys=stems_keys,
            intent=result["intent"],
            params=result["params"],
            raw_prompt=plan.raw_prompt,
            metadata=result.get("metadata"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
