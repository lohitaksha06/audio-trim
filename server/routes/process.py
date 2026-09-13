from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import server.services.storage as storage
from server.ml.prompt_engine import plan_from_prompt
from server.ml.audio_operations import execute_plan

router = APIRouter(prefix="/api", tags=["process"])


class ProcessRequest(BaseModel):
    audio_path: str
    prompt: str
    stem_path: str | None = None
    stem_level: float | None = None


class ProcessResponse(BaseModel):
    output_path: str | None = None
    download_key: str | None = None
    layer_download_key: str | None = None
    layer_label: str | None = None
    stems: dict[str, str] | None = None
    stems_keys: dict[str, str] | None = None
    intent: str
    params: dict
    raw_prompt: str
    metadata: dict | None = None


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

        return ProcessResponse(
            output_path=result.get("output_path"),
            download_key=download_key,
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
