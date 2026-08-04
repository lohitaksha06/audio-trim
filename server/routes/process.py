from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import server.services.storage as storage
from server.ml.prompt_engine import plan_from_prompt
from server.ml.audio_operations import execute_plan

router = APIRouter(prefix="/api", tags=["process"])


class ProcessRequest(BaseModel):
    audio_path: str
    prompt: str


class ProcessResponse(BaseModel):
    output_path: str | None = None
    download_key: str | None = None
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
        result = execute_plan(req.audio_path, plan)

        download_key = None
        stems_keys = None
        if result.get("output_path"):
            download_key = storage.store_local_file(result["output_path"])
        if result.get("stems"):
            stems_keys = {
                name: storage.store_local_file(p, ext=".wav")
                for name, p in result["stems"].items()
            }

        return ProcessResponse(
            output_path=result.get("output_path"),
            download_key=download_key,
            stems=result.get("stems"),
            stems_keys=stems_keys,
            intent=result["intent"],
            params=result["params"],
            raw_prompt=plan.raw_prompt,
            metadata=result.get("metadata"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
