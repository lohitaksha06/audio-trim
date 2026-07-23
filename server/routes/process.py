from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.ml.prompt_engine import plan_from_prompt
from server.ml.audio_operations import execute_plan

router = APIRouter(prefix="/api", tags=["process"])


class ProcessRequest(BaseModel):
    audio_path: str
    prompt: str


class ProcessResponse(BaseModel):
    output_path: str
    intent: str
    params: dict
    raw_prompt: str


@router.post("/process", response_model=ProcessResponse)
async def process_audio(req: ProcessRequest):
    try:
        plan = plan_from_prompt(req.prompt)
        output_path = execute_plan(req.audio_path, plan)
        return ProcessResponse(
            output_path=output_path,
            intent=plan.intent.value,
            params=plan.params,
            raw_prompt=plan.raw_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
