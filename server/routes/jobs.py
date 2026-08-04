"""Async job endpoints for long-running ML tasks."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.ml.prompt_engine import plan_from_prompt
from server.ml.audio_operations import execute_plan
from server.ml.transcription.transcriber import Transcriber
from server.ml.source_separation.separator import SourceSeparator
from server.ml.audio_understanding import (
    classify_instruments,
    detect_structure,
    compute_mood_curve,
    describe_mood,
)
from server.ml.diarization import diarize
from server.ml.inpainting import inpaint
from server.services.jobs import job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def _understand_job(audio_path):
    return {
        "instruments": classify_instruments(audio_path),
        "structure": detect_structure(audio_path),
        "mood": describe_mood(audio_path),
        "energy_curve": compute_mood_curve(audio_path),
    }


class ProcessJobRequest(BaseModel):
    audio_path: str
    prompt: str


class PathJobRequest(BaseModel):
    audio_path: str


class InpaintJobRequest(BaseModel):
    audio_path: str
    start: float
    end: float


@router.post("/process")
async def submit_process(req: ProcessJobRequest):
    plan = plan_from_prompt(req.prompt)
    job_id = job_manager.submit(lambda p, plan: execute_plan(p, plan), req.audio_path, plan)
    return {"job_id": job_id}


@router.post("/separate")
async def submit_separate(req: PathJobRequest):
    job_id = job_manager.submit(
        lambda p: {"stems": SourceSeparator().separate(p)}, req.audio_path
    )
    return {"job_id": job_id}


@router.post("/transcribe")
async def submit_transcribe(req: PathJobRequest):
    job_id = job_manager.submit(lambda p: Transcriber().transcribe(p), req.audio_path)
    return {"job_id": job_id}


@router.post("/understand")
async def submit_understand(req: PathJobRequest):
    job_id = job_manager.submit(_understand_job, req.audio_path)
    return {"job_id": job_id}


@router.post("/diarize")
async def submit_diarize(req: PathJobRequest):
    job_id = job_manager.submit(diarize, req.audio_path)
    return {"job_id": job_id}


@router.post("/inpaint")
async def submit_inpaint(req: InpaintJobRequest):
    job_id = job_manager.submit(inpaint, req.audio_path, req.start, req.end)
    return {"job_id": job_id}


@router.get("/{job_id}")
async def job_status(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_manager.to_dict(job)