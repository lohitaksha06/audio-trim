"""Eval / metrics API — exposes prompt bench + model card for dashboard and CI."""

from fastapi import APIRouter

from server.ml.eval.metrics import genre_model_metrics, system_metrics
from server.ml.eval.prompt_bench import BENCH, evaluate_prompt_bench

router = APIRouter(prefix="/api/eval", tags=["eval"])


@router.get("/metrics")
async def get_metrics():
    """Full system metrics: genre model + prompt NLU bench."""
    return system_metrics()


@router.get("/prompt-bench")
async def get_prompt_bench():
    """Prompt benchmark results + dataset size."""
    result = evaluate_prompt_bench()
    return {"bench_size": len(BENCH), **result}


@router.get("/genre")
async def get_genre_metrics():
    return genre_model_metrics()


@router.get("/bench")
async def get_bench_items():
    return {"items": BENCH}
