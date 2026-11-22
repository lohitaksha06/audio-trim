import asyncio
import contextlib
import logging
import os
import time
from contextlib import asynccontextmanager

# Cap BLAS/torch threads before any ML module loads. One CPU-only job should
# not peg every core on a small host (or the user's laptop).
_THREADS = os.environ.get("AUDIDELLE_TORCH_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.routes.upload import router as upload_router
from server.routes.process import router as process_router
from server.routes.ml import router as ml_router
from server.routes.jobs import router as jobs_router
from server.routes.export import router as export_router
from server.routes.projects import router as projects_router
from server.routes.lab import router as lab_router
from server.routes.eval import router as eval_router
from server.services import limits, storage

logger = logging.getLogger("audelle")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

STARTED_AT = time.time()

# Paths that do real work; these get the rate limit. Cheap reads do not.
_RATE_LIMITED_PREFIXES = (
    "/api/upload",
    "/api/process",
    "/api/ml",
    "/api/export",
    "/api/lab",
    "/api/suggest",
    "/api/projects",
)


async def _cleanup_loop() -> None:
    """Delete old uploads/exports on a timer so a long-lived host stays small."""
    try:
        retention_hours = float(os.environ.get("AUDIDELLE_RETENTION_HOURS", "24"))
    except ValueError:
        retention_hours = 24.0
    try:
        interval_min = float(os.environ.get("AUDIDELLE_CLEANUP_INTERVAL_MIN", "60"))
    except ValueError:
        interval_min = 60.0
    interval = max(60.0, interval_min * 60.0)
    while True:
        try:
            result = await asyncio.to_thread(storage.cleanup, retention_hours)
            if result.get("removed"):
                logger.info(
                    "storage cleanup: removed %s files (%s bytes)",
                    result["removed"], result["bytes"],
                )
        except Exception:
            logger.exception("storage cleanup failed")
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        import torch

        torch.set_num_threads(max(1, int(_THREADS)))
    except Exception:
        pass
    task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="Audelle API",
    description="AI-powered audio editing via natural language prompts",
    version="0.1.0",
    lifespan=lifespan,
)

# Set AUDIDELLE_CORS_ORIGINS to a comma-separated allowlist in production.
# The fallback is permissive so local tools and previews still work; browsers
# reject a wildcard together with credentials, so credentials are only enabled
# when an explicit allowlist is configured.
_origins_env = os.environ.get("AUDIDELLE_CORS_ORIGINS", "").strip()
_allowed_origins = [origin.strip() for origin in _origins_env.split(",") if origin.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=bool(_origins_env),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_and_rate_limit(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path
    if (
        request.method in ("POST", "PUT", "PATCH")
        and path.startswith(_RATE_LIMITED_PREFIXES)
    ):
        key = limits.client_key(request.headers, request.client)
        allowed, retry_after = limits.check(key)
        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down."},
                headers={"Retry-After": str(retry_after)},
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
            logger.warning("%s %s 429 %.1fms", request.method, path, elapsed)
            return response
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    logger.info("%s %s %s %.1fms", request.method, path, response.status_code, elapsed)
    return response


app.include_router(upload_router)
app.include_router(process_router)
app.include_router(ml_router)
app.include_router(jobs_router)
app.include_router(export_router)
app.include_router(projects_router)
app.include_router(lab_router)
app.include_router(eval_router)


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "storage": storage.stats(),
        "rate_limit_per_min": limits.limit_per_minute(),
    }
