"""Minimal in-process async job queue for long-running ML tasks.

Runs jobs on a thread pool so heavy inference (Demucs, Whisper, enlisting)
does not block other requests. Job state is kept in memory and pollable via
``get_job``. Swappable later for Celery/Redis if the server is scaled out.
"""

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

_MAX_WORKERS = 2


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | completed | failed
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None


class JobManager:
    def __init__(self, max_workers: int = _MAX_WORKERS):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable, *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id)
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run, job, fn, args, kwargs)
        return job_id

    def _run(self, job: Job, fn: Callable, args, kwargs):
        job.status = "running"
        job.started_at = time.time()
        try:
            job.result = fn(*args, **kwargs)
            job.status = "completed"
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def to_dict(self, job: Job) -> dict:
        return {
            "id": job.id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "result": job.result,
            "error": job.error,
            "ready": job.status in {"completed", "failed"},
            "success": job.status == "completed",
        }


job_manager = JobManager()