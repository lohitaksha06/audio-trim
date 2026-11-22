"""File storage abstraction: local disk by default, optional S3.

Local files are stored under ``AUDIO_STORAGE_DIR`` (default
``<temp>/audelle-data``). If ``boto3`` is importable and AWS credentials are
present, uploads are mirrored to S3. All callers use ``key`` strings so the
backend can swap without touching business logic.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

STORAGE_DIR = os.environ.get(
    "AUDIO_STORAGE_DIR", str(Path(tempfile.gettempdir()) / "audelle-storage")
)

_BUCKET = os.environ.get("AWS_S3_BUCKET")


def _s3():
    if not _BUCKET:
        return None
    try:
        import boto3
    except ImportError:
        return None
    try:
        return boto3.client("s3")
    except Exception:
        return None


def save_upload(filename: str, content: bytes) -> str:
    """Persist an uploaded file, returning a storage key."""
    ext = Path(filename).suffix.lower() if filename else ".bin"
    key = f"uploads/{uuid.uuid4().hex}{ext}"
    local = Path(STORAGE_DIR) / key
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(content)

    client = _s3()
    if client:
        try:
            client.put_object(Bucket=_BUCKET, Key=key, Body=content)
        except Exception:
            pass  # local copy remains the source of truth
    return key


def resolve(key: str) -> str:
    """Return a local filesystem path for a storage key."""
    key = key.strip("/")
    local = Path(STORAGE_DIR) / key
    if local.exists():
        return str(local)

    client = _s3()
    if client:
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(_BUCKET, key, str(local))
            return str(local)
        except Exception:
            pass

    raise FileNotFoundError(f"Storage key not found: {key}")


def store_local_file(src_path: str, ext: str = ".wav") -> str:
    """Copy a generated file (e.g. an output WAV) into storage."""
    Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
    key = f"exports/{uuid.uuid4().hex}{ext}"
    dst = Path(STORAGE_DIR) / key
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst)
    return key


def delete(key: str) -> bool:
    local = Path(STORAGE_DIR) / key.strip("/")
    if local.exists():
        local.unlink()
        return True
    return False


def stats() -> dict:
    """Small storage summary for the health endpoint."""
    root = Path(STORAGE_DIR)
    if not root.exists():
        return {"files": 0, "bytes": 0, "uploads": 0, "exports": 0}
    files = [p for p in root.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return {
        "files": len(files),
        "bytes": total,
        "uploads": sum(1 for p in files if p.relative_to(root).parts[0] == "uploads"),
        "exports": sum(1 for p in files if p.relative_to(root).parts[0] == "exports"),
    }


def cleanup(max_age_hours: float = 24.0) -> dict:
    """Delete uploads/exports older than ``max_age_hours``.

    Runs on a timer so a long-lived host does not fill its disk. Returns
    ``{removed, bytes}``. A non-positive age disables deletion.
    """
    import time

    if max_age_hours <= 0:
        return {"removed": 0, "bytes": 0}
    root = Path(STORAGE_DIR)
    if not root.exists():
        return {"removed": 0, "bytes": 0}
    cutoff = time.time() - max_age_hours * 3600.0
    removed = 0
    freed = 0
    for sub in ("uploads", "exports"):
        folder = root / sub
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    freed += path.stat().st_size
                    path.unlink()
                    removed += 1
            except OSError:
                # A file can disappear between listing and stat/unlink; ignore.
                continue
    return {"removed": removed, "bytes": freed}
