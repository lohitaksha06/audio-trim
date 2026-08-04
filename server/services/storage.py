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