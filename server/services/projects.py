"""Project persistence for saved sessions.

Projects let users save an upload + AI analysis + edit history and restore them
later. Stored as JSON files under the same storage dir as uploads so the whole
backend stays swappable (local disk now, S3 later).

Each project references storage keys (audio_path, result keys) so no audio bytes
are duplicated here.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from server.services.storage import STORAGE_DIR

PROJECTS_DIR = Path(STORAGE_DIR) / "projects"

_ID_RE = re.compile(r"^[a-z0-9-]{8,64}$")


def _path(project_id: str) -> Path:
    return PROJECTS_DIR / f"{project_id}.json"


def _now() -> float:
    return time.time()


def _sanitize(project_id: str) -> str:
    if not _ID_RE.match(project_id):
        raise ValueError("Invalid project id")
    return project_id


def create_project(data: dict[str, Any]) -> dict[str, Any]:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    project_id = uuid.uuid4().hex[:16]
    now = _now()
    project = {
        "id": project_id,
        "name": data.get("name") or "Untitled project",
        "audio_path": data.get("audio_path"),
        "filename": data.get("filename"),
        "analysis": data.get("analysis") or None,
        "understand": data.get("understand") or None,
        "history": data.get("history") or [],
        "created_at": now,
        "updated_at": now,
    }
    _path(project_id).write_text(json.dumps(project, indent=2), encoding="utf-8")
    return project


def get_project(project_id: str) -> dict[str, Any]:
    p = _path(_sanitize(project_id))
    if not p.exists():
        raise KeyError("Project not found")
    return json.loads(p.read_text(encoding="utf-8"))


def list_projects() -> list[dict[str, Any]]:
    if not PROJECTS_DIR.exists():
        return []
    projects = []
    for p in PROJECTS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        projects.append(
            {
                "id": data.get("id"),
                "name": data.get("name"),
                "filename": data.get("filename"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "history_count": len(data.get("history") or []),
            }
        )
    projects.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
    return projects


def update_project(project_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    project = get_project(project_id)
    for key in ("name", "audio_path", "filename", "analysis", "understand"):
        if key in patch and patch[key] is not None:
            project[key] = patch[key]

    entries = patch.get("history")
    if isinstance(entries, list):
        existing = list(project.get("history") or [])
        if patch.get("append_history"):
            existing.extend(entries)
        else:
            existing = entries
        project["history"] = existing[-500:]

    project["updated_at"] = _now()
    _path(project_id).write_text(json.dumps(project, indent=2), encoding="utf-8")
    return project


def delete_project(project_id: str) -> bool:
    p = _path(_sanitize(project_id))
    if p.exists():
        p.unlink()
        return True
    return False