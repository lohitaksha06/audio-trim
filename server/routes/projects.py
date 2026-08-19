"""Project save/load endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.services.projects import (
    create_project,
    get_project,
    list_projects,
    update_project,
    delete_project,
)

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    name: str | None = None
    audio_path: str | None = None
    filename: str | None = None
    analysis: dict | None = None
    understand: dict | None = None
    history: list | None = None


class UpdateProjectRequest(BaseModel):
    name: str | None = None
    audio_path: str | None = None
    filename: str | None = None
    analysis: dict | None = None
    understand: dict | None = None
    history: list | None = None
    append_history: bool = False


@router.post("")
async def create_project_route(req: CreateProjectRequest):
    try:
        return create_project(req.model_dump(exclude_none=True))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save project: {e}")


@router.get("")
async def list_projects_route():
    return {"projects": list_projects()}


@router.get("/{project_id}")
async def get_project_route(project_id: str):
    try:
        return get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project id")


@router.put("/{project_id}")
async def update_project_route(project_id: str, req: UpdateProjectRequest):
    try:
        return update_project(project_id, req.model_dump(exclude_none=True))
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project id")


@router.delete("/{project_id}")
async def delete_project_route(project_id: str):
    if not delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"ok": True}