import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes.upload import router as upload_router
from server.routes.process import router as process_router
from server.routes.ml import router as ml_router
from server.routes.jobs import router as jobs_router
from server.routes.export import router as export_router
from server.routes.projects import router as projects_router
from server.routes.lab import router as lab_router
from server.routes.eval import router as eval_router

app = FastAPI(
    title="Audelle API",
    description="AI-powered audio editing via natural language prompts",
    version="0.1.0",
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
    return {"status": "ok", "version": "0.1.0"}
