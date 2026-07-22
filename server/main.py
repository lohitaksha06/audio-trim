from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes.upload import router as upload_router

app = FastAPI(
    title="Audio Trim API",
    description="AI-powered audio editing via natural language prompts",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
