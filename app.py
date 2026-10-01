"""
LazarTrack – Free Fire Match Stats Extraction Backend
Pure FastAPI Service.

Usage:
  - POST /api/v1/pipeline/match_stats  → structured JSON extraction
  - GET  /api/v1/health                → health check
"""

import os
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings, setup_logging
from app.api.router import api_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
log = logging.getLogger("lazartrack.backend")

# ---------------------------------------------------------------------------
# Detect hardware
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            log.info("GPU detected: %s", name)
            return f"GPU – {name}"
    except Exception as e:
        log.warning("GPU check failed: %s", e)
    log.info("No GPU detected – using CPU")
    return "CPU"

DEVICE = _detect_device()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "5-step intelligent extraction pipeline for Free Fire match "
        "scoreboard screenshots. Device: " + DEVICE
    ),
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "LazarTrack Match Stats Extraction API is running",
        "device": DEVICE,
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health"
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    log.info("Starting LazarTrack server on port %d | Device: %s", port, DEVICE)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)