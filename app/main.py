from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings, setup_logging
from app.api.router import api_router

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Jobs left queued/running by a previous process cannot be resumed —
    # mark them failed so clients see an actionable state.
    from app.services.jobs import job_manager
    job_manager.recover()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Intelligent Extraction & Safe Zone Tracking Suite for Free Fire esports matches.",
    redoc_url=None,
    lifespan=lifespan,
)

# CORS: restrict browser access to the web app's known origins. Auth is Bearer
# tokens (no cookies), so this is not the primary defense — it stops other
# origins from making credentialed-looking calls and leaking responses to
# arbitrary web pages. Set ALLOWED_ORIGINS="*" only for local testing.
_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
if "*" in _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_origin_regex=settings.ALLOWED_ORIGIN_REGEX or None,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import logging
    logging.getLogger("app.main").exception(f"Unhandled server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


@app.get("/", summary="API Root & Available Endpoints Directory")
async def root():
    """Root endpoint exposing API status and complete directory of available endpoints."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "interactive_docs": "/docs",
        "endpoint_catalog": {
            "zone_and_map_tracker": {
                "process_video": "POST /api/v1/zone/process (Upload MP4 or provide video_path + map_name)",
                "list_sessions": "GET /api/v1/zone/list (List all past processed sessions)",
                "session_status": "GET /api/v1/zone/{video_id}/status",
                "zones_metadata": "GET /api/v1/zone/{video_id}/zones (Raw detected zones JSON)",
                "zones_map_image": "GET /api/v1/zone/{video_id}/zones-map (PNG image of rendered safe zones)",
                "step_artifact": "GET /api/v1/zone/{video_id}/step/{step_num}/{artifact_name}",
                "visual_inspection_dashboard": "GET /api/v1/zone/{video_id}/visual (Interactive 8-step visual verification)",
                "reference_maps": "GET /api/v1/zone/maps (List supported map templates)",
            },
            "battle_log_extraction": {
                "process": "POST /api/v1/battle-log/extract",
            },
            "match_stats_ocr": {
                "extract": "POST /api/v1/pipeline/match_stats",
                "extract_debug": "POST /api/v1/pipeline/match_stats/debug",
                "visualizer": "GET /api/v1/visualize",
            },
            "system_health": {
                "health": "GET /api/v1/health",
            },
        },
    }


@app.get("/endpoints", summary="Dynamic Endpoints Listing")
async def list_all_endpoints():
    """Dynamic list of all registered HTTP routes."""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods - {"HEAD", "OPTIONS"}),
                "name": getattr(route, "name", ""),
            })
    return {"total_routes": len(routes), "routes": sorted(routes, key=lambda x: x["path"])}


app.include_router(api_router, prefix=settings.API_V1_STR)
