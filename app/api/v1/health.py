from fastapi import APIRouter
from app.schemas.common import HealthResponse
from app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint returning system status and configured pipeline IDs."""
    pipelines = ["player_stats_extraction", "zone_tracker"]

    if settings.OPENROUTER_API_KEY:
        pipelines.append("battle_log_extraction")

    return HealthResponse(
        status="ok",
        ocr_engine="local_easyocr",
        pipelines_available=pipelines,
    )
