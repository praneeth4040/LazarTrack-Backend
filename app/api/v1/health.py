from fastapi import APIRouter
from app.schemas.common import HealthResponse
from app.core.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint returning system status and configured pipeline ID."""
    return HealthResponse(
        status="ok",
        easyocr_url=settings.EASYOCR_URL,
        pipelines_available=["player_stats_extraction"],
    )
