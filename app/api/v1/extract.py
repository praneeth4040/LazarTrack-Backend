import logging
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.schemas.common import ExtractionResponse
from app.pipelines.five_step import run_five_step_pipeline
from app.core.config import settings

log = logging.getLogger("app.api.v1.extract")
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/jpg",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/heic",
    "image/heif",
    "application/octet-stream",
}

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".heic", ".heif"
}

PIPELINE_ID = "PSE-001"  # PSE = Player Stats Extraction, 001 = pipeline version/identifier
PIPELINE_NAME = "Player Stats Extraction 1.0"


async def _validate_and_read(file: UploadFile) -> bytes:
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()

    is_allowed = (
        not content_type
        or content_type in ALLOWED_TYPES
        or content_type.startswith("image/")
        or any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)
    )

    if not is_allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: '{file.content_type}'. Please upload a valid image file.",
        )
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file received.")
    if len(raw) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds limit of {settings.MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )
    return raw


def _confidence_from_result(parsed_data: dict) -> float:
    players = parsed_data.get("players", [])
    if len(players) == 0:
        return 0.2
    if len(players) < 3:
        return 0.6
    if parsed_data.get("rank", 0) <= 0:
        return 0.75
    return 1.0


@router.post("/match_stats", response_model=ExtractionResponse)
async def pipeline_match_stats(
    file: UploadFile = File(..., description="Free Fire match scoreboard screenshot"),
):
    """
    Main production endpoint. Runs the 5-step extraction pipeline on a screenshot
    and returns clean structured match stats (map, rank, total_teams, players[]).
    """
    raw = await _validate_and_read(file)

    log.info("[pipeline/match_stats] Running for '%s' (%d bytes)...",
             file.filename or "screenshot.jpg", len(raw))
    parsed_data = run_five_step_pipeline(raw, debug=False)

    return ExtractionResponse(
        pipeline_id=PIPELINE_ID,
        pipeline_name=PIPELINE_NAME,
        confidence=_confidence_from_result(parsed_data),
        data={
            "map": parsed_data.get("map", "Unknown"),
            "rank": parsed_data.get("rank", 1),
            "total_teams": parsed_data.get("total_teams", 12),
            "players": parsed_data.get("players", []),
            "metadata": parsed_data.get("metadata", {}),
        },
        raw_texts=parsed_data.get("metadata", {}).get("map_ocr_texts", []),
    )


@router.post("/match_stats/debug")
async def pipeline_match_stats_debug(
    file: UploadFile = File(..., description="Free Fire match scoreboard screenshot"),
):
    """
    Debug endpoint. Runs the exact same 5-step extraction and returns everything:
    the structured output PLUS a detailed "steps" array with per-step results
    (duration, meta, base64-encoded crop images at each stage, per-player per-cell
    raw crops, scaled crops, raw OCR outputs, and final parsed values).
    """
    raw = await _validate_and_read(file)

    log.info("[pipeline/match_stats/debug] Running for '%s' (%d bytes)...",
             file.filename or "screenshot.jpg", len(raw))
    parsed_data = run_five_step_pipeline(raw, debug=True)

    confidence = _confidence_from_result(parsed_data)
    steps = parsed_data.pop("steps", [])

    return {
        "pipeline_id": PIPELINE_ID,
        "pipeline_name": PIPELINE_NAME,
        "confidence": confidence,
        "result": {
            "map": parsed_data.get("map", "Unknown"),
            "rank": parsed_data.get("rank", 1),
            "total_teams": parsed_data.get("total_teams", 12),
            "players": parsed_data.get("players", []),
            "metadata": parsed_data.get("metadata", {}),
        },
        "steps": steps,
    }


@router.post("/match_stats/compare")
async def pipeline_match_stats_compare(
    file: UploadFile = File(..., description="Free Fire match scoreboard screenshot"),
):
    """
    Comparison endpoint: Runs extraction twice (1. Deployed HF EasyOCR, 2. Local EasyOCR)
    and returns side-by-side results and detailed debug steps for visual comparison.
    """
    from app.ocr.local_easyocr import get_easyocr_reader, get_raw_local_easyocr_reader
    raw = await _validate_and_read(file)

    log.info("[pipeline/match_stats/compare] Running comparison for '%s'...", file.filename or "screenshot.jpg")

    # 1. Deployed HF EasyOCR (with local fallback wrapper)
    deployed_reader = get_easyocr_reader()
    deployed_data = run_five_step_pipeline(raw, easy_reader=deployed_reader, debug=True)
    deployed_steps = deployed_data.pop("steps", [])

    # 2. Pure Local EasyOCR
    local_reader = get_raw_local_easyocr_reader()
    local_data = run_five_step_pipeline(raw, easy_reader=local_reader, debug=True)
    local_steps = local_data.pop("steps", [])

    return {
        "pipeline_id": PIPELINE_ID,
        "pipeline_name": PIPELINE_NAME,
        "deployed": {
            "name": "Deployed EasyOCR (HF Space)",
            "result": deployed_data,
            "steps": deployed_steps,
        },
        "local": {
            "name": "Local EasyOCR",
            "result": local_data,
            "steps": local_steps,
        },
    }
