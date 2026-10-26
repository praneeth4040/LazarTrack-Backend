import asyncio
import logging
from typing import Callable, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, HttpUrl

from app.core.auth import AuthenticatedUser, require_auth
from app.core.config import settings
from app.pipelines.five_step import run_five_step_pipeline
from app.schemas.common import ExtractionResponse

log = logging.getLogger("app.api.v1.extract")
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

PIPELINE_ID   = "PSE-001"
PIPELINE_NAME = "Player Stats Extraction 1.0"


# ─── Request body ─────────────────────────────────────────────────────────────

class MatchStatsRequest(BaseModel):
    match_id:       str
    screenshot_url: str          # public URL already in match-assets bucket
    video_url:      Optional[str] = None   # reserved for future use


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _download_image(url: str) -> bytes:
    """Download an image from a public Supabase Storage URL."""
    # Block server-side fetches of arbitrary URLs (SSRF) — see ALLOWED_ASSET_HOSTS.
    from urllib.parse import urlparse
    allowed = [h.strip().lower() for h in settings.ALLOWED_ASSET_HOSTS.split(",") if h.strip()]
    if allowed:
        host = (urlparse(url).hostname or "").lower()
        if not host or not any(host == h or host.endswith(h) for h in allowed):
            raise HTTPException(
                status_code=400,
                detail=f"URL host is not allowed. Assets must be hosted on: {', '.join(allowed)}",
            )
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, follow_redirects=True)
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Could not download screenshot from storage: {e}",
            )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Storage returned HTTP {resp.status_code} for screenshot URL.",
        )

    raw = resp.content
    if not raw:
        raise HTTPException(status_code=422, detail="Downloaded screenshot is empty.")

    if len(raw) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Screenshot exceeds {settings.MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
        )

    return raw


def _confidence(parsed_data: dict) -> float:
    players = parsed_data.get("players", [])
    if len(players) == 0:
        return 0.2
    if len(players) < 3:
        return 0.6
    if parsed_data.get("rank", 0) <= 0:
        return 0.75
    return 1.0


# ─── Endpoints ────────────────────────────────────────────────────────────────

async def run_match_stats_extraction(
    match_id: str,
    screenshot_url: str,
    user: Optional[AuthenticatedUser] = None,
    progress: Optional[Callable[[str, str, Optional[float]], None]] = None,
) -> dict:
    """
    Shared runner for the 5-step extraction pipeline. Used by both the
    synchronous endpoint and the async Jobs API.

    Returns the ExtractionResponse payload as a JSON-serializable dict.
    """
    def _p(stage: str, message: str, percent: Optional[float] = None):
        if progress:
            progress(stage, message, percent)

    log.info(
        "[pipeline/match_stats] user=%s match_id=%s url=%s",
        getattr(user, "user_id", None), match_id, screenshot_url,
    )

    _p("downloading", "Downloading screenshot from storage…", 10)
    raw = await _download_image(screenshot_url)

    _p("extracting", "Running 5-step stats extraction…", 40)
    parsed_data = await asyncio.to_thread(
        run_five_step_pipeline, raw, debug=False
    )

    _p("done", "Extraction complete", 100)
    return ExtractionResponse(
        pipeline_id=PIPELINE_ID,
        pipeline_name=PIPELINE_NAME,
        confidence=_confidence(parsed_data),
        data={
            "map":         parsed_data.get("map", "Unknown"),
            "rank":        parsed_data.get("rank", 1),
            "total_teams": parsed_data.get("total_teams", 12),
            "players":     parsed_data.get("players", []),
            "metadata":    parsed_data.get("metadata", {}),
        },
        raw_texts=parsed_data.get("metadata", {}).get("map_ocr_texts", []),
    ).model_dump()


@router.post("/match_stats", response_model=ExtractionResponse)
async def pipeline_match_stats(
    body: MatchStatsRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    """
    Authenticated endpoint. Accepts a JSON body with match_id and screenshot_url.

    Flow:
      1. Verify JWT → extract user_id
      2. Download screenshot from Supabase Storage URL
      3. Run 5-step VL extraction pipeline
      4. Return structured stats

    The caller must send:
        Authorization: Bearer <supabase_access_token>

    Note: for long-running use prefer the async Jobs API
    (POST /api/v1/jobs/match-stats) which returns immediately.
    """
    return await run_match_stats_extraction(
        match_id=body.match_id,
        screenshot_url=body.screenshot_url,
        user=user,
    )


@router.post("/match_stats/debug")
async def pipeline_match_stats_debug(
    body: MatchStatsRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    """
    Authenticated debug endpoint. Same as /match_stats but returns
    per-step artifacts (crops, durations, raw OCR outputs).
    """
    log.info(
        "[pipeline/match_stats/debug] user=%s match_id=%s",
        user.user_id, body.match_id,
    )

    raw = await _download_image(body.screenshot_url)

    import asyncio
    parsed_data = await asyncio.to_thread(
        run_five_step_pipeline, raw, debug=True
    )

    confidence = _confidence(parsed_data)
    steps = parsed_data.pop("steps", [])

    return {
        "pipeline_id":   PIPELINE_ID,
        "pipeline_name": PIPELINE_NAME,
        "confidence":    confidence,
        "match_id":      body.match_id,
        "result": {
            "map":         parsed_data.get("map", "Unknown"),
            "rank":        parsed_data.get("rank", 1),
            "total_teams": parsed_data.get("total_teams", 12),
            "players":     parsed_data.get("players", []),
            "metadata":    parsed_data.get("metadata", {}),
        },
        "steps": steps,
    }
