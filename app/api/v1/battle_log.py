"""
Battle Log Video Processing API Endpoints
"""

import asyncio
import os
import logging
import tempfile
import shutil
import uuid
import json
from typing import List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from app.core.auth import AuthenticatedUser, require_auth
from app.core.config import settings
from app.pipelines.video_pipeline import run_complete_video_pipeline, run_step7_on_session
from app.services.supabase_storage import storage_client

log = logging.getLogger("app.api.v1.battle_log")
router = APIRouter(prefix="/battle-log", tags=["Battle Log"])

PIPELINE_OUTPUTS = {}


# ─── Request / Response models ────────────────────────────────────────────────

class BattleLogRequest(BaseModel):
    match_id:  str
    video_url: str        # public URL of the raw video already in match-assets bucket
    run_step7: bool = False


class BattleLogEvent(BaseModel):
    event_id: int
    time: Optional[str]
    type: str
    actor: Optional[str]
    targets: List[str]
    kills: Optional[int]
    zone: Optional[str]
    item: Optional[str]
    text: Optional[str]
    review: bool
    flags: List[str]
    source: str


class BattleLogResponse(BaseModel):
    success: bool
    video_id: str
    events: List[BattleLogEvent] = Field(default_factory=list)
    team_slots: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    download_urls: dict = Field(default_factory=dict)
    storage_path: Optional[str] = None



async def _download_video(url: str) -> bytes:
    """Download the raw battle log video from Supabase Storage."""
    _ensure_allowed_asset_host(url)
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(url, follow_redirects=True)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Could not download video: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Storage returned HTTP {resp.status_code} for video URL.",
        )

    raw = resp.content
    if not raw:
        raise HTTPException(status_code=422, detail="Downloaded video is empty.")

    max_size = 500 * 1024 * 1024  # 500 MB
    if len(raw) > max_size:
        raise HTTPException(status_code=413, detail="Video exceeds 500 MB limit.")

    return raw


import asyncio

def _ensure_allowed_asset_host(url: str) -> None:
    """Block server-side fetches of arbitrary URLs (SSRF).

    Users supply the storage URL; without this check an authenticated caller
    could make the backend fetch internal endpoints and observe status/size.
    Allowed hosts are configured via ALLOWED_ASSET_HOSTS (suffix match).
    """
    from urllib.parse import urlparse
    allowed = [h.strip().lower() for h in settings.ALLOWED_ASSET_HOSTS.split(",") if h.strip()]
    if not allowed:
        return
    host = (urlparse(url).hostname or "").lower()
    if not host or not any(host == h or host.endswith(h) for h in allowed):
        raise HTTPException(
            status_code=400,
            detail=f"URL host is not allowed. Assets must be hosted on: {', '.join(allowed)}",
        )

def _get_pipeline_output(video_id: str) -> dict:
    """Retrieve session outputs from memory or persistent disk metadata."""
    if video_id in PIPELINE_OUTPUTS:
        return PIPELINE_OUTPUTS[video_id]

    # Check disk session metadata
    session_file = os.path.join(settings.VIDEO_OUTPUT_DIR, video_id, "session.json")
    if os.path.exists(session_file):
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
                PIPELINE_OUTPUTS[video_id] = data
                return data
        except Exception as e:
            log.warning(f"Could not load session.json for {video_id}: {e}")

    # Fallback: scan disk folder
    session_dir = os.path.join(settings.VIDEO_OUTPUT_DIR, video_id)
    if os.path.exists(session_dir):
        group_dir = os.path.join(session_dir, "group_images")
        group_images = sorted([
            os.path.join(group_dir, f) for f in os.listdir(group_dir)
            if f.startswith("group_") and f.endswith(".png")
        ]) if os.path.exists(group_dir) else []
        
        map_video = os.path.join(session_dir, "map_video.mp4")
        team_image = os.path.join(session_dir, "team_image.png")
        events_file = os.path.join(session_dir, "events.json")

        session_info = {
            "video_id": video_id,
            "output_dir": session_dir,
            "group_images": group_images,
            "map_video": map_video if os.path.exists(map_video) else None,
            "team_image": team_image if os.path.exists(team_image) else None,
            "events_file": events_file if os.path.exists(events_file) else None,
        }
        PIPELINE_OUTPUTS[video_id] = session_info
        return session_info

    raise HTTPException(status_code=404, detail=f"Video session '{video_id}' not found or expired")


def _require_session_owner(video_id: str, user: AuthenticatedUser) -> dict:
    """Load a session and enforce that it belongs to the requesting user."""
    session = _get_pipeline_output(video_id)
    owner = session.get("user_id")
    if not owner:
        raise HTTPException(
            status_code=403,
            detail="This session has no recorded owner and cannot be accessed.",
        )
    if owner != user.user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this video session.",
        )
    return session


def _cleanup_pipeline_output(video_id: str, output_dir: str):
    """Background task to cleanup old pipeline outputs"""
    try:
        if video_id in PIPELINE_OUTPUTS:
            del PIPELINE_OUTPUTS[video_id]
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            log.info(f"Cleaned up pipeline output for video_id={video_id}")
    except Exception as e:
        log.error(f"Error cleaning up video_id={video_id}: {e}")


def _upload_battle_log_to_supabase(
    user_id: str,
    match_id: str,
    output_dir: str,
    result: dict,
) -> dict:
    """
    Upload all battle log outputs to Supabase Storage
    
    Structure:
      {user_id}/{match_id}/battle_log/
        ├── group_images/
        │   ├── group_000.png
        │   ├── group_001.png
        │   └── ...
        ├── map_video.mp4
        ├── team_image.png
        └── events.json
    
    Returns dict with public URLs for all files
    """
    base_path = f"{user_id}/{match_id}/battle_log"
    urls = {}
    
    try:
        # Upload group images
        group_images = result.get("group_images", [])
        group_urls = []
        for group_img_path in group_images:
            if os.path.exists(group_img_path):
                filename = os.path.basename(group_img_path)
                remote_path = f"{base_path}/group_images/{filename}"
                url = storage_client.upload_file(group_img_path, remote_path, "image/png")
                if url:
                    group_urls.append({"name": filename, "url": url})
        urls["group_images"] = group_urls
        
        # Upload map video
        map_video_path = result.get("map_video_path")
        if map_video_path and os.path.exists(map_video_path):
            remote_path = f"{base_path}/map_video.mp4"
            url = storage_client.upload_file(map_video_path, remote_path, "video/mp4")
            urls["map_video"] = url
        
        # Upload team image
        team_image_path = result.get("team_image")
        if team_image_path and os.path.exists(team_image_path):
            remote_path = f"{base_path}/team_image.png"
            url = storage_client.upload_file(team_image_path, remote_path, "image/png")
            urls["team_image"] = url
        
        # Upload events JSON
        events_file_path = result.get("events_file_path")
        if events_file_path and os.path.exists(events_file_path):
            remote_path = f"{base_path}/events.json"
            url = storage_client.upload_file(events_file_path, remote_path, "application/json")
            urls["events_json"] = url
        
        log.info(f"✓ All battle log files uploaded to Supabase for {user_id}/{match_id}")
        return urls
        
    except Exception as e:
        log.error(f"Error uploading battle log to Supabase: {e}")
        return urls


# Strong references to fire-and-forget cleanup tasks so they are not
# garbage-collected mid-sleep.
_CLEANUP_TASKS: set = set()


async def run_battle_log_extraction(
    user_id: str,
    body: BattleLogRequest,
    progress=None,
) -> dict:
    """
    Shared runner for the 7-step battle log pipeline. Used by both the
    synchronous endpoint and the async Jobs API.

    Returns the BattleLogResponse payload as a JSON-serializable dict.
    """
    def _p(stage: str, message: str, percent: Optional[float] = None):
        if progress:
            progress(stage, message, percent)

    video_id = uuid.uuid4().hex  # full UUID — session URLs must not be guessable
    log.info(
        "[battle-log/extract] user=%s match_id=%s video_id=%s",
        user_id, body.match_id, video_id,
    )

    # Download video from bucket
    _p("downloading", "Downloading video from storage…", 5)
    raw_video = await _download_video(body.video_url)

    temp_dir = tempfile.mkdtemp(prefix=f"video_{video_id}_")
    input_video_path = os.path.join(temp_dir, f"input_{video_id}.mp4")

    try:
        with open(input_video_path, "wb") as f:
            f.write(raw_video)

        output_base_dir = os.path.join(settings.VIDEO_OUTPUT_DIR, video_id)
        os.makedirs(output_base_dir, exist_ok=True)

        _p("pipeline", "Running battle log pipeline (steps 1–6)…", 30)
        result = await asyncio.to_thread(
            run_complete_video_pipeline,
            video_path=input_video_path,
            video_id=video_id,
            output_base_dir=output_base_dir,
            cleanup_intermediate=True,
            run_step7=body.run_step7,
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Pipeline failed: {result.get('error')}")

        # Upload outputs to Supabase Storage
        _p("uploading", "Uploading results to storage…", 80)
        supabase_urls = await asyncio.to_thread(
            _upload_battle_log_to_supabase,
            user_id=user_id,
            match_id=body.match_id,
            output_dir=output_base_dir,
            result=result,
        )

        log.info(f"Supabase upload result: {supabase_urls}")

        session_info = {
            "video_id":    video_id,
            "user_id":     user_id,
            "match_id":    body.match_id,
            "output_dir":  output_base_dir,
            "group_images": result.get("group_images", []),
            "map_video":   result.get("map_video_path"),
            "team_image":  result.get("team_image"),
            "events_file": result.get("events_file_path"),
            "team_slots":  result.get("team_slots", {}),
            "metadata":    result.get("metadata", {}),
            "supabase_urls": supabase_urls,
        }
        PIPELINE_OUTPUTS[video_id] = session_info

        try:
            with open(os.path.join(output_base_dir, "session.json"), "w") as f:
                json.dump(session_info, f, indent=2)
        except Exception as se:
            log.warning(f"Could not write session.json: {se}")

        # Schedule local cleanup after 24 h
        async def delayed_cleanup():
            await asyncio.sleep(24 * 3600)
            _cleanup_pipeline_output(video_id, output_base_dir)
        cleanup_task = asyncio.create_task(delayed_cleanup())
        _CLEANUP_TASKS.add(cleanup_task)
        cleanup_task.add_done_callback(_CLEANUP_TASKS.discard)

        # Use Supabase URLs if successfully uploaded, otherwise use fallback relative URLs
        if supabase_urls and supabase_urls.get("group_images"):
            download_urls = supabase_urls
            log.info(f"Using Supabase URLs for battle log files")
        else:
            log.warning(f"Supabase upload didn't return group_images, using fallback URLs")
            download_urls = {
                "group_images": f"/api/v1/battle-log/{video_id}/group-images",
                "map_video":    f"/api/v1/battle-log/{video_id}/map-video" if result.get("map_video_path") else None,
                "team_image":   f"/api/v1/battle-log/{video_id}/team-image" if result.get("team_image") else None,
                "events_json":  f"/api/v1/battle-log/{video_id}/events.json",
            }

        # ── Update match.assets_url with pipeline URLs ────────────────────────────
        # Stores the Supabase URLs so the frontend can fetch them later. Uses the
        # raw PostgREST client (supabase-py's dependency stack conflicts with the
        # pinned pydantic — see app/services/supabase_db.py).
        _p("saving", "Updating match record…", 95)
        try:
            from app.services.supabase_db import merge_match_pipeline_assets

            # group_images: just URLs, no names
            pipeline_assets = {
                "group_images": [
                    img["url"]
                    for img in download_urls.get("group_images", [])
                ] if isinstance(download_urls.get("group_images"), list) else None,
                "map_video": download_urls.get("map_video"),
                "team_image": download_urls.get("team_image"),
                "events_json": download_urls.get("events_json"),
            }

            if merge_match_pipeline_assets(body.match_id, pipeline_assets):
                log.info(f"✓ Updated match.assets_url for {body.match_id} with pipeline URLs")
            else:
                log.warning("Could not update match.assets_url (see errors above)")

        except Exception as e:
            log.error(f"Error updating match.assets_url: {e}", exc_info=True)
            # Don't fail the extraction if database update fails
            pass

        _p("done", "Battle log extraction complete", 100)
        return BattleLogResponse(
            success=True,
            video_id=video_id,
            events=[BattleLogEvent(**e) for e in result.get("events", [])],
            team_slots=result.get("team_slots", {}),
            metadata=result.get("metadata", {}),
            download_urls=download_urls,
            storage_path=f"{user_id}/{body.match_id}/battle_log",
        ).model_dump()

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error processing video_id={video_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post("/extract", response_model=BattleLogResponse)
async def extract_battle_log(
    body: BattleLogRequest,
    user: AuthenticatedUser = Depends(require_auth),
):
    """
    Authenticated endpoint. Accepts a JSON body with match_id and video_url.

    Flow:
      1. Verify JWT → extract user_id
      2. Download the raw video from the Supabase Storage URL
      3. Run the 7-step battle log pipeline
      4. Upload all outputs back to Supabase Storage under
             {user_id}/{match_id}/battle_log/
      5. Return structured events + public download URLs

    The caller must send:
        Authorization: Bearer <supabase_access_token>

    Note: this call blocks for the full pipeline duration (5–10 min). For a
    fire-and-forget flow prefer the async Jobs API
    (POST /api/v1/jobs/battle-log) which returns a job id immediately.
    """
    return BattleLogResponse(
        **await run_battle_log_extraction(user.user_id, body)
    )


@router.post("/{video_id}/extract-events", summary="Run Step 7 Vision extraction on-demand")
async def extract_events_on_demand(
    video_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    """
    On-demand Step 7 execution:
    Runs the Vision Language model on the existing Step 6 group images in the session.
    Used by admin portal or downstream suggestion engine builder.
    """
    session = _require_session_owner(video_id, user)
    output_base_dir = session.get("output_dir")
    if not output_base_dir or not os.path.exists(output_base_dir):
        raise HTTPException(status_code=404, detail="Session directory not found")

    result = await asyncio.to_thread(
        run_step7_on_session,
        session_dir_path=output_base_dir,
        video_id=video_id,
    )

    # Reload session info with newly generated events
    events_file = os.path.join(output_base_dir, "events.json")
    if os.path.exists(events_file):
        session["events_file"] = events_file
        PIPELINE_OUTPUTS[video_id] = session

    return result


@router.get("/{video_id}/group-images")
async def get_group_images(
    video_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    """Get list of group images from step 6.

    Returns the public Supabase URLs when the outputs were uploaded, so the
    client can render them directly; falls back to relative backend URLs.
    """
    session = _require_session_owner(video_id, user)
    group_images = session.get("group_images", [])

    supabase_images = (session.get("supabase_urls") or {}).get("group_images")
    if supabase_images:
        return {
            "video_id": video_id,
            "group_count": len(supabase_images),
            "images": [
                {"name": img.get("name"), "url": img.get("url")}
                for img in supabase_images
            ],
        }

    return {
        "video_id": video_id,
        "group_count": len(group_images),
        "images": [
            {
                "name": os.path.basename(img),
                "url": f"/api/v1/battle-log/{video_id}/group-image/{os.path.basename(img)}"
            }
            for img in group_images
        ]
    }


@router.get("/{video_id}/group-image/{image_name}")
async def download_group_image(
    video_id: str,
    image_name: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    """Download a specific group image"""
    session = _require_session_owner(video_id, user)
    group_images = session.get("group_images", [])
    image_path = next((img for img in group_images if os.path.basename(img) == image_name), None)
    
    # Also check directly in session_dir/group_images
    if not image_path:
        candidate = os.path.join(session["output_dir"], "group_images", image_name)
        if os.path.exists(candidate):
            image_path = candidate
    
    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image {image_name} not found")
    
    return FileResponse(image_path, media_type="image/png", filename=image_name)


@router.get("/{video_id}/map-video")
async def download_map_video(
    video_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    """Download the normalized video from step 1"""
    session = _require_session_owner(video_id, user)
    map_video = session.get("map_video")
    
    if not map_video:
        candidate = os.path.join(session["output_dir"], "map_video.mp4")
        if os.path.exists(candidate):
            map_video = candidate
    
    if not map_video or not os.path.exists(map_video):
        raise HTTPException(status_code=404, detail="Map video not available")
    
    return FileResponse(map_video, media_type="video/mp4", filename=f"map_{video_id}.mp4")


@router.get("/{video_id}/team-image")
async def download_team_image(
    video_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    """Download the team identification frame"""
    session = _require_session_owner(video_id, user)
    team_image = session.get("team_image")
    
    if not team_image:
        candidate = os.path.join(session["output_dir"], "team_image.png")
        if os.path.exists(candidate):
            team_image = candidate
    
    if not team_image or not os.path.exists(team_image):
        raise HTTPException(status_code=404, detail="Team image not found")
    
    return FileResponse(team_image, media_type="image/png", filename=f"team_{video_id}.png")


@router.get("/{video_id}/events.json")
async def download_events_json(
    video_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    """Download the complete events JSON file"""
    session = _require_session_owner(video_id, user)
    events_file = session.get("events_file")
    
    if not events_file:
        candidate = os.path.join(session["output_dir"], "events.json")
        if os.path.exists(candidate):
            events_file = candidate
    
    if not events_file or not os.path.exists(events_file):
        raise HTTPException(status_code=404, detail="Events file not found")
    
    return FileResponse(events_file, media_type="application/json", filename=f"events_{video_id}.json")


@router.delete("/{video_id}")
async def cleanup_video(
    video_id: str,
    user: AuthenticatedUser = Depends(require_auth),
):
    """Manually cleanup/delete all outputs for a processed video"""
    session = _require_session_owner(video_id, user)
    output_dir = session["output_dir"]
    _cleanup_pipeline_output(video_id, output_dir)
    
    return {"message": f"Video {video_id} outputs cleaned up successfully"}


# Visualize UI (simple HTML interface)
@router.get("/visualize", response_class=HTMLResponse, include_in_schema=False)
async def visualize_ui():
    """Browser UI for battle log extraction: upload video, see extracted events"""
    return HTMLResponse(content=_VISUALIZE_HTML)


_VISUALIZE_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Battle Log Extraction · Visualizer</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #111827;
    --border: #1f2937;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #6366f1;
    --good: #22c55e;
    --warn: #f59e0b;
    --bad: #ef4444;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: ui-sans-serif, system-ui, sans-serif; }
  header { padding: 20px 28px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0 0 4px 0; font-size: 20px; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  main { max-width: 1200px; margin: 0 auto; padding: 24px 28px 80px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; margin-bottom: 20px; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }
  label.file-label {
    display: inline-flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: 10px;
    border: 1px dashed #374151; color: var(--muted); cursor: pointer; font-size: 14px;
  }
  label.file-label:hover { border-color: var(--accent); color: var(--text); }
  input[type=file] { display: none; }
  button.primary {
    background: var(--accent); color: white; border: none; padding: 10px 18px; border-radius: 10px;
    font-size: 14px; font-weight: 600; cursor: pointer;
  }
  button.primary:hover:not(:disabled) { background: #4f46e5; }
  button.primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .spinner { width: 14px; height: 14px; display: inline-block; border: 2px solid rgba(255,255,255,0.35);
    border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .stat-box { background: #0f172a; border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .stat-box .k { font-size: 11px; text-transform: uppercase; color: var(--muted); }
  .stat-box .v { font-size: 20px; font-weight: 700; margin-top: 4px; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 12px; }
  h2 { margin: 8px 0 14px; font-size: 15px; color: var(--muted); text-transform: uppercase; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  table th, table td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; }
  table th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
  .pill.good { background: rgba(34,197,94,0.12); color: #86efac; }
  .pill.warn { background: rgba(245,158,11,0.12); color: #fcd34d; }
  .pill.bad { background: rgba(239,68,68,0.12); color: #fca5a5; }
  .pill.info { background: rgba(34,211,238,0.12); color: #67e8f9; }
  .hidden { display: none !important; }
  .error { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.35); color: #fecaca;
    padding: 12px 14px; border-radius: 10px; font-size: 13px; }
  .img-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
  .img-frame { background: #000; border: 1px solid var(--border); border-radius: 10px; padding: 8px; }
  .img-frame img { width: 100%; height: auto; display: block; border-radius: 6px; }
  .img-frame .caption { margin-top: 6px; font-size: 11px; color: var(--muted); font-family: monospace; }
</style>
</head>
<body>
<header>
  <h1>🎮 Battle Log Extraction · Visualizer</h1>
  <p>Upload a Free Fire battle log video → extract all events, team roster, and view group images</p>
</header>
<main>
  <section class="card">
    <div class="row">
      <label class="file-label" for="file">
        <span>📹 Choose Video (MP4 / MOV / AVI)</span>
      </label>
      <input id="file" type="file" accept="video/mp4,video/quicktime,video/x-msvideo" />
      <div id="fileName" style="color:var(--muted);font-size:13px">No file selected</div>
      <div style="flex:1"></div>
      <button id="runBtn" class="primary" disabled>Extract Battle Log</button>
    </div>
  </section>

  <section id="statusCard" class="card hidden">
    <div id="statusText" style="font-size:14px;color:var(--muted)">Processing...</div>
  </section>

  <section id="errorSection" class="hidden"></section>

  <section id="resultSection" class="hidden">
    <div class="card">
      <h2>Summary</h2>
      <div class="summary-grid" id="summaryGrid"></div>
    </div>

    <div class="card">
      <h2>Team Roster</h2>
      <table id="teamTable">
        <thead><tr><th>Slot</th><th>Player Name</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="card">
      <h2>Extracted Events</h2>
      <table id="eventsTable">
        <thead><tr><th>ID</th><th>Time</th><th>Type</th><th>Actor</th><th>Targets</th><th>Text</th><th title="review=true means the event has quality flags and should be checked manually">Review ⓘ</th><th>Flags</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="card">
      <h2>Group Images (Step 6 — event crops packed 8-per-image)</h2>
      <div class="img-grid" id="groupImages"><div style="color:var(--muted)">No images yet</div></div>
    </div>

    <div class="card">
      <h2>Map Video (isolated minimap footage)</h2>
      <video id="mapVideo" controls style="width:100%;max-width:560px;border:1px solid var(--border);border-radius:8px;display:none"></video>
      <div id="mapVideoMsg" style="color:var(--muted);font-size:13px">No video yet</div>
    </div>

    <div class="card">
      <h2>Team Image (player roster frame from Step 3)</h2>
      <img id="teamImage" style="max-width:100%;border:1px solid var(--border);border-radius:6px;display:none" />
      <div id="teamImageMsg" style="color:var(--muted);font-size:13px">No image yet</div>
    </div>
  </section>
</main>

<script>
const fileInput = document.getElementById('file');
const fileName = document.getElementById('fileName');
const runBtn = document.getElementById('runBtn');
const statusCard = document.getElementById('statusCard');
const statusText = document.getElementById('statusText');
const errorSection = document.getElementById('errorSection');
const resultSection = document.getElementById('resultSection');

let currentFile = null;

fileInput.addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  currentFile = f || null;
  fileName.textContent = f ? `${f.name} · ${(f.size/1024/1024).toFixed(1)} MB` : 'No file selected';
  runBtn.disabled = !f;
});

runBtn.addEventListener('click', () => { if (currentFile) run(currentFile); });

function showError(msg) {
  errorSection.innerHTML = `<div class="card"><div class="error">${escapeHtml(msg)}</div></div>`;
  errorSection.classList.remove('hidden');
  resultSection.classList.add('hidden');
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function run(file) {
  errorSection.classList.add('hidden');
  resultSection.classList.add('hidden');
  statusCard.classList.remove('hidden');
  runBtn.disabled = true;
  try {
    statusText.textContent = 'Uploading video and running 7-step pipeline... (this may take 5-10 minutes)';
    const fd = new FormData();
    fd.append('file', file);
    const t0 = Date.now();
    const resp = await fetch('/api/v1/battle-log/extract', { method: 'POST', body: fd });
    const data = await resp.json();
    if (resp.status !== 200) throw new Error(data.detail || 'Extraction failed');
    statusText.textContent = `Completed in ${((Date.now()-t0)/1000).toFixed(1)}s`;
    render(data);
  } catch (e) {
    console.error(e);
    showError(String(e));
    statusCard.classList.add('hidden');
  } finally {
    runBtn.disabled = !currentFile;
  }
}

function render(data) {
  resultSection.classList.remove('hidden');
  
  const meta = data.metadata || {};
  document.getElementById('summaryGrid').innerHTML = `
    <div class="stat-box"><div class="k">Video ID</div><div class="v" style="font-size:16px">${escapeHtml(data.video_id)}</div></div>
    <div class="stat-box"><div class="k">Events</div><div class="v">${data.events.length}</div></div>
    <div class="stat-box"><div class="k">Team Players</div><div class="v">${Object.keys(data.team_slots).length}</div></div>
    <div class="stat-box"><div class="k">Frames</div><div class="v">${meta.frame_count||'?'}</div></div>
    <div class="stat-box"><div class="k">Group Images</div><div class="v">${meta.group_count||'?'}</div></div>
    <div class="stat-box"><div class="k">Needs Review</div><div class="v ${data.events.filter(e=>e.review).length>0?'warn':'good'}">${data.events.filter(e=>e.review).length}</div></div>
  `;

  const teamTbody = document.querySelector('#teamTable tbody');
  teamTbody.innerHTML = Object.entries(data.team_slots).map(([slot, name]) => `
    <tr><td>${slot}</td><td>${escapeHtml(name)}</td></tr>
  `).join('');

  const eventsTbody = document.querySelector('#eventsTable tbody');
  eventsTbody.innerHTML = data.events.map(e => `
    <tr>
      <td>${e.event_id}</td>
      <td>${escapeHtml(e.time||'?')}</td>
      <td><span class="pill info">${escapeHtml(e.type)}</span></td>
      <td>${escapeHtml(e.actor||'-')}</td>
      <td>${escapeHtml(Array.isArray(e.targets) ? e.targets.join(', ') : (e.targets||'-'))}</td>
      <td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(e.text||'-')}</td>
      <td>${e.review?'<span class="pill warn" title="Has quality flags — check manually">Yes</span>':'<span class="pill good">No</span>'}</td>
      <td style="font-size:11px;color:var(--muted)">${(e.flags||[]).join(', ')||'-'}</td>
    </tr>
  `).join('');

  // Load group images
  fetch(data.download_urls.group_images)
    .then(r => r.json())
    .then(d => {
      document.getElementById('groupImages').innerHTML = d.images.map(img => `
        <div class="img-frame">
          <img src="${img.url}" alt="${escapeHtml(img.name)}" />
          <div class="caption">${escapeHtml(img.name)}</div>
        </div>
      `).join('');
    });

  // Load map video
  const mapVid = document.getElementById('mapVideo');
  const mapMsg = document.getElementById('mapVideoMsg');
  mapVid.src = data.download_urls.map_video;
  mapVid.style.display = 'block';
  mapMsg.style.display = 'none';

  // Load team image
  const teamImg = document.getElementById('teamImage');
  const teamMsg = document.getElementById('teamImageMsg');
  teamImg.src = data.download_urls.team_image;
  teamImg.style.display = 'block';
  teamMsg.style.display = 'none';
}
</script>
</body>
</html>
"""
