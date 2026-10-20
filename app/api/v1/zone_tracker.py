"""
Zone & Map Tracker API Router

Provides complete production endpoints for:
- Processing match videos through the 8-step Zone Tracker Pipeline
- Inspecting status, zones metadata, and downloading rendered map deliverables
- Visualizing all 8 steps via an interactive visual inspection dashboard
- Querying available reference map templates and past session outputs
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.zone_tracker.pipeline import run_player_path_pipeline

log = logging.getLogger("app.api.v1.zone_tracker")
router = APIRouter(prefix="/zone", tags=["Zone & Map Tracker"])

# In-memory session cache and active task tracking
ZONE_SESSIONS: Dict[str, Any] = {}
ACTIVE_TASKS: Dict[str, Any] = {}

# Standard reference map lookup
MAP_CATALOG: Dict[str, str] = {
    "solara": "assets/maps/solara.jpg",
    "nexterra": "assets/maps/nexterra.png",
    "kalahari": "assets/maps/kalahari.jpg",
    "bermuda": "assets/maps/bermuda.jpeg",
    "purgatory": "assets/maps/purgatory.jpg",
}


def _resolve_base_map(map_name_or_path: str) -> str:
    """Resolve a map alias (e.g. 'solara') or direct file path to an absolute path."""
    clean_name = map_name_or_path.strip().lower()
    if clean_name in MAP_CATALOG:
        rel_path = MAP_CATALOG[clean_name]
    else:
        rel_path = map_name_or_path.strip()

    # Resolve against Backend root if relative
    p = Path(rel_path)
    if not p.is_absolute():
        backend_root = Path(__file__).resolve().parent.parent.parent.parent
        p = backend_root / rel_path

    if not p.exists():
        available = list(MAP_CATALOG.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Base map '{map_name_or_path}' not found at '{p}'. Available presets: {available}",
        )
    return str(p)


def _get_zone_dir(video_id: str) -> Path:
    backend_root = Path(__file__).resolve().parent.parent.parent.parent
    return backend_root / "video" / "output" / "zone" / str(video_id)


def _get_session_data(video_id: str) -> Dict[str, Any]:
    """Retrieve session dictionary from memory cache or session.json on disk."""
    if video_id in ZONE_SESSIONS:
        return ZONE_SESSIONS[video_id]

    zone_dir = _get_zone_dir(video_id)
    session_json = zone_dir / "session.json"
    if session_json.exists():
        try:
            with open(session_json, "r") as f:
                data = json.load(f)
                ZONE_SESSIONS[video_id] = data
                return data
        except Exception as e:
            log.warning(f"Could not parse session.json for {video_id}: {e}")

    # Fallback: check if zones.json and zones_map.png exist
    zones_json = zone_dir / "zones.json"
    zones_map = zone_dir / "zones_map.png"
    if zones_json.exists() and zones_map.exists():
        try:
            with open(zones_json, "r") as f:
                z_data = json.load(f)
            data = {
                "success": True,
                "video_id": str(video_id),
                "zones_count": len(z_data.get("zones", [])),
                "zones_json": str(zones_json),
                "zones_map_png": str(zones_map),
                "session_dir": str(zone_dir),
                "zones": z_data.get("zones", []),
            }
            ZONE_SESSIONS[video_id] = data
            return data
        except Exception:
            pass

    raise HTTPException(status_code=404, detail=f"No processed zone session found for video_id '{video_id}'")


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class ZoneProcessResponse(BaseModel):
    success: bool
    pipeline: str
    video_id: str
    status: str
    zones_count: int
    reference_map: str
    execution_time_seconds: Optional[float] = None
    zones: List[Dict[str, Any]] = []
    download_urls: Dict[str, str] = {}
    visual_dashboard_url: str


class MapInfo(BaseModel):
    name: str
    path: str
    exists: bool
    dimensions: Optional[str] = None
    size_bytes: Optional[int] = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.get("/maps", response_model=List[MapInfo], summary="List Available Reference Maps")
async def list_available_maps():
    """List all supported base map templates and their readiness."""
    backend_root = Path(__file__).resolve().parent.parent.parent.parent
    maps_info = []
    import cv2

    for name, rel_path in MAP_CATALOG.items():
        p = backend_root / rel_path
        exists = p.exists()
        dims = None
        size = None
        if exists:
            size = p.stat().st_size
            try:
                im = cv2.imread(str(p))
                if im is not None:
                    dims = f"{im.shape[1]}x{im.shape[0]}"
            except Exception:
                pass
        maps_info.append(MapInfo(
            name=name,
            path=str(rel_path),
            exists=exists,
            dimensions=dims,
            size_bytes=size,
        ))
    return maps_info


@router.get("/list", summary="List All Processed Zone Sessions")
async def list_zone_sessions():
    """List all existing zone tracker sessions stored on disk."""
    backend_root = Path(__file__).resolve().parent.parent.parent.parent
    zone_base = backend_root / "video" / "output" / "zone"
    sessions = []

    if zone_base.exists():
        for item in sorted(zone_base.iterdir()):
            if item.is_dir() and not item.name.startswith((".", "_")):
                vid = item.name
                zones_map = item / "zones_map.png"
                zones_json = item / "zones.json"
                session_json = item / "session.json"
                count = 0
                if zones_json.exists():
                    try:
                        with open(zones_json) as f:
                            d = json.load(f)
                            count = len(d.get("zones", []))
                    except Exception:
                        pass
                sessions.append({
                    "video_id": vid,
                    "has_map": zones_map.exists(),
                    "zones_count": count,
                    "zones_map_url": f"/api/v1/zone/{vid}/zones-map",
                    "visual_url": f"/api/v1/zone/{vid}/visual",
                    "status_url": f"/api/v1/zone/{vid}/status",
                })
    return {"total_sessions": len(sessions), "sessions": sessions}


@router.post("/process", response_model=ZoneProcessResponse, summary="Process Video Through 8-Step Zone Pipeline")
async def process_zone_pipeline(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None, description="Uploaded match video (MP4/MOV)"),
    video_path: Optional[str] = Form(None, description="Local path to video file (alternative to file upload)"),
    map_name: str = Form("solara", description="Map template name ('solara', 'nexterra', 'kalahari', 'bermuda', 'purgatory') or file path"),
    video_id: Optional[str] = Form(None, description="Custom video identifier (auto-generated if omitted)"),
    keep_intermediate: bool = Form(True, description="Preserve step1-step8 deliverables"),
):
    """
    Execute the complete 8-step Free Fire Zone Tracking Pipeline:
    1. Standardize canvas to 1920x1080 at full native FPS
    2. Isolate left 960x1080 minimap stream & extract team roster frame
    3. Detect UI anchors and crop clean map stream
    4. SIFT map alignment with reference template
    5. Camera frame registration (transforms.npz)
    6. Parallel multi-core zone observation extraction
    7. Chronological clustering into discrete safe zone phases
    8. Render clean white safe zones on reference base map
    """
    ref_map_path = _resolve_base_map(map_name)

    if not file and not video_path:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded video file or video_path.")

    assigned_id = str(video_id or uuid.uuid4().hex[:8])
    temp_dir = None
    input_file_path = None

    if file:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")
        temp_dir = tempfile.mkdtemp(prefix=f"zone_{assigned_id}_")
        input_file_path = os.path.join(temp_dir, f"input_{assigned_id}.mp4")
        with open(input_file_path, "wb") as f:
            f.write(raw)
    else:
        p = Path(video_path)
        if not p.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent.parent.parent
            p = backend_root / video_path
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Local video file not found at: {p}")
        input_file_path = str(p)

    t0 = time.time()
    try:
        log.info(f"[zone/process] Starting 8-step pipeline for video_id={assigned_id} on map={ref_map_path}")
        result = await asyncio.to_thread(
            run_player_path_pipeline,
            video_path=input_file_path,
            video_id=assigned_id,
            reference_map_path=ref_map_path,
            cleanup_intermediate=not keep_intermediate,
        )
        elapsed = round(time.time() - t0, 2)
        ZONE_SESSIONS[assigned_id] = result

        zones = result.get("zones", [])
        return ZoneProcessResponse(
            success=True,
            pipeline="zone_tracker_8_step",
            video_id=assigned_id,
            status="completed",
            zones_count=len(zones),
            reference_map=ref_map_path,
            execution_time_seconds=elapsed,
            zones=zones,
            download_urls={
                "zones_map": f"/api/v1/zone/{assigned_id}/zones-map",
                "zones_json": f"/api/v1/zone/{assigned_id}/zones",
                "session_json": f"/api/v1/zone/{assigned_id}/status",
            },
            visual_dashboard_url=f"/api/v1/zone/{assigned_id}/visual",
        )

    except Exception as e:
        log.exception(f"Zone pipeline failed for {assigned_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/{video_id}/status", summary="Get Zone Pipeline Status & Deliverables")
async def get_zone_status(video_id: str):
    """Retrieve metadata, detected zone phases, and deliverable download URLs."""
    data = _get_session_data(video_id)
    return {
        "status": "ready",
        "video_id": video_id,
        "zones_count": data.get("zones_count", len(data.get("zones", []))),
        "reference_map": data.get("reference_map"),
        "fps": data.get("fps"),
        "total_frames": data.get("total_frames"),
        "duration_seconds": data.get("duration_seconds"),
        "zones": data.get("zones", []),
        "download_urls": {
            "zones_map": f"/api/v1/zone/{video_id}/zones-map",
            "zones_json": f"/api/v1/zone/{video_id}/zones",
            "visual_dashboard": f"/api/v1/zone/{video_id}/visual",
        },
        "artifacts": data.get("artifacts", {}),
    }


@router.get("/{video_id}/zones", summary="Get Safe Zones JSON Metadata")
async def get_zone_json(video_id: str):
    """Return the detected safe zones JSON."""
    zone_dir = _get_zone_dir(video_id)
    z_file = zone_dir / "zones.json"
    if not z_file.exists():
        raise HTTPException(status_code=404, detail="zones.json not found for this session.")
    with open(z_file, "r") as f:
        return json.load(f)


@router.get("/{video_id}/zones-map", summary="Download or View Rendered Safe Zones Map")
async def get_zones_map_image(video_id: str):
    """Return the final safe zones map image (clean white rings on reference map)."""
    zone_dir = _get_zone_dir(video_id)
    map_file = zone_dir / "zones_map.png"
    if not map_file.exists():
        raise HTTPException(status_code=404, detail="zones_map.png not found for this session.")
    return FileResponse(str(map_file), media_type="image/png", filename=f"zones_map_{video_id}.png")


@router.get("/{video_id}/step/{step_num}/{artifact_name}", summary="Download Specific Step Deliverable")
async def get_step_artifact(video_id: str, step_num: int, artifact_name: str):
    """Download an intermediate artifact from step1 through step8."""
    zone_dir = _get_zone_dir(video_id)
    step_dir = zone_dir / f"step{step_num}"
    target_file = step_dir / artifact_name

    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_name}' in step{step_num} not found.")

    ext = target_file.suffix.lower()
    media_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp4": "video/mp4",
        ".json": "application/json",
        ".npz": "application/octet-stream",
    }
    media_type = media_map.get(ext, "application/octet-stream")
    return FileResponse(str(target_file), media_type=media_type, filename=artifact_name)


# -----------------------------------------------------------------------------
# Visual Inspection Dashboard
# -----------------------------------------------------------------------------
@router.get("/{video_id}/visual", response_class=HTMLResponse, summary="Interactive Visual Inspection Dashboard")
async def get_visual_inspection_dashboard(video_id: str):
    """
    Renders an interactive, dark-mode visual inspection dashboard allowing users
    to view and verify deliverables across all 8 pipeline steps.
    """
    try:
        session = _get_session_data(video_id)
    except HTTPException:
        session = {"video_id": video_id, "zones": [], "zones_count": 0}

    zones = session.get("zones", [])
    fps = session.get("fps", 60.0)
    total_frames = session.get("total_frames", 0)
    duration = session.get("duration_seconds", 0.0)
    ref_map = session.get("reference_map", "Unknown")
    map_name = Path(ref_map).stem.upper() if ref_map else "BASE MAP"

    # Format zone table rows
    table_rows = ""
    for z in zones:
        name = z.get("name", "Zone")
        cx = round(z.get("cx", 0), 1)
        cy = round(z.get("cy", 0), 1)
        r = round(z.get("r", 0), 1)
        t_start = round(z.get("first_seen_s", z.get("t_start", 0)), 1)
        t_end = round(z.get("last_seen_s", z.get("t_end", 0)), 1)
        conf = z.get("confidence", "high").upper()
        conf_badge = '<span class="badge badge-good">HIGH</span>' if conf == "HIGH" else '<span class="badge badge-warn">MEDIUM</span>'

        table_rows += f"""
        <tr>
          <td><strong>{name}</strong></td>
          <td>({cx}, {cy})</td>
          <td><code>{r} px</code></td>
          <td>{t_start}s &rarr; {t_end}s</td>
          <td>{conf_badge}</td>
        </tr>
        """

    if not table_rows:
        table_rows = '<tr><td colspan="5" style="text-align:center; color: var(--muted);">No zones found</td></tr>'

    html_content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LazarTrack · Zone Pipeline Visualizer ({video_id})</title>
<style>
  :root {{
    --bg: #090d16;
    --card: #111827;
    --card-hover: #1f293d;
    --border: #1e293b;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --good: #22c55e;
    --warn: #eab308;
    --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    line-height: 1.5;
  }}
  header {{
    background: #0f172a;
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }}
  .brand {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .brand h1 {{
    margin: 0;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.5px;
  }}
  .badge {{
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
  }}
  .badge-accent {{ background: rgba(56, 189, 248, 0.15); color: var(--accent); border: 1px solid rgba(56, 189, 248, 0.3); }}
  .badge-good {{ background: rgba(34, 197, 94, 0.15); color: var(--good); border: 1px solid rgba(34, 197, 94, 0.3); }}
  .badge-warn {{ background: rgba(234, 179, 8, 0.15); color: var(--warn); border: 1px solid rgba(234, 179, 8, 0.3); }}
  
  .stats-bar {{
    display: flex;
    gap: 24px;
    font-size: 14px;
  }}
  .stat-item {{
    display: flex;
    flex-direction: column;
  }}
  .stat-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
  .stat-val {{ font-weight: 700; color: #fff; }}

  main {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 32px 24px;
  }}
  
  .grid-2 {{
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 24px;
    margin-bottom: 32px;
  }}
  @media (max-width: 1024px) {{
    .grid-2 {{ grid-template-columns: 1fr; }}
  }}

  .card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  .card-header {{
    padding: 16px 20px;
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .card-header h2 {{
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    color: #f8fafc;
  }}
  .card-body {{
    padding: 20px;
  }}

  .map-preview {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: #000;
    display: block;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th, td {{
    padding: 12px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    color: var(--muted);
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
  }}

  .steps-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 20px;
    margin-top: 16px;
  }}
  .step-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .step-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .step-num {{
    font-weight: 700;
    font-size: 13px;
    color: var(--accent);
  }}
  .step-img {{
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: #020617;
  }}
  .btn {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    text-decoration: none;
    color: #fff;
    background: #2563eb;
    border: none;
    cursor: pointer;
  }}
  .btn:hover {{ background: #1d4ed8; }}
  .btn-outline {{
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
  }}
  .btn-outline:hover {{ background: rgba(255,255,255,0.05); }}
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1>LazarTrack Safe Zone Pipeline</h1>
    <span class="badge badge-accent">Session #{video_id}</span>
    <span class="badge badge-good">{map_name}</span>
  </div>
  <div class="stats-bar">
    <div class="stat-item">
      <span class="stat-label">Zones Detected</span>
      <span class="stat-val">{len(zones)} Safe Zones</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Framerate</span>
      <span class="stat-val">{fps:.2f} FPS</span>
    </div>
    <div class="stat-item">
      <span class="stat-label">Frames</span>
      <span class="stat-val">{total_frames} frames ({duration:.1f}s)</span>
    </div>
  </div>
</header>

<main>
  <!-- Primary Showcase: Rendered Map & Zones Table -->
  <div class="grid-2">
    <div class="card">
      <div class="card-header">
        <h2>Production Safe Zones (Rendered Base Map)</h2>
        <a href="/api/v1/zone/{video_id}/zones-map" target="_blank" class="btn">Full Resolution</a>
      </div>
      <div class="card-body">
        <img src="/api/v1/zone/{video_id}/zones-map" alt="Zones Map" class="map-preview" />
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h2>Detected Safe Zone Hierarchy</h2>
        <a href="/api/v1/zone/{video_id}/zones" target="_blank" class="btn btn-outline">Raw JSON</a>
      </div>
      <div class="card-body" style="padding: 0;">
        <table>
          <thead>
            <tr>
              <th>Zone</th>
              <th>Center (X, Y)</th>
              <th>Radius</th>
              <th>Active Timeline</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Detailed 8-Step Pipeline Visual Verification Grid -->
  <h2 style="font-size: 18px; margin-bottom: 16px;">8-Step Pipeline Visual Deliverables</h2>
  <div class="steps-grid">

    <!-- Step 1 -->
    <div class="step-card">
      <div class="step-head">
        <span class="step-num">Step 1: Canvas Normalizer</span>
        <span class="badge badge-good">1920x1080</span>
      </div>
      <p style="margin: 0; font-size: 13px; color: var(--muted);">Normalized match recording to native canvas at 60 FPS.</p>
      <a href="/api/v1/zone/{video_id}/step/1/normalized_video.mp4" class="btn btn-outline" target="_blank">Download MP4</a>
    </div>

    <!-- Step 2 -->
    <div class="step-card">
      <div class="step-head">
        <span class="step-num">Step 2: Stream Separator</span>
        <span class="badge badge-accent">Team Roster</span>
      </div>
      <img src="/api/v1/zone/{video_id}/step/2/team_image.png" onerror="this.src='/api/v1/zone/{video_id}/step/3/clean_map_frame_0.png'" class="step-img" alt="Team Roster" />
      <p style="margin: 0; font-size: 13px; color: var(--muted);">Isolated left 960px minimap & right team roster slots.</p>
      <a href="/api/v1/zone/{video_id}/step/2/map_video.mp4" class="btn btn-outline" target="_blank">Minimap Stream MP4</a>
    </div>

    <!-- Step 3 -->
    <div class="step-card">
      <div class="step-head">
        <span class="step-num">Step 3: Anchor Detection</span>
        <span class="badge badge-accent">Clean Crop</span>
      </div>
      <img src="/api/v1/zone/{video_id}/step/3/clean_map_frame_0.png" class="step-img" alt="Clean Map Frame 0" />
      <p style="margin: 0; font-size: 13px; color: var(--muted);">UI bounding boxes detected; cropped exact clean terrain stream.</p>
      <a href="/api/v1/zone/{video_id}/step/3/clean_map_video.mp4" class="btn btn-outline" target="_blank">Clean Map MP4</a>
    </div>

    <!-- Step 4 -->
    <div class="step-card">
      <div class="step-head">
        <span class="step-num">Step 4: SIFT Map Alignment</span>
        <span class="badge badge-good">Reproj &lt; 1.5px</span>
      </div>
      <img src="/api/v1/zone/{video_id}/step/4/alignment_overlay.png" class="step-img" alt="Alignment Overlay" />
      <p style="margin: 0; font-size: 13px; color: var(--muted);">Homography mapping frame 0 directly to {map_name} reference map.</p>
      <a href="/api/v1/zone/{video_id}/step/4/alignment_overlay.png" class="btn btn-outline" target="_blank">View 50% Overlay</a>
    </div>

    <!-- Step 5 -->
    <div class="step-card">
      <div class="step-head">
        <span class="step-num">Step 5: Camera Registration</span>
        <span class="badge badge-accent">Transforms</span>
      </div>
      <img src="/api/v1/zone/{video_id}/step/5/ui_mask.png" class="step-img" alt="UI Mask" />
      <p style="margin: 0; font-size: 13px; color: var(--muted);">Camera panning tracking and fixed UI mask creation.</p>
      <a href="/api/v1/zone/{video_id}/step/5/registration.json" class="btn btn-outline" target="_blank">Drift Metrics JSON</a>
    </div>

    <!-- Step 6 -->
    <div class="step-card">
      <div class="step-head">
        <span class="step-num">Step 6: Parallel Zone Detector</span>
        <span class="badge badge-good">8 Cores</span>
      </div>
      <p style="margin: 0; font-size: 13px; color: var(--muted);">Multi-core frame worker pool extracted candidate rings and lit circles.</p>
      <a href="/api/v1/zone/{video_id}/step/6/zone_observations.json" class="btn btn-outline" target="_blank">Raw Observations JSON</a>
    </div>

    <!-- Step 7 -->
    <div class="step-card">
      <div class="step-head">
        <span class="step-num">Step 7: Phase Clustering</span>
        <span class="badge badge-accent">Nested DBSCAN</span>
      </div>
      <p style="margin: 0; font-size: 13px; color: var(--muted);">RANSAC + spatial consensus filtered candidates into {len(zones)} nested zones.</p>
      <a href="/api/v1/zone/{video_id}/zones" class="btn btn-outline" target="_blank">Zones JSON</a>
    </div>

    <!-- Step 8 -->
    <div class="step-card">
      <div class="step-head">
        <span class="step-num">Step 8: Map Renderer</span>
        <span class="badge badge-good">Clean White</span>
      </div>
      <img src="/api/v1/zone/{video_id}/zones-map" class="step-img" alt="Final Map" />
      <p style="margin: 0; font-size: 13px; color: var(--muted);">Clean white circles with dark borders and anti-overlap badges.</p>
      <a href="/api/v1/zone/{video_id}/zones-map" class="btn" target="_blank">Download Map</a>
    </div>

  </div>
</main>

</body>
</html>
    """
    return HTMLResponse(content=html_content)
