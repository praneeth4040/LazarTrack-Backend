# 📡 LazarTrack Backend API Reference

Complete reference for all REST endpoints provided by the **LazarTrack Backend** service.

- **Base URL**: `http://localhost:8000` (or configured production host)
- **API Prefix**: `/api/v1`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

---

## 1. System & Health Endpoints

### 1.1 API Directory
- **Method**: `GET`
- **Path**: `/`
- **Description**: Returns root status and catalog of registered endpoint families.

### 1.2 Dynamic Route Listing
- **Method**: `GET`
- **Path**: `/endpoints`
- **Description**: Returns a complete dynamic list of all registered HTTP paths and methods.

### 1.3 Service Health Check
- **Method**: `GET`
- **Path**: `/api/v1/health`
- **Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "ocr_engine": "qwen_vl_openrouter",
  "vision_model": "qwen/qwen3.7-flash",
  "zone_tracker": "ready",
  "battle_log_extractor": "ready"
}
```

---

## 2. Player Stats Extraction Endpoints

### 2.1 Extract Match Stats (Production)
- **Method**: `POST`
- **Path**: `/api/v1/pipeline/match_stats`
- **Content-Type**: `multipart/form-data`
- **Description**: Ingests a Free Fire post-match scoreboard screenshot and executes the 5-step extraction pipeline.
- **Form Data**:
  - `file`: Scoreboard image (PNG, JPEG, WEBP; max 20 MB).

#### Example Request:
```bash
curl -X POST http://localhost:8000/api/v1/pipeline/match_stats \
  -F "file=@assets/sample_scoreboard.png"
```

#### Example Response:
```json
{
  "pipeline_id": "PSE-001",
  "pipeline_name": "Player Stats Extraction 1.0",
  "confidence": 1.0,
  "data": {
    "map": "Nexterra",
    "rank": 5,
    "total_teams": 11,
    "players": [
      {
        "inGameName": "BOSS WYZEX",
        "kills": 3,
        "deaths": 3,
        "assists": 1,
        "damage": 1711,
        "actualDamage": 1095,
        "knockedDown": 4,
        "heal": 1559,
        "helpUp": 1,
        "revival": 3,
        "headShotRate": 66.67
      }
    ],
    "metadata": {
      "elapsed_seconds": 16.39,
      "model": "qwen/qwen3.7-flash"
    }
  },
  "raw_texts": []
}
```

### 2.2 Visual Debugger Inspection
- **Method**: `POST`
- **Path**: `/api/v1/pipeline/match_stats/debug`
- **Description**: Returns the same structured data as `/match_stats`, plus an array of base64 debug images and durations for all 5 steps.

### 2.3 Interactive Visualizer UI
- **Method**: `GET`
- **Path**: `/api/v1/visualize`
- **Description**: Web-based visual dashboard for uploading screenshots and inspecting each step interactively in the browser.

---

## 3. Battle Log Video Extraction Endpoints

### 3.1 Extract Battle Log Deliverables (Steps 1–6)
- **Method**: `POST`
- **Path**: `/api/v1/battle-log/extract`
- **Content-Type**: `multipart/form-data`
- **Form Data**:
  - `file`: Tournament match video (MP4, MOV, AVI; max 500 MB).
  - `run_step7`: `false` *(default)*. When `false`, completes at Step 6, delivering composite event images, map video, and team roster in ~30s.
  - `cleanup_intermediate`: `true` *(default)*. Removes gigabytes of raw frames after packing.
  - `auto_cleanup_hours`: `24` *(default)*. Session retention duration.

#### Example Request:
```bash
curl -X POST http://localhost:8000/api/v1/battle-log/extract \
  -F "file=@match_video.mp4"
```

#### Example Response:
```json
{
  "success": true,
  "video_id": "a1b2c3d4",
  "events": [],
  "team_slots": {
    "slot_1": "ALPHA",
    "slot_2": "BRAVO",
    "slot_3": "CHARLIE",
    "slot_4": "DELTA"
  },
  "metadata": {
    "frame_count": 850,
    "group_count": 3,
    "elapsed_seconds": 29.8
  },
  "download_urls": {
    "group_images": "/api/v1/battle-log/a1b2c3d4/group-images",
    "map_video": "/api/v1/battle-log/a1b2c3d4/map-video",
    "team_image": "/api/v1/battle-log/a1b2c3d4/team-image",
    "events_json": "/api/v1/battle-log/a1b2c3d4/events.json"
  }
}
```

### 3.2 Trigger On-Demand Step 7 Vision Extraction (Admin)
- **Method**: `POST`
- **Path**: `/api/v1/battle-log/{video_id}/extract-events`
- **Description**: Triggers Vision-Language processing on the session's existing `group_*.png` images to generate `events.json`.

#### Example Request:
```bash
curl -X POST http://localhost:8000/api/v1/battle-log/a1b2c3d4/extract-events
```

#### Example Response:
```json
{
  "success": true,
  "video_id": "a1b2c3d4",
  "events_count": 18,
  "events_file": "/Users/.../Backend/video/output/a1b2c3d4/events.json",
  "events": [
    {
      "event_id": 1,
      "time": "01:24",
      "type": "KNOCKDOWN",
      "actor": "ALPHA",
      "targets": ["ENEMY_SNIPER"],
      "weapon": "AWM",
      "review": false,
      "flags": []
    }
  ]
}
```

### 3.3 List Group Images
- **Method**: `GET`
- **Path**: `/api/v1/battle-log/{video_id}/group-images`
- **Description**: Returns URLs to all packed Step 6 composite event cards.

### 3.4 Stream Minimap Video
- **Method**: `GET`
- **Path**: `/api/v1/battle-log/{video_id}/map-video`
- **Description**: Streams the 760×760 isolated minimap video (`map_video.mp4`).

### 3.5 Download Team Roster Frame
- **Method**: `GET`
- **Path**: `/api/v1/battle-log/{video_id}/team-image`
- **Description**: Downloads the team roster frame captured during Step 3.

### 3.6 Download Events JSON
- **Method**: `GET`
- **Path**: `/api/v1/battle-log/{video_id}/events.json`
- **Description**: Downloads structured elimination and combat event logs generated by Step 7.

### 3.7 Delete Session
- **Method**: `DELETE`
- **Path**: `/api/v1/battle-log/{video_id}`
- **Description**: Deletes all on-disk artifacts for a session.

---

## 4. Zone Tracker Endpoints

### 4.1 Process Match Video
- **Method**: `POST`
- **Path**: `/api/v1/zone/process`
- **Content-Type**: `multipart/form-data`
- **Form Data**:
  - `file`: Video file (MP4).
  - `video_path`: Optional existing local video path if not uploading.
  - `map_name`: Map name (`bermuda`, `purgatory`, `kalahari`, `solara`, `nexterra`).

### 4.2 List Sessions
- **Method**: `GET`
- **Path**: `/api/v1/zone/list`
- **Description**: Lists all processed zone tracking sessions.

### 4.3 Check Status
- **Method**: `GET`
- **Path**: `/api/v1/zone/{video_id}/status`
- **Description**: Returns processing status and stage completion.

### 4.4 Download Rendered Safe Zones Map
- **Method**: `GET`
- **Path**: `/api/v1/zone/{video_id}/zones-map`
- **Description**: Returns `rendered_map.png` with all safe zones rendered on the base map.

### 4.5 Download Raw Zones Metadata
- **Method**: `GET`
- **Path**: `/api/v1/zone/{video_id}/zones`
- **Description**: Returns `zones.json` containing circle center coordinates, radii, and confidence values.

### 4.6 Interactive Visual Verification Dashboard
- **Method**: `GET`
- **Path**: `/api/v1/zone/{video_id}/visual`
- **Description**: Browser UI displaying intermediate artifacts from each of the 8 steps.

---

## 5. Async Jobs API (fire-and-forget pipelines)

Long pipelines run as background jobs. Submit → `202` + job record → poll until terminal.
All endpoints require `Authorization: Bearer <supabase_access_token>`; users only see their own jobs.
Job records persist to disk and survive restarts; jobs orphaned by a restart are marked `failed`.

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/jobs/match-stats` | POST | Submit screenshot stats job. Body: `{match_id, screenshot_url}` → `202 {job}` |
| `/api/v1/jobs/battle-log` | POST | Submit battle-log video job. Body: `{match_id, video_url, run_step7?}` → `202 {job}` |
| `/api/v1/jobs` | GET | List my jobs. Query: `match_id`, `kind`, `status`, `include_result` |
| `/api/v1/jobs/{job_id}` | GET | Full job record incl. `progress` and, when completed, `result` |
| `/api/v1/jobs/{job_id}` | DELETE | Cancel a queued job, or delete a finished one (`409` if running) |

**Job record:** `{id, kind: match_stats|battle_log, status: queued|running|completed|failed|cancelled, match_id, progress: {stage, message, percent}, result, error, created_at, started_at, finished_at}`

- `match_stats` result = the same payload as `POST /pipeline/match_stats`.
- `battle_log` result = the same payload as `POST /battle-log/extract` (events, team_slots, download_urls…).
- Concurrency caps: one battle-log job at a time, two stats jobs; a queued job starts as soon as a slot frees.
- The synchronous `POST /pipeline/match_stats` and `POST /battle-log/extract` endpoints still exist and share the same runners, but the jobs API is preferred for anything long-running.
