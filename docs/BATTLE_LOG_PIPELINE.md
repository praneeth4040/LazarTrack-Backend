# ⚔️ Battle Log Extraction Pipeline (BLE-001)

Comprehensive technical reference for the video-based Free Fire Battle Log Extraction Engine in **LazarTrack Backend**.

---

## 1. Executive Summary & Purpose

The Battle Log Extraction Pipeline ingests recorded Free Fire tournament / competitive match videos (MP4, MOV, AVI) and extracts two critical match assets:
1. **Isolated Minimap Video Stream**: A clean 760×760 crop of the safe-zone and player movement area (`map_video.mp4`).
2. **Structured Battle Event Stream**: High-resolution event crops and parsed elimination/knockdown/revival logs (`events.json`).

### Key Innovation: Decoupled Architecture (Steps 1–6 vs Step 7)
- **User Upload Flow (Steps 1–6)**: Executes purely on local OpenCV / FFmpeg / Python (~30 seconds). Generates the isolated map video, team roster, and compact composite event images (`group_*.png`). Delivers instant visual feedback to the player.
- **On-Demand Vision Flow (Step 7)**: Runs the Vision-Language Model (`qwen/qwen3.7-flash` via OpenRouter) **only when explicitly requested** (e.g. from an admin portal or when training/building the downstream strategic suggestion engine). This eliminates unnecessary LLM API costs and keeps user upload latency ultra-low.

---

## 2. Architecture & Data Flow

```
                      Raw Match Video (MP4 / MOV / AVI)
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Video Normalizer (FFmpeg)                                      │
│ • Scales canvas to 1920×1080 @ 5 FPS                                   │
│ • Standardizes color space and extracts timestamped frame sequence     │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Minimap & Battle Log Separator (OpenCV + FFmpeg)               │
│ • Template/color scan for "BATTLE LOG" header anchor                   │
│ • Splits screen into:                                                  │
│   ├─ Battle Log ROI [960×499] -> frame sequence                        │
│   └─ Minimap Stream [760×760] -> stitched into map_video.mp4          │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Team Roster & Player Identifier (OpenCV / Qwen VL)             │
│ • Locates the team lineup frame at match start                         │
│ • Identifies 4 player slots (IGNs) and exports players.json + team.png │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Battle Log Row Divider (OpenCV)                                │
│ • Slices the 499px high battle log into 8 horizontal event slot rows   │
│ • Normalizes background transparency and prepares row buffers          │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Event Tracker & Deduplication Engine (NumPy / SSIM / Flow)     │
│ • Tracks notification entry, dwell time, and scroll-out across rows    │
│ • Deduplicates persistent notifications across consecutive frames      │
│ • Emits exactly ONE high-clarity event crop per in-game event          │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Group Image Packer (Pillow / OpenCV)                           │
│ • Packs 8 deduplicated event crops into unified composite group images │
│ • Adds event ID badges and visual metadata overlays                    │
│ • Saves deliverables: group_000.png, group_001.png, manifest.json      │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                     ════════════════╪════════════════
                     USER FLOW COMPLETES HERE (~30s)
                     Deliverables: map_video.mp4, group_*.png, players.json
                     ════════════════╪════════════════
                                     │
                      [On-Demand Trigger / Admin Call]
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Vision Language Event Extractor (Qwen 3.7 Flash via OpenRouter)│
│ • Reads existing group_*.png and players.json from session directory   │
│ • Zero need for raw video or temporary frames                         │
│ • Extracts structured JSON events: timestamp, actor, victim, weapon    │
│ • Writes events.json into session directory                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Step-by-Step Technical Details

### Step 1: Normalizer (`Backend/scripts/step1_normalizer.py`)
- **Engine**: Hardware-accelerated FFmpeg.
- **Function**: Standardizes arbitrary device aspect ratios (e.g. iPad 4:3, ultra-wide 20:9) into a standardized **`1920×1080`** canvas sampled at a constant **`5 FPS`**.
- **Outputs**: Normalized frame directory (`normalized_frames/frame_%04d.png`).

### Step 2: Separator (`Backend/scripts/step2_separator.py`)
- **Engine**: OpenCV + FFmpeg.
- **Function**:
  1. Detects the `"BATTLE LOG"` HUD banner using luminance and position thresholding.
  2. Crops the **Battle Log Region** (`[X: 0..960, Y: 280..779]`, dimensions `960×499`).
  3. Crops the **Minimap Stream** (`[X: 1160..1920, Y: 0..760]`, dimensions `760×760`) and encodes it directly into `map_video.mp4` using H.264 (`-c:v libx264 -pix_fmt yuv420p`).
- **Outputs**: `battle_log_frames/` and `map_video.mp4`.

### Step 3: Player Identifier (`Backend/scripts/step3_player_identifier.py`)
- **Engine**: OpenCV frame detection + fallback/VL name extraction.
- **Function**:
  - Scans early frames for the team intro overlay.
  - Crops the 4-man roster bar, saving `team_image.png`.
  - Generates `players.json` mapping slots `slot_1` through `slot_4` to player IGNs.
- **Outputs**: `players.json` and `team_image.png`.

### Step 4: Row Divider (`Backend/scripts/step4_row_divider.py`)
- **Engine**: Pure NumPy array slicing.
- **Function**: The 499px high Battle Log contains up to 8 active event notification rows. Step 4 computes row bounds and slices each battle log frame into discrete horizontal slot strips (`slot_0` to `slot_7`), filtering out blank background noise.
- **Outputs**: `row_crops/frame_XXXX_row_Y.png`.

### Step 5: Event Tracker (`Backend/scripts/step5_event_tracker.py`)
- **Engine**: Frame differencing, SSIM (Structural Similarity), and lifecycle state tracking.
- **Problem Solved**: A single kill feed notification remains visible on screen for 2.5 to 4.0 seconds (12 to 20 frames at 5 FPS) and slowly scrolls downward as new notifications appear.
- **Function**:
  - Tracks the lifecycle of each unique notification from initial appearance to scroll-down to fade-out.
  - Computes the highest-clarity frame for each event.
  - Deduplicates consecutive duplicate frames, reducing hundreds of raw frame rows down to ~15 to 40 unique events per match.
- **Outputs**: `deduped_events/event_XXXX.png` and tracking metadata.

### Step 6: Group Packer (`Backend/scripts/step6_group_packer.py`)
- **Engine**: PIL (Pillow) + OpenCV.
- **Function**:
  - Packs up to 8 deduplicated event crops into vertical composite cards (`group_000.png`, `group_001.png`, etc.).
  - Overlays high-contrast event sequence numbers, frame indices, and metadata badges.
  - Produces `all_videos_manifest.json` indexing all groups and events.
- **Outputs**:
  - `group_images/group_000.png`, `group_001.png`, ...
  - `manifest.json`
- **Session Cleanup**: Automatically purges temporary intermediate frame folders (`normalized_frames/`, `battle_log_frames/`, `row_crops/`), leaving only lightweight persistent assets.

### Step 7: Vision Extractor (`Backend/scripts/step7_vision_extractor.py`)
- **Execution Mode**: **Decoupled / On-Demand**.
- **Model**: `qwen/qwen3.7-flash` via OpenRouter.
- **Input**: Only reads `group_images/group_*.png` and `players.json` from the session directory.
- **Function**:
  - Encodes composite group images to base64.
  - Formulates a system prompt injecting the team roster from `players.json`.
  - Prompts the vision model to transcribe each badge-labeled event crop:
    - Event Type (`KILL`, `KNOCKDOWN`, `REVIVE`, `ZONE_ELIMINATION`)
    - Timestamp / Game Time
    - Actor (Instigator / Killer)
    - Victim / Target
    - Weapon / Vehicle / Zone
  - Outputs structured `events.json`.

---

## 4. Why Decouple Step 7?

| Feature | Monolithic Pipeline (Steps 1–7 All-in-One) | Decoupled Pipeline (Steps 1–6 Now, Step 7 On-Demand) |
|---|---|---|
| **User Upload Turnaround** | 4 to 8 minutes (waiting for LLM calls) | **~30 to 45 seconds** |
| **API Token Cost** | Incurred on *every single video upload* | **Zero API cost on upload**; only billed when admin triggers analysis |
| **Downstream Suggestion Engine** | Bound to upload time; cannot re-parse with improved prompts | **Can run retroactively** across all historical sessions with new models |
| **Server Resource Overhead** | High memory/network hold during LLM polling | **Fast execution, clean termination** |

---

## 5. Session Directory Structure

When a video is processed, its session is stored in `Backend/video/output/<video_id>/`:

```
Backend/video/output/
└── a1b2c3d4/                     # Unique 8-character video session ID
    ├── session.json              # Session metadata & asset filepaths
    ├── map_video.mp4             # Isolated 760x760 minimap video (H.264)
    ├── team_image.png            # Step 3 team roster frame
    ├── players.json              # Extracted player IGNs for slot 1..4
    ├── manifest.json             # Manifest of event groups and crop coordinates
    ├── group_images/             # Step 6 Deliverables
    │   ├── group_000.png         # Composite event card (events 1–8)
    │   ├── group_001.png         # Composite event card (events 9–16)
    │   └── ...
    └── events.json               # (Created only after Step 7 is invoked)
```

---

## 6. REST API Reference

### 1. Upload Video & Run Pipeline (Steps 1–6)
- **Endpoint**: `POST /api/v1/battle-log/extract`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: MP4, MOV, or AVI video file (Max 500 MB).
  - `run_step7`: `false` *(default)*. Set `true` only if immediate LLM extraction is required.
  - `cleanup_intermediate`: `true` *(default)*. Removes gigabytes of raw frames after packing.
  - `auto_cleanup_hours`: `24` *(default)*. Session retention duration.

#### Response:
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

---

### 2. On-Demand Step 7 Event Extraction (Admin / Suggestion Engine)
- **Endpoint**: `POST /api/v1/battle-log/{video_id}/extract-events`
- **Description**: Reads existing `group_*.png` and `players.json` from `Backend/video/output/{video_id}/`, executes Qwen 3.7 Flash extraction, and saves `events.json`.

#### Response:
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

---

### 3. Fetch Deliverables
- **Group Images List**: `GET /api/v1/battle-log/{video_id}/group-images`
- **Stream Single Group Image**: `GET /api/v1/battle-log/{video_id}/group-image/{filename}`
- **Stream Minimap Video**: `GET /api/v1/battle-log/{video_id}/map-video`
- **Download Team Image**: `GET /api/v1/battle-log/{video_id}/team-image`
- **Download Events JSON**: `GET /api/v1/battle-log/{video_id}/events.json`
- **Delete Session**: `DELETE /api/v1/battle-log/{video_id}`

---

## 7. Suggestion Engine Integration Blueprint

The decoupled architecture prepares the data foundation for the **AI Suggestion Engine**:

```
┌────────────────────────────────────────────────────────┐
│ SESSION DELIVERABLES:                                  │
│ 1. Minimap Video (map_video.mp4)                       │
│ 2. Events Log (events.json via Step 7)                 │
│ 3. Safe Zones (zones.json via Zone Tracker)            │
└────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ STRATEGIC AI SUGGESTION ENGINE                         │
│ • Correlates player movement with safe-zone shifts     │
│ • Flags high-risk rotation corridors                   │
│ • Analyzes combat engagements vs. circle timing        │
│ • Outputs tactical coaching feedback to esports teams  │
└────────────────────────────────────────────────────────┘
```
