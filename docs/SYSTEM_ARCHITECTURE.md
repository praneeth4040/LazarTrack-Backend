# 🏗️ LazarTrack System Architecture & Design

Comprehensive technical architecture guide for the complete **LazarTrack** Free Fire esports analytics platform.

---

## 1. System Overview

**LazarTrack** is a professional analytics and coaching suite for Free Fire competitive esports. It ingests match screenshots and video recordings, transforming unstructured visual gameplay into verified, structured competitive data.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND CLIENT LAYER                           │
│     Expo 51 / React Native (iOS, Android, Web at http://localhost:8081)│
│                                                                        │
│   • Match Stats Upload & Review    • Safe Zone Map Inspector           │
│   • Battle Log Viewer              • Team Performance Dashboard        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST (Multipart & JSON)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        BACKEND API & CORE ENGINE                       │
│             FastAPI / Python 3.10+ (at http://localhost:8000)          │
│                                                                        │
│  ┌───────────────────────┐  ┌───────────────────┐  ┌────────────────┐  │
│  │     PLAYER STATS      │  │   ZONE TRACKER    │  │   BATTLE LOG   │  │
│  │  5-Step CV + VL Model │  │ 8-Step CV + SIFT  │  │ 7-Step Video   │  │
│  │   (Scoreboards)       │  │ (Minimap Shrink)  │  │  (Kill Feed)   │  │
│  └───────────┬───────────┘  └─────────┬─────────┘  └───────┬────────┘  │
│              │                        │                    │           │
│              ▼                        ▼                    ▼           │
│       match_stats.json          rendered_map.png      group_*.png      │
│                                    zones.json         map_video.mp4    │
│                                                        events.json     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  EXTERNAL AI / VISION-LANGUAGE MODELS                  │
│       OpenRouter API (`qwen/qwen3.7-flash` with Strict JSON Schema)    │
│                                                                        │
│   • Zero local GPU requirements       • Ultra-low memory footprint     │
│   • 100% precision on player names    • Sub-3s container cold start    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The Three Core Intelligence Pipelines

### 1. Player Stats Extraction Pipeline (`PSE-001`)
- **Input**: Post-match scoreboard screenshot (PNG, JPEG, WEBP).
- **Architecture**: 5-step hybrid computer vision and vision-language extraction.
  - **Steps 1–4 (Pure OpenCV, ~0.06s)**: Standardizes canvas to 1920×1080, detects table bounding box using high-luminance white anchors, trims the bottom golden `"STATS"` button, and slices the table into 8 isolated column images.
  - **Step 5 (Vision Language, ~16s)**: Dispatches header + 8 column slices to `qwen/qwen3.7-flash` via OpenRouter with strict JSON schema.
- **Output**: Structured match stats (`map`, `rank`, `total_teams`, `players[]` with all 10 combat metrics).
- **Docs**: [PLAYER_STATS_PIPELINE.md](file:///Users/praneeth/Lazarflow/LazarTrack/Backend/docs/PLAYER_STATS_PIPELINE.md)

---

### 2. Zone Tracker Pipeline (`ZTE-001`)
- **Input**: Match recording video (MP4) + Reference map name (`bermuda`, `purgatory`, etc.).
- **Architecture**: 8-step computer vision and geometric alignment pipeline.
  - **Steps 1–3**: Video normalization (1920×1080), minimap half-screen isolation (960×1080), and UI border cleaning.
  - **Step 4**: SIFT keypoint feature matching & RANSAC homography alignment against high-res reference map.
  - **Step 5**: Keyframe registration and camera motion tracking (`transforms.npz`).
  - **Steps 6–8**: 8-core parallel Hough + lit-boundary circle detection, DBSCAN consensus clustering, and high-contrast dual-halo map rendering.
- **Output**: `rendered_map.png` with concentric safe zone phases and `zones.json`.
- **Docs**: [ZONE_TRACKER_PIPELINE.md](file:///Users/praneeth/Lazarflow/LazarTrack/Backend/docs/ZONE_TRACKER_PIPELINE.md)

---

### 3. Battle Log Extraction Pipeline (`BLE-001`)
- **Input**: Match recording video (MP4, MOV, AVI).
- **Architecture**: 7-step decoupled video extraction engine.
  - **Steps 1–6 (Automatic on Video Upload, ~30s)**:
    - Normalizes video to 1920×1080 @ 5 FPS.
    - Isolates 760×760 minimap stream into `map_video.mp4`.
    - Crops team roster frame and extracts player IGNs into `players.json`.
    - Slices 8 battle log rows and tracks notification lifecycle across frames.
    - Packs deduplicated event crops into composite `group_*.png` images.
  - **Step 7 (On-Demand Admin / Suggestion Engine Trigger)**:
    - Reads `group_*.png` and `players.json` without needing raw video or temp frames.
    - Dispatches to `qwen/qwen3.7-flash` for structured event transcription (`events.json`).
- **Output**: `group_*.png` images, `map_video.mp4`, `team_image.png`, `players.json`, and on-demand `events.json`.
- **Docs**: [BATTLE_LOG_PIPELINE.md](file:///Users/praneeth/Lazarflow/LazarTrack/Backend/docs/BATTLE_LOG_PIPELINE.md)

---

## 3. Directory Layout

```
LazarTrack/
├── LazarTrack/                     # Frontend Application
│   ├── src/                        # React Native / Expo application source
│   │   ├── screens/                # UI Screens (Dashboard, AddResult, etc.)
│   │   ├── services/               # API clients (extraction.ts, etc.)
│   │   └── components/             # Reusable UI widgets
│   ├── .env                        # EXPO_PUBLIC_EXTRACTION_API_URL
│   └── package.json
│
├── Backend/                        # Backend API & Computer Vision Engine
│   ├── app/
│   │   ├── api/v1/                 # FastAPI REST Endpoints
│   │   │   ├── extract.py          # Match stats extraction endpoint
│   │   │   ├── battle_log.py       # Battle log upload & on-demand Step 7
│   │   │   ├── zone_tracker.py     # Zone tracker process & visual dashboard
│   │   │   ├── visualize.py        # Interactive 5-step debug visualizer
│   │   │   └── health.py           # Health check & engine info
│   │   ├── core/                   # Configuration & logging
│   │   ├── pipelines/              # Orchestrators (five_step.py, video_pipeline.py)
│   │   ├── zone_tracker/           # 8-Step Zone Tracking modules
│   │   ├── ocr/                    # Smart table detector, preprocessors
│   │   └── schemas/                # Pydantic models & validation schemas
│   │
│   ├── assets/maps/                # Reference blank map images
│   ├── docs/                       # Technical Documentation (Markdown)
│   │   ├── SYSTEM_ARCHITECTURE.md  # (This file)
│   │   ├── PLAYER_STATS_PIPELINE.md# Deep-dive on Scoreboard OCR & Qwen VL
│   │   ├── BATTLE_LOG_PIPELINE.md  # Deep-dive on Battle Log Video Pipeline
│   │   ├── ZONE_TRACKER_PIPELINE.md# Deep-dive on 8-Step Zone Tracking
│   │   └── API_REFERENCE.md        # Complete REST API reference
│   │
│   ├── scripts/                    # Step scripts & batch runners
│   ├── video/                      # Video processing workspace
│   │   ├── input/                  # Local test video files (gitignored)
│   │   └── output/                 # Session output directories (gitignored)
│   ├── .env                        # OPENROUTER_API_KEY, host/port settings
│   ├── Dockerfile                  # Containerized deployment spec
│   └── requirements.txt            # Python dependencies
```

---

## 4. Environment Configuration

### Backend (`Backend/.env`):
```bash
# Core API Settings
PROJECT_NAME="LazarTrack Backend"
API_V1_STR="/api/v1"

# OpenRouter API Key for Vision Language Inference
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx

# Video & Asset Directories
VIDEO_INPUT_DIR=video/input
VIDEO_OUTPUT_DIR=video/output
```

### Frontend (`LazarTrack/.env`):
```bash
# Backend REST API Base URL
EXPO_PUBLIC_EXTRACTION_API_URL=http://localhost:8000
```

---

## 5. Deployment & Production Considerations

- **Docker Containerization**:
  The `Dockerfile` packages Python 3.10, FFmpeg with hardware acceleration hooks, and OpenCV headless libraries.
- **Stateless Cloud Run Friendly**:
  Replacing heavy local PyTorch / EasyOCR models with OpenRouter API calls reduced container image size from ~4.5 GB to ~600 MB and cold start times from 45 seconds to under 3 seconds.
- **Session Cleanup**:
  Video processing creates temporary disk artifacts. The backend includes automated background cleanup routines (`auto_cleanup_hours`) to prevent disk exhaustion.
