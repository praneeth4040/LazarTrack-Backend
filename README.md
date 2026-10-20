---
title: LazarTrack Backend
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Free Fire esports analysis — Player Stats, Battle Log & Zone Tracker
---

# 🎯 LazarTrack Backend

Production-grade, end-to-end computer vision and multimodal AI analytics suite for competitive Free Fire esports matches.

---

## 📑 In-Depth Documentation

Comprehensive technical documentation is maintained in the [`docs/`](docs/README.md) directory:

- 🏛️ **[System Architecture](docs/SYSTEM_ARCHITECTURE.md)**: End-to-end design, data flows, and external AI models.
- 📊 **[Player Stats Extraction Pipeline](docs/PLAYER_STATS_PIPELINE.md)**: 5-step scoreboard extraction via OpenCV and Qwen 3.7 Flash.
- ⚔️ **[Battle Log Video Pipeline](docs/BATTLE_LOG_PIPELINE.md)**: 7-step video engine with decoupled on-demand vision extraction.
- 🗺️ **[Zone Tracker Pipeline](docs/ZONE_TRACKER_PIPELINE.md)**: 8-step minimap SIFT alignment and safe-zone detection.
- 📡 **[REST API Reference](docs/API_REFERENCE.md)**: Full endpoint reference, schemas, and curl examples.

---

## 🚀 The Three Core Intelligence Pipelines

| Pipeline | Code | Input | Output | Model / Tech |
|---|---|---|---|---|
| **Player Stats Extraction** | `PSE-001` | Scoreboard Screenshot (PNG/JPG) | Structured Match Stats JSON (`map`, `rank`, `players[]` with 10 metrics) | OpenCV + `qwen/qwen3.7-flash` (via OpenRouter) |
| **Battle Log Extractor** | `BLE-001` | Match Recording Video (MP4/MOV) | Isolated Minimap Video (`map_video.mp4`), Event Groups (`group_*.png`), and On-Demand Events (`events.json`) | FFmpeg + OpenCV + SSIM + Qwen 3.7 Flash |
| **Zone Tracker** | `ZTE-001` | Match Video + Map Name | Rendered Zone Map (`rendered_map.png`) & Circle Coordinates (`zones.json`) | SIFT Homography + Hough Transform + DBSCAN |

---

## 🗺️ Zone Tracker — 8-Step Pipeline

Detects every safe-zone circle shrink from a Free Fire match video and renders them onto the reference map.

```
Input: MP4 match recording + map name
      │
      ▼
Step 1 │ Normalizer        Scale canvas to 1920×1080 (hardware-accelerated FFmpeg)
Step 2 │ Map Separator     Crop minimap stream from normalised video (FFmpeg)
Step 3 │ Map Cleaner       Remove HUD anchors, isolate clean map bounds (FFmpeg)
Step 4 │ Map Aligner       SIFT feature alignment to reference blank map
Step 5 │ Frame Registrar   Sample unique minimap frames at up to 5 FPS
Step 6 │ Zone Detector     Parallel Hough + lit-boundary circle detection (8-core)
Step 7 │ Zone Clusterer    DBSCAN consensus clustering → final zone list
Step 8 │ Zone Renderer     Render white zones with dark halos onto base map PNG
      │
      ▼
Output: rendered_map.png (zones overlaid on reference map)
        zones.json       (centre x/y, radius, confidence per zone)
```

### Zone Tracker API Examples
```bash
# Process a match video
curl -X POST http://localhost:8000/api/v1/zone/process \
  -F "file=@match.mp4" \
  -F "map_name=bermuda"

# Get rendered map image
curl http://localhost:8000/api/v1/zone/{video_id}/zones-map --output map.png

# Interactive visual inspection dashboard
open http://localhost:8000/api/v1/zone/{video_id}/visual
```

---

## ⚔️ Battle Log Extractor — 7-Step Decoupled Pipeline

Extracts structured kill/death events from the Free Fire battle log video overlay.

```
Input Video (MP4 / MOV / AVI)
      │
      ▼
Step 1 │ Normalizer        Scale to 1920×1080 @ 5 FPS via FFmpeg
Step 2 │ Separator         Locates "BATTLE LOG" header; splits Battle Log (960×499) + Map (760×760)
Step 3 │ Player Identifier Extracts team roster frame (team_image.png) & player IGNs (players.json)
Step 4 │ Row Divider       Segments battle log into 8 horizontal slot rows
Step 5 │ Event Tracker     Tracks multi-row lifecycle & deduplicates repetitive frames
Step 6 │ Group Packer      Packs 8 deduplicated event crops into composite group cards (group_*.png)
      │
      ├───────────────────► USER DELIVERABLES RETURNED (~30s)
      │                     (group_*.png, map_video.mp4, team_image.png)
      │
Step 7 │ Vision Extractor  [ON-DEMAND ADMIN CALL] Qwen 3.7 Flash structured transcription (events.json)
```

### Battle Log API Examples
```bash
# 1. Upload video — runs Steps 1–6 automatically (~30s)
curl -X POST http://localhost:8000/api/v1/battle-log/extract \
  -F "file=@match.mp4"

# 2. Trigger Step 7 on-demand (Admin portal / suggestion engine builder)
curl -X POST http://localhost:8000/api/v1/battle-log/{video_id}/extract-events

# 3. Download structured events JSON
curl http://localhost:8000/api/v1/battle-log/{video_id}/events.json
```

---

## 📊 Player Stats Extraction — 5-Step Pipeline

Extracts all 10 in-game player combat stats with 100% precision from post-match scoreboard screenshots.

```bash
# Extract match stats from screenshot
curl -X POST http://localhost:8000/api/v1/pipeline/match_stats \
  -F "file=@assets/sample_scoreboard.png"

# Visual debug UI
open http://localhost:8000/api/v1/visualize
```

---

## 🌐 REST API Directory

Start the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

| Domain | Method | Endpoint | Description |
|---|---|---|---|
| **System** | `GET` | `/` | API root directory |
| | `GET` | `/endpoints` | Dynamic registered route listing |
| | `GET` | `/api/v1/health` | Service health & models info |
| | `GET` | `/docs` | Interactive Swagger UI |
| **Player Stats** | `POST` | `/api/v1/pipeline/match_stats` | Extract stats from scoreboard screenshot |
| | `POST` | `/api/v1/pipeline/match_stats/debug`| Debug mode with base64 step images |
| | `GET` | `/api/v1/visualize` | Interactive browser visualizer |
| **Battle Log** | `POST` | `/api/v1/battle-log/extract` | Upload video (Runs Steps 1–6) |
| | `POST` | `/api/v1/battle-log/{id}/extract-events`| Run Step 7 Vision extraction on-demand |
| | `GET` | `/api/v1/battle-log/{id}/group-images` | List packed Step 6 event cards |
| | `GET` | `/api/v1/battle-log/{id}/map-video` | Stream isolated 760×760 minimap video |
| | `GET` | `/api/v1/battle-log/{id}/team-image`| Download team roster frame |
| | `GET` | `/api/v1/battle-log/{id}/events.json`| Download extracted events |
| | `DELETE`| `/api/v1/battle-log/{id}` | Clean up session files |
| **Zone Tracker**| `POST` | `/api/v1/zone/process` | Process video for safe zones |
| | `GET` | `/api/v1/zone/list` | List processed sessions |
| | `GET` | `/api/v1/zone/maps` | List supported map templates |
| | `GET` | `/api/v1/zone/{id}/status` | Processing status |
| | `GET` | `/api/v1/zone/{id}/zones` | Download raw `zones.json` |
| | `GET` | `/api/v1/zone/{id}/zones-map` | Download rendered PNG map |
| | `GET` | `/api/v1/zone/{id}/visual` | Visual inspection dashboard |

---

## 🐳 Docker

```bash
docker build -t lazartrack-backend .
docker run -p 8000:8000 --env-file .env lazartrack-backend
```

---

## 📁 Project Structure

```
Backend/
├── app/
│   ├── api/v1/                  # API routers (battle_log, extract, zone_tracker, visualize, health)
│   ├── core/                    # Settings & logging configuration
│   ├── ocr/                     # Smart table detector & normalization
│   ├── pipelines/               # Orchestrators (five_step.py, video_pipeline.py)
│   ├── schemas/                 # Pydantic models
│   └── zone_tracker/            # 8-step safe zone detection modules
├── assets/maps/                 # Blank reference map images (Bermuda, Purgatory, etc.)
├── docs/                        # Comprehensive technical documentation
│   ├── README.md                # Documentation index
│   ├── SYSTEM_ARCHITECTURE.md   # Overall system design
│   ├── PLAYER_STATS_PIPELINE.md # Scoreboard extraction guide
│   ├── BATTLE_LOG_PIPELINE.md   # Video battle log guide
│   ├── ZONE_TRACKER_PIPELINE.md # Safe zone tracking guide
│   └── API_REFERENCE.md         # Complete REST API reference
├── scripts/                     # Standalone step runners & batch scripts
├── video/
│   ├── input/                   # Raw match videos (gitignored)
│   └── output/                  # Pipeline session outputs (gitignored)
├── Dockerfile
├── requirements.txt
└── README.md
```

## 📄 License

MIT
