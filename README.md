---
title: LazarTrack Backend
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Free Fire esports analysis — Battle Log Extraction & Zone Tracker
---

# 🎯 LazarTrack Backend

Production-grade, end-to-end video analysis suite for Free Fire esports matches.  
Two independent pipelines:

| Pipeline | What It Does |
|---|---|
| **Battle Log Extractor** | Extracts structured kill/elimination events from battle log video |
| **Zone Tracker** | Detects and renders all safe-zone circles onto the minimap |

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

### Running via API
```bash
# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Process a match video
curl -X POST http://localhost:8000/api/v1/zone/process \
  -F "file=@match.mp4" \
  -F "map_name=bermuda"

# Get rendered map image
curl http://localhost:8000/api/v1/zone/{video_id}/zones-map --output map.png

# Interactive visual inspection dashboard
open http://localhost:8000/api/v1/zone/{video_id}/visual
```

### Running via script (batch)
```bash
python3 scripts/run_all_zone_pipeline.py
```

### Supported Maps
Reference map images live in `assets/maps/`:
- `bermuda` · `purgatory` · `kalahari` · `solara` · `nexterra`

---

## 📋 Battle Log Extractor — 7-Step Pipeline

Extracts structured kill/death events from Free Fire battle log video overlay.

```
Input Video (MP4 / MOV / AVI)
      │
      ▼
Step 1 │ Normalizer        Scale to 1920×1080 @ 5 FPS via FFmpeg
Step 2 │ Separator         Multi-frame scan for "BATTLE LOG" header
       │                   Splits into: Battle Log (960×499) + Map (760×760)
Step 3 │ Player Identifier Extract team roster & player slots via Qwen VL
Step 4 │ Row Divider       Segment battle log into 8 horizontal slot rows
Step 5 │ Event Tracker     Track multi-row entry/exit lifecycle & dedup events
Step 6 │ Group Packer      Pack 8 deduplicated event crops per group image
Step 7 │ Vision Extractor  Structured JSON events via vision model
      │
      ▼
Output: events.json + map_video.mp4 + team_image.png + group_images/
```

### Running the battle log pipeline
```bash
python3 -c "
from app.pipelines.video_pipeline import run_complete_video_pipeline
result = run_complete_video_pipeline('video/input/1.mp4', video_id='match_1', output_base_dir='video/output/match_1')
print('Events:', len(result['events']))
"
```

---

## 🌐 REST API Endpoints

Start the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Zone Tracker
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/zone/process` | Upload MP4 or provide `video_path` + `map_name` |
| `GET` | `/api/v1/zone/list` | List all processed sessions |
| `GET` | `/api/v1/zone/maps` | List supported map templates |
| `GET` | `/api/v1/zone/{id}/status` | Session processing status |
| `GET` | `/api/v1/zone/{id}/zones` | Raw zone JSON (centre, radius, confidence) |
| `GET` | `/api/v1/zone/{id}/zones-map` | Rendered PNG map with zones |
| `GET` | `/api/v1/zone/{id}/step/{n}/{artifact}` | Download step artifact file |
| `GET` | `/api/v1/zone/{id}/visual` | Interactive 8-step visual inspection dashboard |

### Battle Log Extractor
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/battle-log/extract` | Process video upload |
| `GET` | `/api/v1/battle-log/{id}/events.json` | Download extracted events |
| `GET` | `/api/v1/battle-log/{id}/map-video` | Stream isolated minimap video |
| `GET` | `/api/v1/battle-log/{id}/team-image` | Download team roster frame |
| `DELETE` | `/api/v1/battle-log/{id}` | Clean up session files |

### Match Stats OCR
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/extract` | OCR match stats screenshot |
| `GET` | `/api/v1/visualize` | Stats visualizer |

### System
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API root — endpoint directory |
| `GET` | `/endpoints` | Dynamic route listing |
| `GET` | `/api/v1/health` | Service health & available pipelines |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🐳 Docker

```bash
docker build -t lazartrack-backend .
docker run -p 8000:8000 --env-file .env lazartrack-backend
```

## 📁 Project Structure

```
Backend/
├── app/
│   ├── api/v1/
│   │   ├── zone_tracker.py      # Zone tracker endpoints & visual dashboard
│   │   ├── battle_log.py        # Battle log extractor endpoints
│   │   ├── extract.py           # Match stats OCR
│   │   └── health.py
│   ├── zone_tracker/            # 8-step zone detection pipeline
│   │   ├── pipeline.py
│   │   ├── normalizer.py
│   │   ├── map_separator.py
│   │   ├── map_cleaner.py
│   │   ├── map_aligner.py
│   │   ├── frame_registrar.py
│   │   ├── zone_detector.py
│   │   ├── zone_clusterer.py
│   │   └── zone_renderer.py
│   ├── pipelines/               # Battle log extraction pipeline
│   └── ocr/                     # Match stats OCR modules
├── assets/maps/                 # Reference blank map images
├── scripts/                     # Standalone step runners & batch scripts
├── video/
│   ├── input/                   # Drop raw match videos here (gitignored)
│   └── output/                  # Pipeline session outputs (gitignored)
├── Dockerfile
├── render.yaml
└── requirements.txt
```

## 📄 License

MIT

