---
title: LazarTrack Backend
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Free Fire match battle log video extraction pipeline
---

# 🎯 LazarTrack – Production Battle Log Extraction Pipeline

Production-grade, end-to-end video-to-JSON pipeline for extracting structured battle log events from Free Fire match recordings.

## 🎬 7-Step Pipeline Architecture

```
Input Video (MP4 / MOV / AVI)
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Step 1: Normalizer (scripts/step1_normalizer.py)       │
│ Scales to 1920×1080 @ 5FPS via FFmpeg/OpenCV           │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Step 2: Separator (scripts/step2_separator.py)         │
│ Multi-frame anchor scan for "BATTLE LOG" header       │
│ Splits into: Battle Log (960×499) + Map Video (760×760)│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Step 3: Player Identifier (scripts/step3_player_id.py) │
│ Extracts team roster & player slots via Qwen VL        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Step 4: Row Divider (scripts/step4_row_divider.py)     │
│ Segments battle log into 8 horizontal slot rows        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Step 5: Event Tracker (scripts/step5_event_tracker.py) │
│ Tracks multi-row entry/exit lifecycle & dedups events  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Step 6: Group Image Packer (scripts/step6_group_packer)│
│ Packs 8 deduplicated event crops per group image       │
│ Generates manifest and high-density composite grid     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Step 7: Vision Extractor (scripts/step7_vision_extract)│
│ Extracts structured JSON events with Markdown defense  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
Output: events.json + map_video.mp4 + team_image.png + group_images/
```

## 🚀 Running the Pipeline

### 1. Unified Programmatic / CLI Pipeline
```bash
# Run complete 7-step pipeline on any video
python3 -c "
from app.pipelines.video_pipeline import run_complete_video_pipeline
result = run_complete_video_pipeline('video/input/1.mp4', video_id='match_1', output_base_dir='video/output/match_1')
print('Events:', len(result['events']))
"
```

### 2. Standalone Step Runners (Targeted or Batch)
```bash
# Step 1: Normalize a single video or directory
python3 scripts/step1_normalizer.py --input video/input/1.mp4 --output video/output/step1 --video 1

# Step 2: Separate Battle Log and Minimap
python3 scripts/step2_separator.py --step1 video/output/step1 --out video/output/step2 --video 1

# Step 3: Player Identifier
python3 scripts/step3_player_identifier.py --step2 video/output/step2 --out video/output/step3 --video 1

# Step 4: Row Divider
python3 scripts/step4_row_divider.py --step2 video/output/step2 --out video/output/step4 --video 1

# Step 5: Event Tracker
python3 scripts/step5_event_tracker.py --step4 video/output/step4 --out video/output/step5 --video 1

# Step 6: Group Image Packer
python3 scripts/step6_group_packer.py --step5 video/output/step5 --out video/output/step6 --video 1

# Step 7: Vision Model Extraction
python3 scripts/step7_vision_extractor.py --step3 video/output/step3 --step6 video/output/step6 --out video/output/step7 --video 1
```

## 🌐 Production REST API

Start the API server:
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Endpoints
- `POST /api/v1/battle-log/extract`: Process video upload (non-blocking thread pool execution)
- `GET /api/v1/battle-log/{video_id}/events.json`: Download extracted structured events
- `GET /api/v1/battle-log/{video_id}/map-video`: Stream/download isolated minimap video
- `GET /api/v1/battle-log/{video_id}/team-image`: Download identified team frame
- `GET /api/v1/battle-log/{video_id}/group-images`: List all packed group images
- `GET /api/v1/battle-log/{video_id}/group-image/{filename}`: Download individual group image
- `DELETE /api/v1/battle-log/{video_id}`: Clean up output session files

## 🐳 Docker

```bash
docker build -t lazartrack-backend .
docker run -p 8000:8000 --env-file .env lazartrack-backend
```

## 📄 License

MIT