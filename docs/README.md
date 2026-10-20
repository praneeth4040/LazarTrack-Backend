# 📚 LazarTrack Backend Documentation Hub

Welcome to the technical documentation for the **LazarTrack Backend** esports analytics engine.

---

## 📑 Core Documentation Index

| Document | Topic | Description |
|---|---|---|
| [System Architecture](file:///Users/praneeth/Lazarflow/LazarTrack/Backend/docs/SYSTEM_ARCHITECTURE.md) | High-Level Architecture | End-to-end overview of frontend/backend design, data flows, and external AI integrations. |
| [Player Stats Pipeline](file:///Users/praneeth/Lazarflow/LazarTrack/Backend/docs/PLAYER_STATS_PIPELINE.md) | Scoreboard Extraction (`PSE-001`) | 5-step hybrid computer vision + Qwen 3.7 Flash structured extraction engine. |
| [Battle Log Pipeline](file:///Users/praneeth/Lazarflow/LazarTrack/Backend/docs/BATTLE_LOG_PIPELINE.md) | Video Kill Feed (`BLE-001`) | 7-step video pipeline, decoupled Step 7 admin on-demand trigger, and composite group packing. |
| [Zone Tracker Pipeline](file:///Users/praneeth/Lazarflow/LazarTrack/Backend/docs/ZONE_TRACKER_PIPELINE.md) | Minimap Tracking (`ZTE-001`) | 8-step SIFT alignment, Hough circle detection, DBSCAN clustering, and map renderer. |
| [REST API Reference](file:///Users/praneeth/Lazarflow/LazarTrack/Backend/docs/API_REFERENCE.md) | API Catalog | Complete HTTP API documentation with request schemas, query parameters, and curl examples. |

---

## 🚀 Quick Start Links

- **Interactive API Docs (Swagger UI)**: `http://localhost:8000/docs`
- **Player Stats Visual Debugger**: `http://localhost:8000/api/v1/visualize`
- **Zone Tracker Visual Inspector**: `http://localhost:8000/api/v1/zone/{video_id}/visual`
- **Dynamic Endpoint Directory**: `http://localhost:8000/endpoints`
