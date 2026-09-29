---
title: LazarTrack Backend
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
short_description: Free Fire match scoreboard OCR extraction API
---


# 🎯 LazarTrack – Match Stats Extractor

FastAPI + Gradio backend for the **LazarTrack** Free Fire match tracking app.

## What it does
Upload a Free Fire match scoreboard screenshot → get structured player stats (kills, damage, assists, rank, map, etc.) as JSON via a 5-step OCR pipeline.

## API Endpoints (for mobile app)
- `POST /api/v1/pipeline/match_stats` – upload image → returns structured JSON
- `GET /api/v1/health` – health check

## Tech Stack
- **EasyOCR** – text extraction (GPU if available, else CPU)
- **FastAPI** – REST API
- **Gradio** – Space UI
