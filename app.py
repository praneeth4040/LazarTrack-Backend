"""
LazarTrack – Free Fire Match Stats Extraction Backend
Deployed on HuggingFace Spaces (Gradio – Blank SDK).

Usage:
  - POST /api/v1/pipeline/match_stats  → structured JSON extraction
  - GET  /api/v1/health                → health check
"""

import os
import asyncio
import logging
from io import BytesIO

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request

from app.core.config import settings, setup_logging
from app.api.router import api_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
log = logging.getLogger("lazartrack.space")

# ---------------------------------------------------------------------------
# Detect hardware
# ---------------------------------------------------------------------------
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            log.info("GPU detected: %s", name)
            return f"GPU – {name}"
    except Exception as e:
        log.warning("GPU check failed: %s", e)
    log.info("No GPU detected – using CPU")
    return "CPU"

DEVICE = _detect_device()

# ---------------------------------------------------------------------------
# FastAPI app (the real extraction API)
# ---------------------------------------------------------------------------
fastapi_app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "5-step intelligent extraction pipeline for Free Fire match "
        "scoreboard screenshots. Device: " + DEVICE
    ),
    redoc_url=None,
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@fastapi_app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


fastapi_app.include_router(api_router, prefix=settings.API_V1_STR)

# ---------------------------------------------------------------------------
# Gradio UI (visible on the Space landing page)
# ---------------------------------------------------------------------------
def analyze_screenshot(image) -> tuple[str, str]:
    """
    Gradio inference function: accepts a PIL Image, runs the pipeline,
    returns a (JSON string, device string) tuple.
    """
    if image is None:
        return "No image provided.", DEVICE

    import json
    from PIL import Image
    from app.pipelines.five_step import run_five_step_pipeline

    buf = BytesIO()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    try:
        result = run_five_step_pipeline(raw_bytes, debug=False)
        return json.dumps(result, indent=2, ensure_ascii=False), DEVICE
    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=True)
        return f"Error: {exc}", DEVICE


with gr.Blocks(title="LazarTrack – Match Stats Extractor") as demo:
    gr.Markdown(
        """
        # 🎯 LazarTrack – Free Fire Match Stats Extractor
        Upload a Free Fire match scoreboard screenshot and get structured player stats in JSON.

        **API Endpoints (for the mobile app):**
        - `POST /api/v1/pipeline/match_stats` – upload an image file, get JSON stats
        - `GET /api/v1/health` – health check
        """
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Scoreboard Screenshot", type="pil")
            run_btn = gr.Button("Extract Stats", variant="primary")
        with gr.Column():
            json_output = gr.Textbox(label="Extracted Stats (JSON)", lines=25, show_copy_button=True)
            device_label = gr.Textbox(label="Running on", value=DEVICE, interactive=False)

    run_btn.click(fn=analyze_screenshot, inputs=[img_input], outputs=[json_output, device_label])

    gr.Markdown(
        """
        ---
        > Built with ❤️ by Lazarflow | LazarTrack v1.0
        """
    )

# ---------------------------------------------------------------------------
# Mount Gradio inside FastAPI so both are served on port 7860
# ---------------------------------------------------------------------------
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    log.info("Starting LazarTrack server on port %d | Device: %s", port, DEVICE)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
