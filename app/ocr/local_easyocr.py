"""
Local EasyOCR — single, process-wide singleton reader.

All callers use get_easyocr_reader() which returns the same instance after the
first call. Model weights (~200 MB) are loaded once from ~/.EasyOCR/ cache.
"""

import logging
import cv2
import numpy as np
import easyocr
import torch
from typing import List, Dict, Any

log = logging.getLogger("app.ocr.local")

# Process-wide singleton — initialized lazily on first call
_reader: easyocr.Reader | None = None


def get_easyocr_reader() -> easyocr.Reader:
    """
    Returns the process-wide EasyOCR reader singleton.
    Initialized on first call; subsequent calls return the cached instance.
    GPU is used automatically if CUDA is available (CPU otherwise).
    """
    global _reader
    if _reader is None:
        use_gpu = False
        try:
            use_gpu = torch.cuda.is_available()
        except Exception as e:
            log.warning("Could not check CUDA availability (%s) — defaulting to CPU", e)
        log.info("Initializing local EasyOCR reader (gpu=%s)…", use_gpu)
        _reader = easyocr.Reader(["en"], gpu=use_gpu)
        log.info("EasyOCR reader ready.")
    return _reader


# Alias kept for backwards-compat with callers that import get_raw_local_easyocr_reader
get_raw_local_easyocr_reader = get_easyocr_reader


def run_local_easyocr(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Runs local EasyOCR on raw image bytes and returns structured results.
    """
    reader = get_easyocr_reader()
    results = reader.readtext(image_bytes, detail=1)

    formatted_items = []
    for bbox, text, confidence in results:
        pts = [[float(pt[0]), float(pt[1])] for pt in bbox] if bbox else []
        xs = [p[0] for p in pts] if pts else [0]
        ys = [p[1] for p in pts] if pts else [0]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        formatted_items.append({
            "bbox": pts,
            "text": str(text).strip(),
            "confidence": float(confidence),
            "center_x": (min_x + max_x) / 2.0,
            "center_y": (min_y + max_y) / 2.0,
            "width": max_x - min_x,
            "height": max_y - min_y,
        })

    return formatted_items
