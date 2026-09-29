import os
import logging
import numpy as np
from typing import List, Dict, Any

# Disable model source check for fast initialization
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

log = logging.getLogger("app.ocr.paddle")
_PADDLE_OCR_READER = None


def get_paddle_ocr_reader():
    """
    Returns a singleton instance of PaddleOCR with dynamic GPU detection.
    """
    global _PADDLE_OCR_READER
    if _PADDLE_OCR_READER is None:
        log.info("[PaddleOCR] Initializing PP-OCRv5 model...")
        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            pass
        from paddleocr import PaddleOCR
        _PADDLE_OCR_READER = PaddleOCR(lang="en", use_gpu=use_gpu)
        log.info("[PaddleOCR] Loaded successfully with use_gpu=%s.", use_gpu)
    return _PADDLE_OCR_READER


def run_paddle_ocr(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Runs PaddleOCR on a given image numpy array and returns structured detections.
    """
    reader = get_paddle_ocr_reader()
    res = reader.predict(img)

    results = []
    if not res or not res[0]:
        return results

    rec_data = res[0]
    rec_texts = rec_data.get("rec_texts", [])
    rec_scores = rec_data.get("rec_scores", [])
    rec_boxes = rec_data.get("rec_polys", [])

    for box, text, score in zip(rec_boxes, rec_texts, rec_scores):
        pts = box.tolist() if hasattr(box, "tolist") else list(box)
        results.append({
            "text": str(text),
            "confidence": round(float(score), 3),
            "bbox": pts
        })

    return results
