import logging
import easyocr
from typing import List, Dict, Any

log = logging.getLogger("app.ocr.local")

# Singleton reader instance
_reader = None

def get_easyocr_reader():
    global _reader
    if _reader is None:
        log.info("Initializing local EasyOCR reader instance...")
        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception as e:
            log.warning("Could not check CUDA GPU availability (%s), defaulting to CPU", e)
        
        log.info("EasyOCR initializing with GPU=%s", use_gpu)
        _reader = easyocr.Reader(['en'], gpu=use_gpu)
    return _reader

def run_local_easyocr(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Runs local EasyOCR on raw image bytes with tuned detection parameters
    (text_threshold=0.3, low_text=0.2) to detect standalone digits (1, 0, 2, 4).
    """
    reader = get_easyocr_reader()
    log.info("Running local EasyOCR readtext with tuned single-digit sensitivity...")
    
    # Tuned parameters for high sensitivity on single isolated digits
    results = reader.readtext(
        image_bytes,
        text_threshold=0.3,
        low_text=0.2,
        link_threshold=0.2,
        min_size=5,
    )
    
    formatted_items = []
    for bbox, text, confidence in results:
        pts = [[float(point[0]), float(point[1])] for point in bbox]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
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
        
    log.info("Local EasyOCR returned %d bounding boxes", len(formatted_items))
    return formatted_items
