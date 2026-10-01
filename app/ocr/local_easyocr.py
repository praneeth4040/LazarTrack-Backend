import logging
import cv2
import numpy as np
import httpx
import easyocr
from typing import List, Dict, Any, Union
from app.core.config import settings

log = logging.getLogger("app.ocr.remote")

# Singleton reader instance for local fallback
_reader = None

def get_easyocr_reader():
    """Returns local EasyOCR reader instance (used as fallback or wrapper)."""
    global _reader
    if _reader is None:
        log.info("Initializing local EasyOCR reader instance (fallback)...")
        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception as e:
            log.warning("Could not check CUDA GPU availability (%s), defaulting to CPU", e)
        _reader = easyocr.Reader(['en'], gpu=use_gpu)
    return RemoteEasyOCRWrapper(_reader)


class RemoteEasyOCRWrapper:
    """
    Wrapper presenting an easyocr.Reader interface (`readtext()`),
    primary call goes to HuggingFace deployed EasyOCR microservice, with local EasyOCR as fallback.
    """
    def __init__(self, fallback_reader):
        self.fallback = fallback_reader
        self.remote_url = settings.EASYOCR_URL.rstrip("/")

    def readtext(self, img_input: Union[np.ndarray, bytes], allowlist: str = None, detail: int = 0, **kwargs):
        # Prepare image bytes for HTTP POST
        if isinstance(img_input, np.ndarray):
            success, buf = cv2.imencode(".png", img_input)
            img_bytes = buf.tobytes() if success else None
        elif isinstance(img_input, bytes):
            img_bytes = img_input
        else:
            img_bytes = None

        if img_bytes and self.remote_url:
            try:
                endpoint = f"{self.remote_url}/ocr" if not self.remote_url.endswith("/ocr") else self.remote_url
                params = {}
                if allowlist:
                    params["allowlist"] = allowlist
                files = {"file": ("image.png", img_bytes, "image/png")}
                
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(endpoint, files=files, params=params)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_items = data.get("results", data.get("text", []))
                        if isinstance(raw_items, list):
                            texts = []
                            detailed_items = []
                            for item in raw_items:
                                if isinstance(item, dict):
                                    t = item.get("text", "")
                                    b = item.get("bbox", [])
                                    c = item.get("confidence", 1.0)
                                    detailed_items.append((b, t, c))
                                    texts.append(t)
                                elif isinstance(item, str):
                                    texts.append(item)
                                    detailed_items.append(([], item, 1.0))
                            return detailed_items if detail != 0 else texts
            except Exception as exc:
                log.warning("Remote deployed EasyOCR request failed (%s). Falling back to local EasyOCR.", exc)

        # Fallback to local EasyOCR execution
        return self.fallback.readtext(img_input, allowlist=allowlist, detail=detail, **kwargs)


def run_local_easyocr(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Runs primary remote / fallback local EasyOCR on raw image bytes.
    """
    wrapper = get_easyocr_reader()
    results = wrapper.readtext(image_bytes, detail=1)
    
    formatted_items = []
    for bbox, text, confidence in results:
        pts = [[float(point[0]), float(point[1])] for point in bbox] if bbox else []
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

