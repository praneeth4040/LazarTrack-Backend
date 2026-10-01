import logging
import cv2
import numpy as np
import httpx
import easyocr
from typing import List, Dict, Any, Union
from app.core.config import settings
import torch

log = logging.getLogger("app.ocr.remote")

# Singleton reader instance for local fallback — initialized lazily on first remote failure
_reader = None

def get_raw_local_easyocr_reader():
    """Returns pure local EasyOCR reader instance without remote wrapper."""
    global _reader
    if _reader is None:
        log.info("Initializing local EasyOCR reader instance...")
        use_gpu = False
        try:
            use_gpu = torch.cuda.is_available()
        except Exception as e:
            log.warning("Could not check CUDA GPU availability (%s), defaulting to CPU", e)
        _reader = easyocr.Reader(['en'], gpu=use_gpu)
    return _reader

def get_easyocr_reader():
    """Returns RemoteEasyOCRWrapper. Local EasyOCR is only initialized if remote fails."""
    return RemoteEasyOCRWrapper(fallback_factory=get_raw_local_easyocr_reader)


class RemoteEasyOCRWrapper:
    """
    Wrapper presenting an easyocr.Reader interface (`readtext()`).
    Primary call goes to HuggingFace deployed EasyOCR microservice.
    Local EasyOCR is only initialized on the first remote failure (lazy fallback).
    """
    def __init__(self, fallback_factory=None, fallback_reader=None):
        # Support both a pre-built reader (legacy) and a lazy factory callable
        self._fallback_factory = fallback_factory
        self._fallback = fallback_reader  # None until first remote failure
        self.remote_url = settings.EASYOCR_URL.rstrip("/")

    @property
    def fallback(self):
        """Lazily initializes and caches the local EasyOCR reader on first access."""
        if self._fallback is None:
            if self._fallback_factory is not None:
                self._fallback = self._fallback_factory()
            else:
                self._fallback = get_raw_local_easyocr_reader()
        return self._fallback

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
            # Route to /ocr/detail when caller passes preprocess= or scale=
            # These kwargs signal the new single-call path with confidence support.
            # Legacy callers (no preprocess/scale) continue hitting /ocr unchanged.
            use_detail_endpoint = "preprocess" in kwargs or "scale" in kwargs

            try:
                if use_detail_endpoint:
                    return self._call_detail_endpoint(
                        img_bytes=img_bytes,
                        allowlist=allowlist,
                        detail=detail,
                        preprocess=kwargs.get("preprocess", "contrast"),
                        scale=kwargs.get("scale", 4),
                    )
                else:
                    return self._call_legacy_endpoint(
                        img_bytes=img_bytes,
                        allowlist=allowlist,
                        detail=detail,
                    )
            except Exception as exc:
                log.warning("Remote OCR request failed (%s). Falling back to local EasyOCR.", exc)

        # Fallback to local EasyOCR execution (initializes local model on first call)
        return self.fallback.readtext(img_input, allowlist=allowlist, detail=detail)

    def _call_detail_endpoint(self, img_bytes: bytes, allowlist: str, detail: int,
                               preprocess: str, scale: int):
        """
        Calls POST /ocr/detail — applies preprocessing server-side, returns
        { results: [ { text, confidence, bbox } ] } natively.
        """
        endpoint = f"{self.remote_url}/ocr/detail"
        params = {"preprocess": preprocess, "scale": scale}
        if allowlist:
            params["allowlist"] = allowlist
        files = {"file": ("image.png", img_bytes, "image/png")}

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(endpoint, files=files, params=params)

        if resp.status_code != 200:
            raise RuntimeError(f"/ocr/detail returned HTTP {resp.status_code}")

        items = resp.json().get("results", [])
        # items: [ { "text": str, "confidence": float, "bbox": [[x,y],...] } ]
        detailed = [(item["bbox"], item["text"], float(item["confidence"])) for item in items]
        if detail != 0:
            return detailed
        return [item["text"] for item in items]

    def _call_legacy_endpoint(self, img_bytes: bytes, allowlist: str, detail: int):
        """
        Calls POST /ocr — the original endpoint, untouched, for existing pipelines.
        """
        endpoint = f"{self.remote_url}/ocr"
        params = {}
        if allowlist:
            params["allowlist"] = allowlist
        files = {"file": ("image.png", img_bytes, "image/png")}

        with httpx.Client(timeout=10.0) as client:
            resp = client.post(endpoint, files=files, params=params)

        if resp.status_code != 200:
            raise RuntimeError(f"/ocr returned HTTP {resp.status_code}")

        data = resp.json()
        raw_items = data.get("results", data.get("text", []))

        if isinstance(raw_items, str):
            lines = [line.strip() for line in raw_items.splitlines() if line.strip()]
            _zero_bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]
            return [(_zero_bbox, l, 1.0) for l in lines] if detail != 0 else lines

        if isinstance(raw_items, list):
            texts = []
            detailed_items = []
            for item in raw_items:
                if isinstance(item, dict):
                    t = item.get("text", "")
                    b = item.get("bbox", [[0,0],[0,0],[0,0],[0,0]])
                    c = item.get("confidence", 1.0)
                    detailed_items.append((b, t, c))
                    texts.append(t)
                elif isinstance(item, str):
                    _zero_bbox = [[0, 0], [0, 0], [0, 0], [0, 0]]
                    detailed_items.append((_zero_bbox, item, 1.0))
                    texts.append(item)
            return detailed_items if detail != 0 else texts

        return []


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

