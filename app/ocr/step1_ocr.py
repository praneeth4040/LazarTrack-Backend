import time
import hashlib
import logging
import httpx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import HTTPException
from app.core.config import settings
from app.ocr.preprocessor import preprocess_image
from app.ocr.local_easyocr import run_local_easyocr

log = logging.getLogger("app.ocr.step1")


class BoundingBoxItem(BaseModel):
    bbox: List[List[float]] = Field(default_factory=list)
    text: str
    confidence: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0


class OCRResult(BaseModel):
    """
    Structured storage object for Step 1 OCR Output containing
    both text fragments and precise 2D bounding boxes.
    """
    request_id: str
    timestamp: float
    engine_used: str
    extracted_texts: List[str] = Field(default_factory=list)
    bounding_boxes: List[BoundingBoxItem] = Field(default_factory=list)
    raw_response: Any = None
    item_count: int = 0


class OCRResultStore:
    """
    In-memory store for raw OCR results produced by Step 1.
    """
    def __init__(self):
        self._store: Dict[str, OCRResult] = {}

    def save(self, result: OCRResult) -> None:
        self._store[result.request_id] = result
        log.info("Stored OCR result [id=%s] with %d items", result.request_id, result.item_count)

    def get(self, request_id: str) -> Optional[OCRResult]:
        return self._store.get(request_id)

    def clear(self) -> None:
        self._store.clear()


ocr_store = OCRResultStore()


async def execute_step1_ocr(
    raw_image_bytes: bytes,
    filename: str = "screenshot.jpg",
    use_local: bool = True
) -> OCRResult:
    """
    STEP 1 OF PIPELINE:
    -------------------
    1. Preprocess raw image bytes.
    2. Execute Local EasyOCR (or remote fallback) to extract text + 2D Bounding Boxes.
    3. Store bounding box structures in OCRResult object.
    4. Save to OCR store and return result object.
    """
    processed_bytes = preprocess_image(raw_image_bytes)
    image_hash = hashlib.md5(processed_bytes).hexdigest()[:12]
    request_id = f"ocr_{image_hash}_{int(time.time())}"

    bounding_boxes: List[BoundingBoxItem] = []
    extracted_texts: List[str] = []
    engine_used = "local_easyocr"
    raw_payload = None

    if use_local:
        try:
            log.info("[Step 1] Running LOCAL EasyOCR...")
            local_items = run_local_easyocr(processed_bytes)
            raw_payload = local_items

            for item in local_items:
                box_obj = BoundingBoxItem(**item)
                bounding_boxes.append(box_obj)
                extracted_texts.append(box_obj.text)

            log.info("[Step 1] Local EasyOCR completed with %d items", len(bounding_boxes))
        except Exception as exc:
            log.error("[Step 1] Local EasyOCR failed (%s). Falling back to remote HF Space...", exc)
            use_local = False

    if not use_local:
        settings.validate()
        base_url = settings.EASYOCR_URL.rstrip("/")
        endpoint = f"{base_url}/ocr" if not base_url.endswith("/ocr") else base_url
        engine_used = f"remote:{endpoint}"

        log.info("[Step 1] POSTing to Remote EasyOCR endpoint: %s", endpoint)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    endpoint,
                    files={"image": (filename, processed_bytes, "image/jpeg")},
                )
                if response.status_code in (404, 405):
                    response = await client.post(base_url, files={"image": (filename, processed_bytes, "image/jpeg")})
                if response.status_code == 422 and "file" in response.text.lower():
                    response = await client.post(endpoint, files={"file": (filename, processed_bytes, "image/jpeg")})

            if response.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Remote OCR error {response.status_code}")

            raw_payload = response.json()
            if isinstance(raw_payload, dict) and "text" in raw_payload:
                for line in raw_payload["text"].split("\n"):
                    line_clean = line.strip()
                    if line_clean:
                        extracted_texts.append(line_clean)

        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"OCR execution failed: {exc}")

    ocr_result = OCRResult(
        request_id=request_id,
        timestamp=time.time(),
        engine_used=engine_used,
        extracted_texts=extracted_texts,
        bounding_boxes=bounding_boxes,
        raw_response=raw_payload,
        item_count=len(extracted_texts),
    )

    ocr_store.save(ocr_result)
    return ocr_result
