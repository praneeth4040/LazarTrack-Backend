import cv2
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException

def binarize_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Enhanced Adaptive Binarization:
    Uses Adaptive Gaussian Thresholding & Morphological Cleaning.
    Preserves thin standalone digits (like 1, 0, 2, 4) without bloating progress bars.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Bilateral filter to smooth noise while keeping text edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # 2. Adaptive Gaussian Thresholding (window size 15, constant C 4)
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
    )

    # 3. Convert back to 3-channel BGR for EasyOCR compatibility
    bw_3channel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return bw_3channel


def preprocess_image(raw_bytes: bytes, target_width: int = 1920) -> bytes:
    """
    Enhanced Image Preprocessing:
    1. Resizes image to standard 1920px width for constant coordinate scaling.
    2. Applies sharpening and contrast enhancement (CLAHE) for clear OCR reading.
    """
    try:
        img_pil = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Ensure the file is a valid PNG, JPG, or WEBP.",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image processing error: {exc}")

    w, h = img_pil.size
    if w != target_width:
        scale = target_width / float(w)
        new_h = int(h * scale)
        img_pil = img_pil.resize((target_width, new_h), Image.LANCZOS)

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3.0, -0.5],
        [0, -0.5, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(img_cv, -1, kernel)

    _, buf = cv2.imencode(".jpg", sharpened, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return buf.tobytes()
