import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple
from app.ocr.preprocessor import preprocess_image

log = logging.getLogger("app.ocr.smart_table")

def detect_smart_table_roi(
    img_bgr: np.ndarray,
    target_w: int = 1920,
    target_h: int = 1080
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    STEP 1: Image Normalization (1920x1080 + CLAHE + Sharpening)
    STEP 2: Pure Computer Vision Smart Table ROI Detection via 2 Diagonal Anchors
      - Anchor 1 (Top-Left): High-contrast header block ("BATTLE ROYALE")
      - Anchor 2 (Bottom-Right): High-contrast white BACK button contour
    Returns (table_crop, annotated_img, metadata)
    """
    h, w, _ = img_bgr.shape
    if w != target_w or h != target_h:
        normalized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        normalized = img_bgr

    # Convert to HSV for high-luminance white color thresholding
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))

    # Morphological dilation to group text character contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(white_mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bounding box fallbacks
    x1, y1 = int(target_w * 0.08), int(target_h * 0.02)
    x2, y2 = int(target_w * 0.92), int(target_h * 0.85)

    top_anchor_found = False
    bottom_anchor_found = False

    top_candidates = []
    bottom_candidates = []

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        if area < 300 or area > (target_w * target_h * 0.4):
            continue

        center_x = x + bw / 2.0
        center_y = y + bh / 2.0
        aspect_ratio = bw / float(bh)

        # Candidate for Top-Left Header Anchor ("BATTLE ROYALE")
        if center_x < target_w * 0.35 and center_y < target_h * 0.20:
            top_candidates.append((x, y, bw, bh, area))

        # Candidate for Bottom-Right BACK Button Anchor
        elif center_x > target_w * 0.70 and center_y > target_h * 0.75 and 1.5 <= aspect_ratio <= 5.5:
            bottom_candidates.append((x, y, bw, bh, area))

    if top_candidates:
        top_candidates.sort(key=lambda c: c[1])
        tx, ty, _, _, _ = top_candidates[0]
        x1 = tx
        y1 = ty
        top_anchor_found = True

    if bottom_candidates:
        bottom_candidates.sort(key=lambda c: (c[0] + c[2] + c[1] + c[3]), reverse=True)
        bx, by, bbw, bbh, _ = bottom_candidates[0]
        x2 = bx + bbw
        y2 = by + bbh
        bottom_anchor_found = True

    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(target_w, x2 + padding)
    y2 = min(target_h, y2 + padding)

    raw_crop = normalized[y1:y2, x1:x2]

    # Normalize the crop to a fixed standard size so OCR always sees
    # the same scale regardless of source device (phone, tablet, 16:9, 4:3, etc.)
    TABLE_STD_W, TABLE_STD_H = 1560, 1040
    table_crop = cv2.resize(raw_crop, (TABLE_STD_W, TABLE_STD_H), interpolation=cv2.INTER_LANCZOS4)

    # Create annotated visualization canvas
    annotated = normalized.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 3)
    cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.circle(annotated, (x1, y1), 8, (0, 255, 0), -1)
    cv2.putText(annotated, "[ANCHOR 1: TOP-LEFT HEADER]", (x1 + 10, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.circle(annotated, (x2, y2), 8, (0, 0, 255), -1)
    cv2.putText(annotated, "[ANCHOR 2: BACK BUTTON]", (x2 - 300, min(target_h - 10, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    metadata = {
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
        "raw_crop_width": x2 - x1,
        "raw_crop_height": y2 - y1,
        "crop_width": TABLE_STD_W,
        "crop_height": TABLE_STD_H,
        "top_anchor_found": top_anchor_found,
        "bottom_anchor_found": bottom_anchor_found,
        "method": "Step 1: Normalization + Step 2: Pure CV Diagonal Anchors"
    }

    return table_crop, annotated, metadata


def detect_stats_button_crop(table_crop: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    STEP 2.5: Tighter Bottom Crop via Golden STATS Button Detection.
    After Step 2 produces the full 1560x1040 standardized table crop,
    this step:
      - Detects the golden/yellow STATS button using HSV color masking
      - Uses the top of the STATS button as the new bottom boundary
      - Returns a tighter crop containing only the player data rows
    Falls back to 90% of height if STATS button is not found.
    """
    h, w = table_crop.shape[:2]

    # Convert to HSV and threshold for golden/yellow (STATS button color)
    hsv = cv2.cvtColor(table_crop, cv2.COLOR_BGR2HSV)
    golden_mask = cv2.inRange(hsv, np.array([15, 100, 150]), np.array([38, 255, 255]))

    # Only search in the bottom 25% of the image where the STATS button lives
    search_mask = golden_mask.copy()
    search_mask[:int(h * 0.75), :] = 0

    contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stats_top_y = None
    stats_found = False

    if contours:
        # Pick the largest golden contour — that's the STATS button
        largest = max(contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
        x, y, bw, bh = cv2.boundingRect(largest)
        area = bw * bh
        if area > 500:  # minimum area guard against noise
            stats_top_y = y
            stats_found = True
            log.info(f"[Step 2.5] STATS button detected at y={y}, area={area}")

    if stats_top_y is None:
        # Fallback: cut at 90% of height
        stats_top_y = int(h * 0.90)
        log.warning("[Step 2.5] STATS button not found — using 90% height fallback")

    tighter_crop = table_crop[:stats_top_y, :]

    metadata = {
        "stats_button_found": stats_found,
        "stats_top_y": stats_top_y,
        "tighter_crop_width": w,
        "tighter_crop_height": stats_top_y,
    }

    return tighter_crop, metadata


def split_match_info_and_player_table(table_crop: np.ndarray, reader=None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    STEP 3: Dynamic OCR-based Rank & Tagline Anchor Detection.
    Detects the Rank (e.g. #5/11, #9/12) and Tagline text (e.g. "CLOSE, BUT NO CIGAR!",
    "YOU MADE IT TO THE TOP 10!") using OCR in the upper region (Y=40 to 320).
    Sets split_y dynamically right below the detected tagline bounding box.
    """
    h, w = table_crop.shape[:2]
    split_y = 285  # Fallback anchor Y
    detected_rank = None
    detected_tagline = None

    try:
        if reader is None:
            from app.ocr.local_easyocr import get_easyocr_reader
            reader = get_easyocr_reader()

        # Scan top region Y = 40 to 320 for Rank & Tagline text
        top_region = table_crop[40:320, :]
        results = reader.readtext(top_region, detail=1)

        tagline_bottom_y = 0
        for bbox, text, conf in results:
            abs_y1 = 40 + int(bbox[0][1])
            abs_y2 = 40 + int(bbox[2][1])
            txt_upper = text.strip().upper()

            # Rank detection (e.g. #5/11, #9/12)
            if ('#' in text or '/' in text) and abs_y1 < 250 and conf > 0.4:
                detected_rank = text.strip()

            # Dynamic Tagline detection right below rank (Y=200..290)
            elif abs_y1 > 180 and abs_y2 < 300 and conf > 0.3:
                detected_tagline = text.strip()
                if abs_y2 > tagline_bottom_y:
                    tagline_bottom_y = abs_y2

        if tagline_bottom_y > 240:
            split_y = tagline_bottom_y + 5

        log.info(f"[Step 3 OCR Anchor] Rank='{detected_rank}', Tagline='{detected_tagline}' -> split_y={split_y}")
    except Exception as e:
        log.warning(f"[Step 3 OCR Anchor] Fallback to Y=285 due to OCR exception: {e}")
        split_y = 285

    match_info_section = table_crop[:split_y, :]
    player_table_section = table_crop[split_y:, :]

    # Truncate player_table_section to 590px height if longer (to exclude STATS/BACK footer)
    if player_table_section.shape[0] > 590:
        player_table_section = player_table_section[:590, :]

    metadata = {
        "split_y": split_y,
        "detected_rank": detected_rank,
        "detected_tagline": detected_tagline,
        "match_info_height": match_info_section.shape[0],
        "player_table_height": player_table_section.shape[0],
        "width": w,
    }

    return match_info_section, player_table_section, metadata






COLUMN_BOUNDARIES = [
    ("col1_player_info", 20, 420),
    ("col2_dmg", 420, 580),
    ("col3_actual_damage", 580, 740),
    ("col4_knockdown", 740, 900),
    ("col5_heal", 900, 1060),
    ("col6_help_up", 1060, 1220),
    ("col7_revival", 1220, 1380),
    ("col8_headshot_rate", 1380, 1540),
]


def slice_player_table_columns(player_table: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    STEP 3: Column-wise Slicing of Player Table.
    Slices the 1560-wide player table into 8 distinct vertical column crops
    using user-calibrated column grid X-coordinates:
      [20, 420, 580, 740, 900, 1060, 1220, 1380, 1540]

    Returns:
      - column_crops: Dict mapping column key -> np.ndarray image
      - annotated_grid: player_table with cyan vertical grid lines and labels
      - metadata: list of column definitions and dimensions
    """
    h, w = player_table.shape[:2]
    column_crops = {}
    annotated = player_table.copy()

    col_meta = []
    for key, x_start, x_end in COLUMN_BOUNDARIES:
        # Ensure bounds stay within image dimensions
        xs = max(0, min(w, x_start))
        xe = max(0, min(w, x_end))
        col_crop = player_table[:, xs:xe]
        column_crops[key] = col_crop

        col_meta.append({
            "key": key,
            "x_start": xs,
            "x_end": xe,
            "width": xe - xs,
            "height": h
        })

        # Draw column boundary lines and labels on annotated image
        cv2.line(annotated, (xs, 0), (xs, h), (255, 255, 0), 2)
        cv2.putText(annotated, key, (xs + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Draw final right boundary
    last_x = COLUMN_BOUNDARIES[-1][2]
    cv2.line(annotated, (last_x, 0), (last_x, h), (255, 255, 0), 2)

    metadata = {
        "num_columns": len(COLUMN_BOUNDARIES),
        "columns": col_meta,
        "table_width": w,
        "table_height": h
    }

    return column_crops, annotated, metadata

