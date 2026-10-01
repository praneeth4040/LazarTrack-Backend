import re
import time
import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Optional, Tuple

from app.ocr.preprocessor import preprocess_image
from app.ocr.smart_table_detector import (
    detect_smart_table_roi,
    detect_stats_button_crop,
    split_match_info_and_player_table,
    slice_player_table_columns,
)
from app.ocr.local_easyocr import get_easyocr_reader
from app.schemas.common import MatchPlayerStats

log = logging.getLogger("app.pipelines.five_step")

_KNOWN_MAPS = [
    "Bermuda",
    "Purgatory", "Kalahari", "Nexterra", "Solara"
]

PLAYER_CELLS_Y = [
    ("Player 1", 165, 225),
    ("Player 2", 260, 320),
    ("Player 3", 355, 415),
    ("Player 4", 450, 510),
]


def _parse_kda_smart(raw_str: str) -> Tuple[int, int, int]:
    if not raw_str:
        return 0, 0, 0
    s = re.sub(r'[^0-9/]', '', str(raw_str))
    parts = [p for p in s.split('/') if p]
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return int(parts[0]), int(parts[1]), int(parts[2])
    digits = re.sub(r'[^0-9]', '', s)
    if len(digits) == 3:
        return int(digits[0]), int(digits[1]), int(digits[2])
    elif len(digits) == 4 and len(parts) == 2:
        if len(parts[0]) == 2:
            return int(parts[0][0]), int(parts[0][1]), int(parts[1])
        elif len(parts[0]) == 3:
            return int(parts[0][0]), int(parts[0][1]), int(parts[0][2])
    elif len(digits) == 2:
        return int(digits[0]), int(digits[1]), 0
    elif len(digits) == 1:
        return int(digits[0]), 0, 0
    return 0, 0, 0


def _force_digits(raw_str: str) -> int:
    cleaned = re.sub(r'[^0-9]', '', str(raw_str))
    return int(cleaned) if cleaned else 0


def _force_headshot_rate(raw_str: str) -> float:
    cleaned = re.sub(r'[^0-9.]', '', str(raw_str))
    if not cleaned or cleaned == '.':
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _extract_rank(rank_str: Optional[str]) -> int:
    if not rank_str:
        return 0
    m = re.search(r'#\s*(\d+)', rank_str)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*/\s*\d+', rank_str)
    if m:
        return int(m.group(1))
    m = re.search(r'(?:rank|place)\s*[:\-]?\s*(\d+)', rank_str, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 0


def _extract_total_teams(rank_str: Optional[str]) -> int:
    if not rank_str:
        return 12
    m = re.search(r'/\s*(\d+)', rank_str)
    if m:
        return int(m.group(1))
    return 12


def run_five_step_pipeline(
    raw_bytes: bytes,
    easy_reader=None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Runs the full 5-step Free Fire match extraction pipeline on raw image bytes.

    Args:
        raw_bytes: Uploaded screenshot bytes.
        easy_reader: Optional pre-instantiated EasyOCR reader (for reuse).
        debug: If True, the returned dict also contains a "steps" key with
               per-step results (crop images encoded as base64 data URIs,
               OCR texts, metadata, per-player per-cell OCR debug rows).

    Returns:
        Dict with keys: map, rank, total_teams, players, metadata.
        When debug=True, additionally contains: "steps" (list of step dicts).
    """
    t_start = time.time()

    if easy_reader is None:
        easy_reader = get_easyocr_reader()

    metadata: Dict[str, Any] = {}
    debug_steps: List[Dict[str, Any]] = []

    def _encode_png(img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return ""
        import base64
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    # ------------------------------------------------------------
    # STEP 1: Normalization (1920w target)
    # ------------------------------------------------------------
    step1_t = time.time()
    norm_bytes = preprocess_image(raw_bytes, target_width=1920)
    nparr = np.frombuffer(norm_bytes, np.uint8)
    normalized_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if normalized_img is None:
        raise ValueError("Could not decode normalized image bytes")
    step1_dur = round(time.time() - step1_t, 3)

    if debug:
        h, w = normalized_img.shape[:2]
        debug_steps.append({
            "step": 1,
            "name": "Normalization",
            "duration_seconds": step1_dur,
            "width": w,
            "height": h,
            "target_width": 1920,
            "image": _encode_png(normalized_img),
        })

    # ------------------------------------------------------------
    # STEP 2: Smart Table ROI Detection
    # ------------------------------------------------------------
    step2_t = time.time()
    table_crop, _, meta2 = detect_smart_table_roi(
        normalized_img, target_w=1920, target_h=1080
    )
    metadata["roi"] = meta2
    step2_dur = round(time.time() - step2_t, 3)

    if debug:
        h, w = table_crop.shape[:2]
        debug_steps.append({
            "step": 2,
            "name": "Smart Table ROI Detection",
            "duration_seconds": step2_dur,
            "width": w,
            "height": h,
            "meta": meta2,
            "image": _encode_png(table_crop),
        })

    # ------------------------------------------------------------
    # STEP 3: STATS button removal + section splitting
    # ------------------------------------------------------------
    step3_t = time.time()
    tighter_crop, meta3a = detect_stats_button_crop(table_crop)
    metadata["stats_button"] = meta3a

    match_info, player_table, meta3b = split_match_info_and_player_table(
        tighter_crop, reader=easy_reader
    )
    detected_rank = meta3b.get("detected_rank")
    detected_tagline = meta3b.get("detected_tagline")
    metadata["split"] = meta3b
    step3_dur = round(time.time() - step3_t, 3)

    if debug:
        debug_steps.append({
            "step": 3,
            "name": "STATS Button Remove + Section Split",
            "duration_seconds": step3_dur,
            "stats_button_meta": meta3a,
            "split_meta": meta3b,
            "detected_rank": detected_rank,
            "detected_tagline": detected_tagline,
            "images": {
                "tighter_crop": _encode_png(tighter_crop),
                "match_info": _encode_png(match_info),
                "player_table": _encode_png(player_table),
            },
        })

    # ------------------------------------------------------------
    # Extract map from match info (lightweight EasyOCR)
    # ------------------------------------------------------------
    map_name = "Unknown"
    map_ocr_texts: List[str] = []
    try:
        map_ocr_texts = easy_reader.readtext(match_info, detail=0)
        joined = " ".join(map_ocr_texts)
        for name in _KNOWN_MAPS:
            if re.search(re.escape(name), joined, re.IGNORECASE):
                map_name = name
                break
    except Exception as e:
        log.warning("Map detection OCR failed: %s", e)

    # Rank + total_teams
    if detected_tagline:
        joined_tag = str(detected_tagline).lower()
        if "booyah" in joined_tag:
            rank = 1
        else:
            rank = _extract_rank(detected_rank)
    else:
        rank = _extract_rank(detected_rank)

    total_teams = _extract_total_teams(detected_rank)

    # Also check if any match_texts contain "booyah" -> rank=1
    if rank == 0:
        for t in map_ocr_texts:
            if re.search(r"booyah", t, re.IGNORECASE):
                rank = 1
                break
        if rank == 0:
            joined = " ".join(map_ocr_texts)
            m = re.search(r"#\s*(\d+)", joined)
            if m:
                rank = int(m.group(1))
                m2 = re.search(r"/\s*(\d+)", joined)
                if m2:
                    total_teams = int(m2.group(1))

    # ------------------------------------------------------------
    # STEP 4: Column grid identification & slicing
    # ------------------------------------------------------------
    step4_t = time.time()
    column_crops, _, meta4 = slice_player_table_columns(player_table)
    metadata["columns"] = {
        "num_columns": meta4["num_columns"],
        "width": meta4["table_width"],
        "height": meta4["table_height"],
    }
    step4_dur = round(time.time() - step4_t, 3)

    if debug:
        col_images: Dict[str, str] = {}
        for c in meta4.get("columns", []):
            key = c["key"]
            crop = player_table[:, c["x_start"]:c["x_end"]]
            col_images[key] = _encode_png(crop)
        debug_steps.append({
            "step": 4,
            "name": "Column Grid Slicing",
            "duration_seconds": step4_dur,
            "num_columns": meta4["num_columns"],
            "table_width": meta4["table_width"],
            "table_height": meta4["table_height"],
            "column_defs": meta4.get("columns", []),
            "column_images": col_images,
        })

    # ------------------------------------------------------------
    # STEP 5: Per-player per-cell super-scale OCR
    # ------------------------------------------------------------
    step5_t = time.time()
    structured_players: List[MatchPlayerStats] = []
    col_defs = {c["key"]: (c["x_start"], c["x_end"]) for c in meta4["columns"]}

    debug_players: List[Dict[str, Any]] = []

    for p_label, y1, y2 in PLAYER_CELLS_Y:
        p_debug: Dict[str, Any] = {
            "label": p_label,
            "y_start": y1,
            "y_end": y2,
            "cells": {},
        }

        # Empty slot detection using column 1
        c1_xs, c1_xe = col_defs["col1_player_info"]
        c1_crop_raw = player_table[y1 : y2 + 30, c1_xs:c1_xe]
        try:
            gray_c1 = cv2.cvtColor(c1_crop_raw, cv2.COLOR_BGR2GRAY)
            std_val = float(np.std(gray_c1))
        except Exception:
            continue
        scaled_c1 = cv2.resize(
            c1_crop_raw,
            (c1_crop_raw.shape[1] * 4, c1_crop_raw.shape[0] * 4),
            interpolation=cv2.INTER_LANCZOS4,
        )
        c1_ocr_res = easy_reader.readtext(scaled_c1, detail=1)

        p_debug["preflight_std"] = std_val
        p_debug["preflight_ocr_items"] = len(c1_ocr_res)

        if not c1_ocr_res or std_val < 20.0:
            log.info("Slot %s is empty (std=%.1f, ocr_items=%d). Skipping.",
                     p_label, std_val, len(c1_ocr_res))
            p_debug["empty"] = True
            if debug:
                debug_players.append(p_debug)
            continue

        p_debug["empty"] = False

        # Dual-region col 1: name (top 55%) + KDA (bottom)
        h_c1 = c1_crop_raw.shape[0]
        name_crop = c1_crop_raw[0 : int(h_c1 * 0.55), :]
        kda_crop = c1_crop_raw[int(h_c1 * 0.48) :, :]

        # --- Name ---
        scaled_name = cv2.resize(
            name_crop,
            (name_crop.shape[1] * 4, name_crop.shape[0] * 4),
            interpolation=cv2.INTER_LANCZOS4,
        )
        e_names = easy_reader.readtext(scaled_name, detail=0)
        full_ign = " ".join(e_names).strip() if e_names else "Unknown"
        if not full_ign:
            full_ign = "Unknown"
        if debug:
            p_debug["cells"]["col1_name"] = {
                "raw_crop": _encode_png(name_crop),
                "scaled_4x": _encode_png(scaled_name),
                "ocr_raw": e_names,
                "final": full_ign,
            }

        # --- K/D/A Ensemble OCR ---
        from app.ocr.preprocessor import generate_preprocessing_variants, get_interpolation_methods
        kda_variants = generate_preprocessing_variants(kda_crop)
        interpolations = get_interpolation_methods()

        kda_candidates = []
        # Try primary contrast/sharpened variants with Lanczos & Cubic interpolation at 4x-5x scale
        for var_name in ["contrast", "sharpened", "thresholded", "raw"]:
            img_var = kda_variants.get(var_name, kda_crop)
            for scale_factor in [4, 5]:
                for interp_name in ["lanczos", "cubic"]:
                    interp_flag = interpolations[interp_name]
                    scaled_kda = cv2.resize(
                        img_var,
                        (img_var.shape[1] * scale_factor, img_var.shape[0] * scale_factor),
                        interpolation=interp_flag,
                    )
                    e_res = easy_reader.readtext(scaled_kda, allowlist="0123456789/", detail=0)
                    if e_res:
                        kda_candidates.append(e_res[0])

        raw_kda = kda_candidates[0] if kda_candidates else ""
        # Find candidate with 2 slashes or matching 3 numbers
        for cand in kda_candidates:
            if cand.count('/') == 2 or len(re.sub(r'[^0-9]', '', cand)) == 3:
                raw_kda = cand
                break

        kills, _death, assists = _parse_kda_smart(raw_kda)
        if debug:
            p_debug["cells"]["col1_kda"] = {
                "raw_crop": _encode_png(kda_crop),
                "scaled_sample": _encode_png(cv2.resize(kda_crop, (kda_crop.shape[1] * 4, kda_crop.shape[0] * 4))),
                "candidates": kda_candidates,
                "parsed_kda": [kills, _death, assists],
            }

        # --- Col 2..8 stat cells ---
        col2_damage = 0
        col3_actual_damage = 0
        col4_knockdown = 0
        col5_heal = 0
        col6_help_up = 0
        col7_revival = 0
        col8_hsr = 0.0

        for col_info in meta4["columns"]:
            key = col_info["key"]
            if key == "col1_player_info":
                continue
            xs, xe = col_info["x_start"], col_info["x_end"]
            w_cell = xe - xs

            if key == "col8_headshot_rate":
                cell_crop = player_table[y1 + 5 : y2 + 25, xs:xe]
                cell_debug: Dict[str, Any] = {"raw_crop": _encode_png(cell_crop) if debug else ""}
                try:
                    scaled = cv2.resize(
                        cell_crop,
                        (cell_crop.shape[1] * 5, cell_crop.shape[0] * 5),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    if debug:
                        cell_debug["scaled_5x"] = _encode_png(scaled)
                    e_res = easy_reader.readtext(
                        scaled, allowlist="0123456789.%", detail=0
                    )
                    cell_debug["ocr_raw"] = e_res
                    if e_res:
                        col8_hsr = _force_headshot_rate(e_res[0])
                    cell_debug["final"] = col8_hsr
                except Exception as ex:
                    cell_debug["error"] = str(ex)
                if debug:
                    p_debug["cells"][key] = cell_debug
            else:
                # Columns 2..7: left 48% only (skip progress bars)
                crop_w = int(w_cell * 0.48)
                cell_crop = player_table[y1 : y2 - 10, xs : xs + crop_w]
                cell_debug = {"raw_crop": _encode_png(cell_crop) if debug else "", "left_48_only": True}
                try:
                    scaled_6x = cv2.resize(
                        cell_crop,
                        (cell_crop.shape[1] * 6, cell_crop.shape[0] * 6),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    if debug:
                        cell_debug["scaled_6x"] = _encode_png(scaled_6x)
                    e_res = easy_reader.readtext(
                        scaled_6x, allowlist="0123456789", detail=0
                    )
                    cell_debug["ocr_raw"] = e_res
                    val_str = e_res[0].strip() if (e_res and e_res[0].strip()) else "0"
                    val = _force_digits(val_str)
                    cell_debug["final"] = val
                    if key == "col2_dmg":
                        col2_damage = val
                    elif key == "col3_actual_damage":
                        col3_actual_damage = val
                    elif key == "col4_knockdown":
                        col4_knockdown = val
                    elif key == "col5_heal":
                        col5_heal = val
                    elif key == "col6_help_up":
                        col6_help_up = val
                    elif key == "col7_revival":
                        col7_revival = val
                except Exception as ex:
                    cell_debug["error"] = str(ex)
                if debug:
                    p_debug["cells"][key] = cell_debug

        structured_players.append(MatchPlayerStats(
            inGameName=full_ign,
            kills=kills,
            deaths=_death,
            assists=assists,
            damage=col2_damage,
            actualDamage=col3_actual_damage,
            revival=col7_revival,
            knockedDown=col4_knockdown,
            heal=col5_heal,
            helpUp=col6_help_up,
            headShotRate=col8_hsr,
        ))

        if debug:
            p_debug["final"] = structured_players[-1].model_dump()
            debug_players.append(p_debug)

    step5_dur = round(time.time() - step5_t, 3)
    if debug:
        debug_steps.append({
            "step": 5,
            "name": "Per-Cell Super-Scale OCR",
            "duration_seconds": step5_dur,
            "player_cells_y": PLAYER_CELLS_Y,
            "num_players": len(structured_players),
            "players": debug_players,
        })

    elapsed = time.time() - t_start
    metadata["elapsed_seconds"] = round(elapsed, 3)
    metadata["detected_rank"] = detected_rank
    metadata["detected_tagline"] = detected_tagline
    metadata["map_ocr_texts"] = map_ocr_texts
    metadata["step_durations"] = {
        "step1_normalization": step1_dur,
        "step2_smart_roi": step2_dur,
        "step3_section_split": step3_dur,
        "step4_column_slice": step4_dur,
        "step5_per_cell_ocr": step5_dur,
    }

    result: Dict[str, Any] = {
        "map": map_name,
        "rank": rank if rank >= 1 else 1,
        "total_teams": total_teams if total_teams >= 1 else 12,
        "players": [p.model_dump() for p in structured_players],
        "metadata": metadata,
    }

    if debug:
        result["steps"] = debug_steps

    log.info(
        "5-step pipeline completed in %.2fs: map=%s rank=%d teams=%d players=%d",
        elapsed, map_name, result["rank"], result["total_teams"], len(structured_players)
    )

    return result
