import os
import re
import time
import json
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import httpx

from app.ocr.preprocessor import preprocess_image
from app.ocr.smart_table_detector import (
    detect_smart_table_roi,
    detect_stats_button_crop,
    split_match_info_and_player_table,
    slice_player_table_columns,
)
from app.core.config import settings
from app.schemas.common import MatchPlayerStats

log = logging.getLogger("app.pipelines.five_step")

# ========== VL MODEL CONFIGURATION ==========
VL_MODEL = os.getenv("VL_MODEL", "qwen/qwen3.7-flash")
VL_API_URL = "https://openrouter.ai/api/v1/chat/completions"
VL_TEMPERATURE = 0.0
VL_MAX_TOKENS = 1500

MATCH_STATS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "map": {
            "type": "string",
            "description": "Map name: Bermuda, Purgatory, Kalahari, Nexterra, Solara, or Unknown",
        },
        "rank": {
            "type": "integer",
            "description": "Placement rank of team (1 if Booyah, or number from #X/Y)",
        },
        "total_teams": {
            "type": "integer",
            "description": "Total teams count in match (e.g. 11 or 12)",
        },
        "players": {
            "type": "array",
            "description": "List of active players in the team (up to 4 rows). Skip completely empty or absent rows.",
            "items": {
                "type": "object",
                "properties": {
                    "inGameName": {"type": "string", "description": "Exact in-game player nickname from Column 1"},
                    "kills": {"type": "integer", "description": "Kills (K from K/D/A in Column 1)"},
                    "deaths": {"type": "integer", "description": "Deaths (D from K/D/A in Column 1)"},
                    "assists": {"type": "integer", "description": "Assists (A from K/D/A in Column 1)"},
                    "damage": {"type": "integer", "description": "Damage dealt (DMG from Column 2)"},
                    "actualDamage": {"type": "integer", "description": "Actual damage (from Column 3)"},
                    "knockedDown": {"type": "integer", "description": "Knocked down count (from Column 4)"},
                    "heal": {"type": "integer", "description": "Heal amount (from Column 5)"},
                    "helpUp": {"type": "integer", "description": "Help up count (from Column 6)"},
                    "revival": {"type": "integer", "description": "Revival count (from Column 7)"},
                    "headShotRate": {"type": "number", "description": "Headshot rate percentage as float without % sign (e.g. 25.0 from Column 8)"},
                },
                "required": [
                    "inGameName",
                    "kills",
                    "deaths",
                    "assists",
                    "damage",
                    "actualDamage",
                    "knockedDown",
                    "heal",
                    "helpUp",
                    "revival",
                    "headShotRate",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["map", "rank", "total_teams", "players"],
    "additionalProperties": False,
}

VL_PROMPT = (
    "You are an expert Free Fire esports match statistics extractor.\n"
    "You are given the extracted sections from an end-of-match scoreboard:\n"
    "- Image 1: Match Header (shows Map name, Placement Rank like #5/11 or Booyah #1, and Total Teams).\n"
    "- Image 2: Column 1 - Player Info (shows Player In-Game Names and K/D/A 'Kills/Deaths/Assists' for each row).\n"
    "- Image 3: Column 2 - DMG (Total Damage dealt for each row).\n"
    "- Image 4: Column 3 - Actual Damage for each row.\n"
    "- Image 5: Column 4 - Knock Down count for each row.\n"
    "- Image 6: Column 5 - Heal amount for each row.\n"
    "- Image 7: Column 6 - Help Up count for each row.\n"
    "- Image 8: Column 7 - Revival count for each row.\n"
    "- Image 9: Column 8 - Headshot Rate percentage for each row.\n\n"
    "Instructions:\n"
    "1. Extract the map name (one of Bermuda, Purgatory, Kalahari, Nexterra, Solara, or Unknown).\n"
    "2. Extract the team placement rank and total teams. If 'Booyah' is visible, rank is 1. If format is '#X/Y', rank is X and total_teams is Y.\n"
    "3. For each active player row (from top to bottom, row 1 to up to 4):\n"
    "   - inGameName: The exact nickname visible in Column 1. Preserve characters and case.\n"
    "   - kills: K from K/D/A (first number).\n"
    "   - deaths: D from K/D/A (second number).\n"
    "   - assists: A from K/D/A (third number).\n"
    "   - damage: Integer from Column 2 (ignore progress bar, read digits).\n"
    "   - actualDamage: Integer from Column 3.\n"
    "   - knockedDown: Integer from Column 4.\n"
    "   - heal: Integer from Column 5.\n"
    "   - helpUp: Integer from Column 6.\n"
    "   - revival: Integer from Column 7.\n"
    "   - headShotRate: Float percentage from Column 8 without '%' symbol (e.g. 25.0).\n"
    "4. Only include active, visible player rows. Skip completely empty or absent player rows.\n"
    "5. Return strictly valid JSON adhering to the specified schema."
)
# ============================================


def _encode_png_b64(img: np.ndarray) -> str:
    """Encode an OpenCV BGR image as a base64 data URI."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def run_vl_stats_extraction(
    match_info: np.ndarray,
    column_crops: Dict[str, np.ndarray],
    api_key: Optional[str] = None,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Step 5: Sends the 8 column images and match header image directly to the VL model
    (qwen/qwen3.7-flash via OpenRouter) and retrieves structured output.
    """
    key = api_key or os.getenv("OPENROUTER_API_KEY", "") or getattr(settings, "OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is not configured in environment or settings")

    # Encode images
    header_uri = _encode_png_b64(match_info)
    col1_uri = _encode_png_b64(column_crops.get("col1_player_info", match_info))
    col2_uri = _encode_png_b64(column_crops.get("col2_dmg", match_info))
    col3_uri = _encode_png_b64(column_crops.get("col3_actual_damage", match_info))
    col4_uri = _encode_png_b64(column_crops.get("col4_knockdown", match_info))
    col5_uri = _encode_png_b64(column_crops.get("col5_heal", match_info))
    col6_uri = _encode_png_b64(column_crops.get("col6_help_up", match_info))
    col7_uri = _encode_png_b64(column_crops.get("col7_revival", match_info))
    col8_uri = _encode_png_b64(column_crops.get("col8_headshot_rate", match_info))

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": VL_PROMPT},
        {"type": "text", "text": "Image 1: Match Header (Map, Rank, Total Teams):"},
        {"type": "image_url", "image_url": {"url": header_uri}},
        {"type": "text", "text": "Image 2: Column 1 - Player Info (Names and K/D/A):"},
        {"type": "image_url", "image_url": {"url": col1_uri}},
        {"type": "text", "text": "Image 3: Column 2 - DMG (Total Damage):"},
        {"type": "image_url", "image_url": {"url": col2_uri}},
        {"type": "text", "text": "Image 4: Column 3 - Actual Damage:"},
        {"type": "image_url", "image_url": {"url": col3_uri}},
        {"type": "text", "text": "Image 5: Column 4 - Knock Down:"},
        {"type": "image_url", "image_url": {"url": col4_uri}},
        {"type": "text", "text": "Image 6: Column 5 - Heal:"},
        {"type": "image_url", "image_url": {"url": col5_uri}},
        {"type": "text", "text": "Image 7: Column 6 - Help Up:"},
        {"type": "image_url", "image_url": {"url": col6_uri}},
        {"type": "text", "text": "Image 8: Column 7 - Revival:"},
        {"type": "image_url", "image_url": {"url": col7_uri}},
        {"type": "text", "text": "Image 9: Column 8 - Headshot Rate (%):"},
        {"type": "image_url", "image_url": {"url": col8_uri}},
    ]

    body = {
        "model": VL_MODEL,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": VL_TEMPERATURE,
        "max_tokens": VL_MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "match_stats_extraction",
                "schema": MATCH_STATS_RESPONSE_SCHEMA,
                "strict": True,
            },
        },
    }

    last_error = None
    with httpx.Client(timeout=45.0) as client:
        for attempt in range(1, max_retries + 2):
            try:
                log.info("[Step 5] Calling VL model %s (attempt %d)...", VL_MODEL, attempt)
                res = client.post(
                    VL_API_URL,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )

                if res.status_code == 200:
                    resp_json = res.json()
                    raw_text = resp_json["choices"][0]["message"]["content"].strip()
                    log.info("[Step 5] VL model response received (%d chars)", len(raw_text))
                    parsed = json.loads(raw_text)
                    return parsed
                elif res.status_code in (429, 500, 502, 503, 504):
                    log.warning("[Step 5] Transient API status %d: %s. Retrying...", res.status_code, res.text[:200])
                    time.sleep(1.5 * attempt)
                    continue
                else:
                    raise ValueError(f"OpenRouter API returned HTTP {res.status_code}: {res.text[:300]}")
            except Exception as e:
                last_error = e
                log.warning("[Step 5] Attempt %d failed: %s", attempt, e)
                if attempt <= max_retries:
                    time.sleep(1.5 * attempt)

    raise RuntimeError(f"Step 5 VL model extraction failed after retries: {last_error}")


def run_five_step_pipeline(
    raw_bytes: bytes,
    easy_reader=None,
    api_key: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Runs the 5-step Free Fire match extraction pipeline:
      - Step 1: Normalization (1920w target)
      - Step 2: Smart Table ROI Detection (Diagonal Anchors, 1560x1040)
      - Step 3: STATS button removal & Section split (Match Info & Player Table)
      - Step 4: Column Grid Slicing (Slices into the 8 stat columns)
      - Step 5: VL Model Multi-Column Extraction (qwen/qwen3.7-flash structured output)
    """
    t_start = time.time()
    metadata: Dict[str, Any] = {}
    debug_steps: List[Dict[str, Any]] = []

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
            "image": _encode_png_b64(normalized_img),
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
            "image": _encode_png_b64(table_crop),
        })

    # ------------------------------------------------------------
    # STEP 3: STATS button removal + section splitting
    # ------------------------------------------------------------
    step3_t = time.time()
    tighter_crop, meta3a = detect_stats_button_crop(table_crop)
    metadata["stats_button"] = meta3a

    match_info, player_table, meta3b = split_match_info_and_player_table(
        tighter_crop, use_ocr=False
    )
    metadata["split"] = meta3b
    step3_dur = round(time.time() - step3_t, 3)

    if debug:
        debug_steps.append({
            "step": 3,
            "name": "STATS Button Remove + Section Split",
            "duration_seconds": step3_dur,
            "stats_button_meta": meta3a,
            "split_meta": meta3b,
            "images": {
                "tighter_crop": _encode_png_b64(tighter_crop),
                "match_info": _encode_png_b64(match_info),
                "player_table": _encode_png_b64(player_table),
            },
        })

    # ------------------------------------------------------------
    # STEP 4: Column grid identification & slicing (8 columns)
    # ------------------------------------------------------------
    step4_t = time.time()
    column_crops, annotated_grid, meta4 = slice_player_table_columns(player_table)
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
            if key in column_crops:
                col_images[key] = _encode_png_b64(column_crops[key])
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
    # STEP 5: VL Model Multi-Column Structured Extraction
    # ------------------------------------------------------------
    step5_t = time.time()
    vl_result = run_vl_stats_extraction(
        match_info=match_info,
        column_crops=column_crops,
        api_key=api_key,
    )
    step5_dur = round(time.time() - step5_t, 3)

    raw_players = vl_result.get("players", [])
    structured_players: List[MatchPlayerStats] = []
    for p in raw_players:
        structured_players.append(
            MatchPlayerStats(
                inGameName=p.get("inGameName", "Unknown"),
                kills=int(p.get("kills", 0)),
                deaths=int(p.get("deaths", 0)),
                assists=int(p.get("assists", 0)),
                damage=int(p.get("damage", 0)),
                actualDamage=int(p.get("actualDamage", 0)),
                knockedDown=int(p.get("knockedDown", 0)),
                heal=int(p.get("heal", 0)),
                helpUp=int(p.get("helpUp", 0)),
                revival=int(p.get("revival", 0)),
                headShotRate=float(p.get("headShotRate", 0.0)),
            )
        )

    map_name = vl_result.get("map", "Unknown")
    rank = int(vl_result.get("rank", 1))
    total_teams = int(vl_result.get("total_teams", 12))

    if debug:
        debug_steps.append({
            "step": 5,
            "name": "VL Model Multi-Column Extraction (Qwen 3.7 Flash)",
            "duration_seconds": step5_dur,
            "model": VL_MODEL,
            "num_players": len(structured_players),
            "vl_result": vl_result,
        })

    elapsed = time.time() - t_start
    metadata["elapsed_seconds"] = round(elapsed, 3)
    metadata["model"] = VL_MODEL
    metadata["step_durations"] = {
        "step1_normalization": step1_dur,
        "step2_smart_roi": step2_dur,
        "step3_section_split": step3_dur,
        "step4_column_slice": step4_dur,
        "step5_vl_extraction": step5_dur,
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
        "5-step VL pipeline completed in %.2fs: map=%s rank=%d teams=%d players=%d",
        elapsed, map_name, result["rank"], result["total_teams"], len(structured_players)
    )

    return result
