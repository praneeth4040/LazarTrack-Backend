import os
import cv2
import json
import time
import glob
import re
import numpy as np
import pytesseract
from app.ocr.preprocessor import preprocess_image
from app.ocr.smart_table_detector import (
    detect_smart_table_roi,
    detect_stats_button_crop,
    split_match_info_and_player_table,
    slice_player_table_columns,
)
from app.ocr.local_easyocr import get_easyocr_reader
from app.ocr.paddle_ocr import run_paddle_ocr

IMG_DIR = os.path.join(os.path.dirname(__file__), "img")
OUTPUT_BASE_DIR = os.path.join(IMG_DIR, "output")


def parse_player_cell_ocr(paddle_items: list, easy_items: list, col_key: str):
    """
    Parses OCR results for a specific cell, picking the best text from PaddleOCR or EasyOCR.
    """
    p_texts = [item["text"].strip() for item in paddle_items if item.get("text")]
    e_texts = [item["text"].strip() for item in easy_items if item.get("text")]

    # Prefer PaddleOCR as primary
    primary_text = p_texts[0] if p_texts else (e_texts[0] if e_texts else "0")
    return primary_text


def process_image_pipeline(image_path: str, output_dir: str, easy_reader) -> dict:
    """
    Executes the complete 5-step pipeline on a single image file
    and saves step-by-step outputs into output_dir.
    """
    image_name = os.path.basename(image_path)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"PROCESSING: {image_name}")
    print(f"OUTPUT DIR: {output_dir}")
    print("=" * 70)

    # ------------------------------------------------------------
    # STEP 1: Normalization
    # ------------------------------------------------------------
    print("\n--- STEP 1: Normalization ---")
    with open(image_path, "rb") as f:
        raw_bytes = f.read()

    norm_bytes = preprocess_image(raw_bytes, target_width=1920)
    nparr = np.frombuffer(norm_bytes, np.uint8)
    normalized_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    step1_path = os.path.join(output_dir, "step1_normalization.png")
    cv2.imwrite(step1_path, normalized_img)
    print(f"  [OK] Saved -> {step1_path} ({normalized_img.shape[1]}x{normalized_img.shape[0]})")

    # ------------------------------------------------------------
    # STEP 2: Table layout detection
    # ------------------------------------------------------------
    print("\n--- STEP 2: Table Layout Detection ---")
    table_crop, annotated_crop, meta2 = detect_smart_table_roi(normalized_img, target_w=1920, target_h=1080)

    step2_path = os.path.join(output_dir, "step2_table_layout.png")
    cv2.imwrite(step2_path, table_crop)
    print(f"  [OK] Saved -> {step2_path} ({meta2['crop_width']}x{meta2['crop_height']})")

    # ------------------------------------------------------------
    # STEP 3: STATS button removal & section splitting
    # ------------------------------------------------------------
    print("\n--- STEP 3: STATS Button Removal & Section Splitting ---")
    tighter_crop, meta3a = detect_stats_button_crop(table_crop)
    step3_tighter_path = os.path.join(output_dir, "step3_tighter_table_crop.png")
    cv2.imwrite(step3_tighter_path, tighter_crop)
    print(f"  [OK] Saved -> {step3_tighter_path} ({meta3a['tighter_crop_width']}x{meta3a['tighter_crop_height']})")

    match_info, player_table, meta3b = split_match_info_and_player_table(tighter_crop, reader=easy_reader)
    step3_info_path = os.path.join(output_dir, "step3_match_info.png")
    step3_table_path = os.path.join(output_dir, "step3_player_table.png")
    cv2.imwrite(step3_info_path, match_info)
    cv2.imwrite(step3_table_path, player_table)
    print(f"  [OK] Detected Rank:    {meta3b.get('detected_rank')}")
    print(f"  [OK] Detected Tagline: {meta3b.get('detected_tagline')}")
    print(f"  [OK] Match info saved -> {step3_info_path} ({meta3b['width']}x{meta3b['match_info_height']})")
    print(f"  [OK] Player table saved -> {step3_table_path} ({meta3b['width']}x{meta3b['player_table_height']})")

    # ------------------------------------------------------------
    # STEP 4: Identifying column grids and cutting them
    # ------------------------------------------------------------
    print("\n--- STEP 4: Column Grid Identification & Slicing ---")
    column_crops, grid_annotated, meta4 = slice_player_table_columns(player_table)

    step4_grid_path = os.path.join(output_dir, "step4_column_grid.png")
    cv2.imwrite(step4_grid_path, grid_annotated)

    cols_dir = os.path.join(output_dir, "step4_columns")
    os.makedirs(cols_dir, exist_ok=True)

    for col_info in meta4["columns"]:
        key = col_info["key"]
        col_img = column_crops[key]
        col_img_path = os.path.join(cols_dir, f"{key}.png")
        cv2.imwrite(col_img_path, col_img)

    print(f"  [OK] Column crops saved to -> {cols_dir} (8 columns)")

    # ------------------------------------------------------------
    # STEP 5: High-Accuracy Dynamic Cell OCR (Columns 1 to 8)
    # ------------------------------------------------------------
    print("\n--- STEP 5: High-Accuracy Dynamic Cell OCR (Columns 1 to 8) ---")
    t0 = time.time()
    ocr_results = {
        "metadata": {
            "source_image": image_name,
            "detected_rank": meta3b.get("detected_rank"),
            "detected_tagline": meta3b.get("detected_tagline"),
            "split_y": meta3b.get("split_y"),
        }
    }

    # 5a: Match Info OCR
    match_info_paddle = run_paddle_ocr(match_info)
    ann_match_info = match_info.copy()
    for item in match_info_paddle:
        pts = np.array(item["bbox"], dtype=np.int32)
        cv2.polylines(ann_match_info, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(ann_match_info, item["text"], (int(pts[0][0]), int(pts[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    step5_ann_info_path = os.path.join(output_dir, "step5_annotated_match_info.png")
    cv2.imwrite(step5_ann_info_path, ann_match_info)
    ocr_results["match_info"] = match_info_paddle

    # 5b: Dynamic Player Slot Detection & Super-Scaled Cell Cropping
    player_cells_y = [
        ("Player 1", 165, 225),
        ("Player 2", 260, 320),
        ("Player 3", 355, 415),
        ("Player 4", 450, 510),
    ]

    def parse_kda_smart(raw_str: str) -> str:
        if not raw_str:
            return "0/0/0"
        s = re.sub(r'[^0-9/]', '', str(raw_str))
        parts = [p for p in s.split('/') if p]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return f"{parts[0]}/{parts[1]}/{parts[2]}"
        digits = re.sub(r'[^0-9]', '', s)
        if len(digits) == 3:
            return f"{digits[0]}/{digits[1]}/{digits[2]}"
        elif len(digits) == 4 and len(parts) == 2:
            if len(parts[0]) == 2:
                return f"{parts[0][0]}/{parts[0][1]}/{parts[1]}"
            elif len(parts[0]) == 3:
                return f"{parts[0][0]}/{parts[0][1]}/{parts[0][2]}"
        elif len(digits) == 2:
            return f"{digits[0]}/{digits[1]}/0"
        elif len(digits) == 1:
            return f"{digits[0]}/0/0"
        return "0/0/0"

    def force_digits_only(raw_str: str) -> str:
        cleaned = re.sub(r'[^0-9]', '', str(raw_str))
        return cleaned if cleaned else "0"

    def force_headshot_rate(raw_str: str) -> str:
        cleaned = re.sub(r'[^0-9.]', '', str(raw_str))
        if not cleaned or cleaned == '.':
            return "0.00%"
        if not cleaned.endswith('%'):
            cleaned += '%'
        return cleaned

    structured_players = []
    ann_cols_dir = os.path.join(output_dir, "step5_columns_annotated")
    os.makedirs(ann_cols_dir, exist_ok=True)
    ann_player_table = player_table.copy()

    col_ann_imgs = {col_info["key"]: column_crops[col_info["key"]].copy() for col_info in meta4["columns"]}

    for p_name, y1, y2 in player_cells_y:
        # Step 1: Check if this player slot is valid or empty (e.g. 3-player team)
        c1_crop_raw = player_table[y1:y2+30, 20:420]
        gray_c1 = cv2.cvtColor(c1_crop_raw, cv2.COLOR_BGR2GRAY)
        std_val = np.std(gray_c1)
        
        scaled_c1 = cv2.resize(c1_crop_raw, (c1_crop_raw.shape[1]*4, c1_crop_raw.shape[0]*4), interpolation=cv2.INTER_LANCZOS4)
        c1_ocr_res = easy_reader.readtext(scaled_c1, detail=1)

        # Empty slot criteria: no OCR text in col1 or uniform flat dark background
        if not c1_ocr_res or std_val < 20.0:
            print(f"  [INFO] {p_name} is an EMPTY slot (e.g. 3-player team). Skipping row.")
            continue

        p_dict = {"player": p_name, "stats": {}}

        # Dual-Region Processing for Column 1
        h_c1 = c1_crop_raw.shape[0]
        name_crop = c1_crop_raw[0:int(h_c1*0.55), :]
        kda_crop = c1_crop_raw[int(h_c1*0.48):, :]

        # 1. Player Name
        scaled_name = cv2.resize(name_crop, (name_crop.shape[1]*4, name_crop.shape[0]*4), interpolation=cv2.INTER_LANCZOS4)
        e_names = easy_reader.readtext(scaled_name, detail=0)
        p_dict["name"] = " ".join(e_names) if e_names else "Unknown"

        # 2. K/D/A Ratio with 6x Lanczos and strict 0123456789/ allowlist
        scaled_kda = cv2.resize(kda_crop, (kda_crop.shape[1]*6, kda_crop.shape[0]*6), interpolation=cv2.INTER_LANCZOS4)
        e_kda = easy_reader.readtext(scaled_kda, allowlist="0123456789/", detail=0)
        raw_kda = e_kda[0] if e_kda else ""
        if not raw_kda:
            tess_kda = pytesseract.image_to_string(scaled_kda, config="--psm 6 -c tessedit_char_whitelist=0123456789/").strip()
            raw_kda = tess_kda
        p_dict["kda"] = parse_kda_smart(raw_kda)

        # Draw exact OCR text bounding boxes for Column 1
        for bbox, text, conf in c1_ocr_res:
            pts_1x = [[int(pt[0]/4) + 20, int(pt[1]/4) + y1] for pt in bbox]
            pts_np = np.array(pts_1x, dtype=np.int32)
            cv2.polylines(ann_player_table, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(ann_player_table, text, (pts_1x[0][0], pts_1x[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            pts_c1 = [[int(pt[0]/4), int(pt[1]/4) + y1] for pt in bbox]
            cv2.polylines(col_ann_imgs["col1_player_info"], [np.array(pts_c1, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Process Columns 2 to 8 for valid player row
        for col_info in meta4["columns"]:
            key = col_info["key"]
            if key == "col1_player_info":
                continue

            xs, xe = col_info["x_start"], col_info["x_end"]
            w_cell = xe - xs
            cv2.line(ann_player_table, (xs, 0), (xs, meta4["table_height"]), (255, 255, 0), 1)

            if key == "col8_headshot_rate":
                cell_crop = player_table[y1+5:y2+25, xs:xe]
                scaled = cv2.resize(cell_crop, (cell_crop.shape[1]*5, cell_crop.shape[0]*5), interpolation=cv2.INTER_LANCZOS4)
                e_res = easy_reader.readtext(scaled, allowlist="0123456789.%", detail=0)
                val = e_res[0] if e_res else "0.00%"
                final_val = force_headshot_rate(val)
                p_dict["stats"][key] = final_val

                # Annotate box directly on headshot rate text
                cv2.rectangle(col_ann_imgs[key], (10, y1+5), (w_cell-10, y2+20), (0, 255, 0), 2)
                cv2.putText(col_ann_imgs[key], final_val, (15, y1+22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                cv2.rectangle(ann_player_table, (xs+10, y1+5), (xe-10, y2+20), (0, 255, 0), 2)
                cv2.putText(ann_player_table, final_val, (xs+15, y1+22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            else:
                # Columns 2..7: Strictly Left 48% of cell width (excl. progress bar)
                crop_w = int(w_cell * 0.48)
                cell_crop = player_table[y1:y2-10, xs:xs+crop_w]
                
                scaled_6x = cv2.resize(cell_crop, (cell_crop.shape[1]*6, cell_crop.shape[0]*6), interpolation=cv2.INTER_LANCZOS4)
                e_res = easy_reader.readtext(scaled_6x, allowlist="0123456789", detail=0)
                
                if e_res and e_res[0].strip():
                    val = e_res[0].strip()
                else:
                    tess_res = pytesseract.image_to_string(scaled_6x, config="--psm 6 -c tessedit_char_whitelist=0123456789").strip()
                    val = tess_res if tess_res else "0"

                final_val = force_digits_only(val)
                p_dict["stats"][key] = final_val

                # Annotate box directly on the left 48% main stat region (NEVER on right progress bar!)
                cv2.rectangle(col_ann_imgs[key], (5, y1), (crop_w, y2-10), (0, 255, 0), 2)
                cv2.putText(col_ann_imgs[key], final_val, (8, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                cv2.rectangle(ann_player_table, (xs+5, y1), (xs+crop_w, y2-10), (0, 255, 0), 2)
                cv2.putText(ann_player_table, final_val, (xs+8, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        structured_players.append(p_dict)

    # Save column visualizations
    for col_info in meta4["columns"]:
        key = col_info["key"]
        cv2.imwrite(os.path.join(ann_cols_dir, f"{key}_annotated.png"), col_ann_imgs[key])

    ocr_results["players"] = structured_players
    elapsed = time.time() - t0

    step5_ann_table_path = os.path.join(output_dir, "step5_annotated_player_table.png")
    cv2.imwrite(step5_ann_table_path, ann_player_table)

    ann_tighter_crop = np.vstack([ann_match_info, ann_player_table])
    step5_ann_full_path = os.path.join(output_dir, "step5_annotated_tighter_crop.png")
    cv2.imwrite(step5_ann_full_path, ann_tighter_crop)
    step5_ann_path = os.path.join(output_dir, "step5_annotated.png")
    cv2.imwrite(step5_ann_path, ann_tighter_crop)

    print(f"  [OK] OCR completed in {elapsed:.2f}s using Dynamic Cell OCR ({len(structured_players)} players)")
    print("\n  Extracted Players Table:")
    for p in structured_players:
        s = p.get("stats", {})
        print(f"    - {p.get('name', 'N/A'):<16} | KDA: {p.get('kda', '0/0/0'):<7} | Dmg: {s.get('col2_dmg', '0'):<5} | ActDmg: {s.get('col3_actual_damage', '0'):<5} | Knock: {s.get('col4_knockdown', '0'):<2} | Heal: {s.get('col5_heal', '0'):<4} | Help: {s.get('col6_help_up', '0'):<2} | Rev: {s.get('col7_revival', '0'):<2} | Headshot: {s.get('col8_headshot_rate', '0%')}")

    step5_json_path = os.path.join(output_dir, "step5_ocr_results.json")
    with open(step5_json_path, "w") as f:
        json.dump(ocr_results, f, indent=2, default=str)
    print(f"\n  [OK] Saved OCR results -> {step5_json_path}")

    return ocr_results


def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    image_patterns = [
        os.path.join(IMG_DIR, "image*.png"),
        os.path.join(IMG_DIR, "image*.jpg"),
        os.path.join(IMG_DIR, "sample*.png"),
    ]

    image_files = []
    for pat in image_patterns:
        image_files.extend(glob.glob(pat))

    image_files = sorted(list(set(image_files)))

    if not image_files:
        fallback = os.path.join(IMG_DIR, "image.png")
        if os.path.exists(fallback):
            image_files = [fallback]
        else:
            print(f"Error: No target images found in {IMG_DIR}")
            return

    print("=" * 70)
    print(f"FOUND {len(image_files)} IMAGES TO PROCESS IN BATCH:")
    for img_f in image_files:
        print(f"  - {os.path.basename(img_f)}")
    print("=" * 70)
    print()

    easy_reader = get_easyocr_reader()
    batch_summary = {}

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]
        out_sub_dir = os.path.join(OUTPUT_BASE_DIR, stem)

        try:
            res = process_image_pipeline(img_path, out_sub_dir, easy_reader)
            batch_summary[stem] = {
                "source_image": img_name,
                "output_directory": out_sub_dir,
                "status": "SUCCESS",
                "rank": res["metadata"].get("detected_rank"),
                "tagline": res["metadata"].get("detected_tagline"),
                "players_count": len(res.get("players", [])),
                "players": res.get("players", [])
            }
        except Exception as e:
            print(f"  [ERROR] Failed to process {img_name}: {e}")
            batch_summary[stem] = {
                "source_image": img_name,
                "status": "FAILED",
                "error": str(e)
            }
        print()

    summary_path = os.path.join(OUTPUT_BASE_DIR, "batch_ocr_summary.json")
    with open(summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2, default=str)

    print("=" * 70)
    print("ALL IMAGES PROCESSED SUCCESSFULLY!")
    print(f"Summary JSON saved -> {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
