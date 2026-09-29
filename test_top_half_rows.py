import os
import cv2
import numpy as np
from app.ocr.preprocessor import preprocess_image
from app.ocr.smart_table_detector import detect_smart_table_roi
from app.ocr.local_easyocr import get_easyocr_reader

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "img", "image.png")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "img", "output")

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found.")
        return

    print("=" * 75)
    print("🚀 4 SQUAD PLAYERS MAIN DIGITS ONLY (ALL BARS & PERCENTAGES CUT OFF)")
    print("=" * 75)

    with open(IMAGE_PATH, "rb") as f:
        raw_bytes = f.read()

    norm_bytes = preprocess_image(raw_bytes, target_width=1920)
    nparr = np.frombuffer(norm_bytes, np.uint8)
    normalized_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    reader = get_easyocr_reader()

    # Exact Y-starts for the 4 squad player main digit lines (excluding side headings & bars):
    # Player 1 (WYZEX):      Y = 390..425
    # Player 2 (NZF NORMAL): Y = 450..485
    # Player 3 (RYG LUFFY):   Y = 510..545
    # Player 4 (OD'8 RAHUL):  Y = 570..605
    row_starts = [390, 450, 510, 570]
    row_height = 35  # Top 35px ONLY! Cuts off all bars & sub-percentages completely

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, y_start in enumerate(row_starts, 1):
        row_strip = normalized_img[y_start : y_start + row_height, 240:1750]

        strip_path = os.path.join(OUTPUT_DIR, f"squad_player_{idx}_main_digits_only.png")
        cv2.imwrite(strip_path, row_strip)

        ocr_results = reader.readtext(row_strip)

        print(f"\n🎮 SQUAD PLAYER #{idx} MAIN DIGITS (NO BARS):")
        for bbox, text, conf in ocr_results:
            cx = (bbox[0][0] + bbox[1][0]) / 2.0
            print(f"   • Text: '{text:<18}' | Conf: {conf:.2f} | X-Center: {cx:.1f}")

    print("\n" + "=" * 75)
    print("✅ Saved all 4 squad player strips to img/output/squad_player_*_main_digits_only.png")
    print("=" * 75)

if __name__ == "__main__":
    main()
