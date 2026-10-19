"""
Step 8 (Zone Pipeline) - Zone Renderer
Input : zones.json, assets/maps/bermuda.jpeg
Output: zones_map.png

Renders all detected safe zones onto the base Bermuda map with distinct color coding,
labels, center anchors, and a detailed progression legend.
"""
import argparse
import json
import os
import cv2
import numpy as np


# Single clean white zone styling with dark contrast
ZONE_COLOR_WHITE = (255, 255, 255)  # Pure White (BGR)
ZONE_BORDER_DARK = (10, 10, 10)     # Solid Dark Outline for contrast on light terrain


def render_zones_map(
    zones_json_path: str,
    base_map_path: str,
    out_png_path: str,
    show_labels: bool = True,
):
    base = cv2.imread(base_map_path)
    if base is None:
        raise FileNotFoundError(f"Base map not found at {base_map_path}")
    
    with open(zones_json_path) as f:
        data = json.load(f)
    zones = data.get("zones", [])

    vis = base.copy()
    h_base, w_base = base.shape[:2]
    scale = max(1.0, max(h_base, w_base) / 1024.0)

    # Line metrics: crisp white stroke with dark outer shadow
    line_w = max(2, int(3 * scale))
    halo_w = max(5, int(6 * scale))
    dot_r1 = max(4, int(5 * scale))
    dot_r2 = max(2, int(2.5 * scale))

    # 1. Draw circles: solid dark shadow + pure white stroke
    for z in zones:
        cx, cy, r = int(z["cx"]), int(z["cy"]), int(z["r"])
        # Outer dark shadow for contrast against terrain
        cv2.circle(vis, (cx, cy), r, ZONE_BORDER_DARK, halo_w, cv2.LINE_AA)
        # Inner clean white stroke
        cv2.circle(vis, (cx, cy), r, ZONE_COLOR_WHITE, line_w, cv2.LINE_AA)
        # Center anchor
        cv2.circle(vis, (cx, cy), dot_r1, ZONE_BORDER_DARK, -1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), dot_r2, ZONE_COLOR_WHITE, -1, cv2.LINE_AA)

    # 2. Sleek minimal zone badges (NO radius, NO px, NO timestamps)
    if show_labels:
        font_scale = 0.52 * scale
        angle_offsets = [-np.pi / 4, -3 * np.pi / 4, np.pi / 6, 5 * np.pi / 6, -np.pi / 2, np.pi / 2]

        for i, z in enumerate(zones):
            cx, cy, r = int(z["cx"]), int(z["cy"]), int(z["r"])
            angle = angle_offsets[i % len(angle_offsets)]
            lx = int(cx + r * np.cos(angle))
            ly = int(cy + r * np.sin(angle))
            label = str(z.get("name", f"Zone {i+1}"))

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, max(1, int(scale)))
            pad_x, pad_y = int(5 * scale), int(5 * scale)

            # High-contrast dark badge backing with white border
            cv2.rectangle(
                vis,
                (lx - pad_x, ly - th - pad_y),
                (lx + tw + pad_x, ly + pad_y),
                ZONE_BORDER_DARK,
                -1,
            )
            cv2.rectangle(
                vis,
                (lx - pad_x, ly - th - pad_y),
                (lx + tw + pad_x, ly + pad_y),
                ZONE_COLOR_WHITE,
                max(1, int(scale)),
            )
            # Crisp white text inside dark badge
            cv2.putText(
                vis,
                label,
                (lx, ly),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                ZONE_COLOR_WHITE,
                max(1, int(scale)),
                cv2.LINE_AA,
            )

    out_dir = os.path.dirname(out_png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_png_path, vis)
    print(f"[zone_renderer] Saved complete zones map -> {out_png_path}")
    return out_png_path


def main():
    ap = argparse.ArgumentParser(description="Render detected safe zones on Bermuda base map")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--base", default="assets/maps/bermuda.jpeg")
    ap.add_argument("--out", default="zones_map.png")
    a = ap.parse_args()
    render_zones_map(a.zones, a.base, a.out)


if __name__ == "__main__":
    main()
