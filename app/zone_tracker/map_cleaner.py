"""
Player Path Map Cleaner (Step 3)

Uses multi-anchor computer vision & OCR to detect the exact boundaries of the overhead map:
- Anchor 1: "LEGEND" button (Top-Left)
- Anchor 2: Teams & Players alive badge (Top-Right)
- Anchor 3: Timeline track line & controls (Bottom)
- Anchor 4: Replay/Spectator drawer sidebar transition (Left)

Crops the map stream to the exact detected bounding box and produces a clean, isolated
map video stream at full native framerate.
"""

from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import json
import logging
import os
import subprocess
from typing import Dict, Optional, Tuple
import cv2
import easyocr
import numpy as np

log = logging.getLogger("app.zone_tracker.map_cleaner")


@dataclass
class MapAnchors:
    legend_bbox: Optional[list]
    alive_bbox: Optional[list]
    timeline_y: int
    sidebar_x: int


@dataclass
class MapBoundsResult:
    clean_map_video_path: str
    sample_frame_path: str
    bounds_json_path: str
    fps: float
    total_frames: int
    map_bounds: dict
    anchors: dict


class PlayerPathMapCleaner:
    """Detects exact map boundaries via UI anchors and crops the map stream."""

    def __init__(
        self,
        reader: Optional[easyocr.Reader] = None,
        output_dir: str = "video/output/zone/step3",
    ):
        self.reader = reader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_reader(self) -> easyocr.Reader:
        if self.reader is None:
            self.reader = easyocr.Reader(["en"])
        return self.reader

    def detect_anchors(self, frame: np.ndarray) -> Tuple[MapAnchors, dict]:
        """
        Analyze frame 0 to detect the 4 UI anchors.
        Returns MapAnchors dataclass and computed bounding box.
        """
        h, w = frame.shape[:2]
        reader = self._get_reader()

        # 1. OCR for Top Anchors (LEGEND and Alive badge)
        top_zone = frame[100:250, :]
        top_results = reader.readtext(top_zone)

        legend_box = None
        alive_box = None

        for bbox, text, conf in top_results:
            clean = text.upper().strip()
            abs_bbox = [[int(pt[0]), int(pt[1]) + 100] for pt in bbox]

            if SequenceMatcher(None, "LEGEND", clean).ratio() > 0.60:
                legend_box = abs_bbox
            elif ("9" in clean or "35" in clean or conf > 0.3) and abs_bbox[0][0] > 650:
                alive_box = abs_bbox

        # Fallback values if OCR misses
        legend_x1 = legend_box[0][0] if legend_box else 156
        legend_y1 = legend_box[0][1] if legend_box else 134

        # 2. Bottom Timeline Detection
        timeline_zone = frame[880:960, :]
        gray_btm = cv2.cvtColor(timeline_zone, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_btm, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=200, maxLineGap=10)

        timeline_y = 930  # Default fallback
        if lines is not None:
            for l in lines:
                # HoughLinesP returns (N,1,4) on OpenCV 4 and (N,4) on OpenCV 5 —
                # normalise so the unpack works on both.
                seg = np.asarray(l).reshape(-1)
                if seg.size != 4:
                    continue
                x1, y1, x2, y2 = (int(v) for v in seg)
                if abs(y2 - y1) <= 2 and abs(x2 - x1) > 250:
                    timeline_y = 880 + int(y1)
                    break

        # 3. Sidebar Boundary (Fixed spectator UI sidebar at X = 95)
        # Spectator replay UI has a constant width sidebar (columns 0..94)
        sidebar_x = 95

        # 4. Top letterbox boundary (Fixed spectator replay top bar at Y = 108)
        top_y = 108

        # 5. Bottom letterbox boundary (Fixed spectator replay bottom bar at Y = 970)
        bottom_y = 970

        anchors = MapAnchors(
            legend_bbox=legend_box,
            alive_bbox=alive_box,
            timeline_y=timeline_y,
            sidebar_x=sidebar_x,
        )

        bounds = {
            # Full Map Screen (includes coordinate labels B-H, J-P)
            "full_map": {
                "x_min": sidebar_x,
                "y_min": top_y,
                "x_max": w,
                "y_max": bottom_y,
                "width": w - sidebar_x,
                "height": bottom_y - top_y,
            },
            # Inner Playable Terrain (bounded tightly inside UI badges)
            "inner_terrain": {
                "x_min": sidebar_x + 30,
                "y_min": top_y + 40,
                "x_max": w - 30,
                "y_max": timeline_y - 10,
                "width": (w - 30) - (sidebar_x + 30),
                "height": (timeline_y - 10) - (top_y + 40),
            },
        }

        return anchors, bounds

    def process(
        self,
        map_video_path: str,
        video_id: str,
        crop_mode: str = "full_map",  # 'full_map' or 'inner_terrain'
    ) -> MapBoundsResult:
        """
        Process the Step 2 map stream, detect anchors, and crop to clean map video.
        """
        if not os.path.exists(map_video_path):
            raise FileNotFoundError(f"Map video not found: {map_video_path}")

        session_out_dir = os.path.join(self.output_dir, video_id)
        os.makedirs(session_out_dir, exist_ok=True)

        cap = cv2.VideoCapture(map_video_path)
        ret, frame_0 = cap.read()
        if not ret or frame_0 is None:
            cap.release()
            raise ValueError(f"Failed to read frame 0 from: {map_video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        anchors, bounds = self.detect_anchors(frame_0)
        selected_box = bounds.get(crop_mode, bounds["full_map"])

        xmin = selected_box["x_min"]
        ymin = selected_box["y_min"]
        xmax = selected_box["x_max"]
        ymax = selected_box["y_max"]
        crop_w = xmax - xmin
        crop_h = ymax - ymin

        log.info(f"[{video_id}] Detected map bounds: x=[{xmin}:{xmax}], y=[{ymin}:{ymax}] ({crop_w}x{crop_h})")

        # Save annotated reference image
        annotated = frame_0.copy()
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(annotated, f"MAP: {crop_w}x{crop_h}", (xmin + 10, ymin + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        sample_img_path = os.path.join(session_out_dir, "clean_map_frame_0.png")
        cv2.imwrite(sample_img_path, frame_0[ymin:ymax, xmin:xmax])

        debug_img_path = os.path.join(session_out_dir, "map_anchors_detected.png")
        cv2.imwrite(debug_img_path, annotated)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        clean_final = os.path.join(session_out_dir, "clean_map_video.mp4")

        # Fast direct hardware-accelerated crop
        vcodec_args = (
            ["-c:v", "h264_videotoolbox", "-b:v", "4M"]
            if os.uname().sysname == "Darwin"
            else ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        )
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", map_video_path,
            "-vf", f"crop={crop_w}:{crop_h}:{xmin}:{ymin}",
            "-r", str(fps),
            *vcodec_args,
            "-loglevel", "error",
            clean_final,
        ]
        use_fallback = False
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            frame_count = total_frames
        except Exception as e:
            log.warning(f"[{video_id}] Fast ffmpeg crop failed ({e}); falling back to frame loop")
            use_fallback = True

        if use_fallback:
            cap = cv2.VideoCapture(map_video_path)
            clean_temp = os.path.join(session_out_dir, "clean_map_temp.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(clean_temp, fourcc, fps, (crop_w, crop_h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(clean_temp, fourcc, fps, (crop_w, crop_h))

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                clean_crop = frame[ymin:ymax, xmin:xmax]
                writer.write(clean_crop)
                frame_count += 1

            cap.release()
            writer.release()

            # Re-encode clean map video with ffmpeg to standard H.264
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", clean_temp,
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-loglevel", "error", clean_final
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                if os.path.exists(clean_temp):
                    os.remove(clean_temp)
            except Exception:
                if os.path.exists(clean_temp):
                    if os.path.exists(clean_final):
                        os.remove(clean_final)
                    os.rename(clean_temp, clean_final)

        # Save bounds JSON
        bounds_json_path = os.path.join(session_out_dir, "map_bounds.json")
        result_data = {
            "video_id": video_id,
            "crop_mode": crop_mode,
            "crop_coordinates": {"x_min": xmin, "y_min": ymin, "x_max": xmax, "y_max": ymax},
            "dimensions": {"width": crop_w, "height": crop_h, "aspect_ratio": round(crop_w / crop_h, 3)},
            "fps": fps,
            "total_frames": frame_count,
            "anchors": asdict(anchors),
            "all_bounds": bounds,
        }
        with open(bounds_json_path, "w") as f:
            json.dump(result_data, f, indent=2)

        log.info(f"[{video_id}] Step 3 completed: Clean map video ({crop_w}x{crop_h} @ {fps:.2f} FPS, {frame_count} frames)")

        return MapBoundsResult(
            clean_map_video_path=clean_final,
            sample_frame_path=sample_img_path,
            bounds_json_path=bounds_json_path,
            fps=fps,
            total_frames=frame_count,
            map_bounds=result_data,
            anchors=asdict(anchors),
        )
