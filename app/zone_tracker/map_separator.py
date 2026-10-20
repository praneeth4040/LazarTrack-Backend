"""
Player Path Map Separator (Step 2)

Isolates the 960x1080 map stream at full native framerate and extracts the team roster
reference image for tracking player movements.
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

log = logging.getLogger("app.zone_tracker.map_separator")


@dataclass
class MapSeparationResult:
    """Artifacts produced by Step 2 Map Separator."""
    map_video_path: str
    team_image_path: str
    split_x: int
    split_y_header: int
    fps: float
    total_frames: int
    region_bounds: dict


class PlayerPathMapSeparator:
    """Isolates the left 960px Minimap stream from normalized 1920x1080 video."""

    def __init__(
        self,
        reader: Optional[easyocr.Reader] = None,
        split_x: int = 960,
        output_dir: str = "video/output/zone/step2",
    ):
        self.reader = reader
        self.split_x = split_x
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_reader(self) -> easyocr.Reader:
        if self.reader is None:
            self.reader = easyocr.Reader(["en"])
        return self.reader

    def find_header_split_y(self, right_half_frame: np.ndarray) -> int:
        """Find the Y coordinate of the BATTLE LOG header on the right half."""
        reader = self._get_reader()
        search_zone = right_half_frame[180:450, :]
        ocr_results = reader.readtext(search_zone)

        best_score = 0.0
        split_y = 365  # Sensible default fallback

        for bbox, text, conf in ocr_results:
            clean_text = text.upper().strip()
            ratio = SequenceMatcher(None, "BATTLE LOG", clean_text).ratio()
            if ratio > best_score:
                best_score = ratio
                header_bottom = 180 + int(bbox[2][1])
                split_y = header_bottom + 10

        return split_y

    def process(
        self,
        normalized_video_path: str,
        video_id: str,
        save_map_frames: bool = False,
    ) -> MapSeparationResult:
        """
        Extract isolated 960x1080 Map stream and team roster reference frame.
        """
        if not os.path.exists(normalized_video_path):
            raise FileNotFoundError(f"Normalized video not found: {normalized_video_path}")

        video_out_dir = os.path.join(self.output_dir, video_id)
        os.makedirs(video_out_dir, exist_ok=True)

        cap = cv2.VideoCapture(normalized_video_path)
        ret, frame_0 = cap.read()
        if not ret or frame_0 is None:
            cap.release()
            raise ValueError(f"Failed to read frame 0 from: {normalized_video_path}")

        h, w = frame_0.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0

        # Detect split_y from frame 0
        right_half_0 = frame_0[:, self.split_x :]
        split_y_header = self.find_header_split_y(right_half_0)

        # Save frame 0 of team slots as team_image.png
        team_crop_0 = right_half_0[:split_y_header, :]
        team_image_path = os.path.join(video_out_dir, "team_image.png")
        cv2.imwrite(team_image_path, team_crop_0)

        total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        map_w = self.split_x
        map_final = os.path.join(video_out_dir, "map_video.mp4")

        if not save_map_frames:
            vcodec_args = (
                ["-c:v", "h264_videotoolbox", "-b:v", "4M"]
                if os.uname().sysname == "Darwin"
                else ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
            )
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", normalized_video_path,
                "-vf", f"crop={map_w}:{h}:0:0",
                "-r", str(fps),
                *vcodec_args,
                "-loglevel", "error",
                map_final,
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                frame_idx = total_count
            except Exception as e:
                log.warning(f"[PlayerPath:Step2] Fast ffmpeg crop failed ({e}); falling back to frame loop")
                save_map_frames = True

        if save_map_frames:
            cap = cv2.VideoCapture(normalized_video_path)
            # Map video writer setup
            map_temp = os.path.join(video_out_dir, "map_temp.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(map_temp, fourcc, fps, (map_w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(map_temp, fourcc, fps, (map_w, h))

            frames_dir = os.path.join(video_out_dir, "map_frames")
            os.makedirs(frames_dir, exist_ok=True)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                map_crop = frame[:, : self.split_x]
                writer.write(map_crop)
                cv2.imwrite(os.path.join(frames_dir, f"map_{frame_idx:05d}.png"), map_crop)
                frame_idx += 1

            cap.release()
            writer.release()

            # Re-encode map video to standard H.264
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", map_temp,
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-loglevel", "error", map_final
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                if os.path.exists(map_temp):
                    os.remove(map_temp)
            except Exception:
                if os.path.exists(map_temp):
                    if os.path.exists(map_final):
                        os.remove(map_final)
                    os.rename(map_temp, map_final)

        region_bounds = {
            "map_region": {"x_min": 0, "y_min": 0, "x_max": map_w, "y_max": h, "width": map_w, "height": h},
            "team_slots_region": {"x_min": self.split_x, "y_min": 0, "x_max": w, "y_max": split_y_header, "width": w - self.split_x, "height": split_y_header},
        }

        bounds_path = os.path.join(video_out_dir, "region_bounds.json")
        with open(bounds_path, "w") as f:
            json.dump(region_bounds, f, indent=2)

        log.info(f"[PlayerPath:Step2] Isolated map video for '{video_id}' -> {frame_idx} frames @ {fps:.2f} FPS")

        return MapSeparationResult(
            map_video_path=map_final,
            team_image_path=team_image_path,
            split_x=self.split_x,
            split_y_header=split_y_header,
            fps=fps,
            total_frames=frame_idx,
            region_bounds=region_bounds,
        )
