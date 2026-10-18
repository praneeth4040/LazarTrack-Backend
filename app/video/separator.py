"""
Unified Video Region Separator Module for LazarTrack Pipeline (Step 2).

Performs complete canvas division on normalized 1920x1080 frames into 3 functional regions:
1. Map Region (Left side: x = 0 to 960 px, y = 0 to 1080 px)
2. Team Slots Region (Top-right: x = 960 to 1920 px, y = 0 to split_y px)
3. Battle Log Event Feed Region (Bottom-right: x = 960 to 1920 px, y = split_y to 1080 px)

Uses dynamic fuzzy text search for 'BATTLE LOG' header on frame 0 to determine exact split_y.
"""

from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
import json
import os
import subprocess
from typing import Dict, Optional, Tuple
import cv2
import easyocr
import numpy as np


@dataclass
class RegionBounds:
    """Bounding box coordinates for all 3 separated regions."""

    split_x: int  # Vertical split boundary (960px)
    split_y_header: int  # Upper horizontal split (detected via 'BATTLE LOG' header)
    split_y_timer: int  # Lower horizontal split (detected via timer MM:SS)
    header_text_found: str
    header_fuzzy_score: float
    timer_text_found: str
    timer_score: float
    map_region: dict  # Left 50%
    team_slots_region: dict  # Top-right (between header and timer)
    battle_log_region: dict  # Bottom-right (below timer)


class RegionSeparator:
    """Splits normalized video frames into Map, Team Slots, and Battle Log Event Feed streams."""

    def __init__(
        self,
        reader: Optional[easyocr.Reader] = None,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
        split_x: int = 960,
        output_dir: str = "Backend/video/output/step2",
        target_header: str = "BATTLE LOG",
    ):
        """
        Initialize RegionSeparator.

        :param reader: EasyOCR Reader instance (creates a new instance if None)
        :param canvas_width: Normalized width (1920)
        :param canvas_height: Normalized height (1080)
        :param split_x: Vertical split boundary x-coordinate (960)
        :param output_dir: Directory where Step 2 separated outputs are stored
        """
        self.reader = reader if reader is not None else easyocr.Reader(["en"])
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.split_x = split_x
        self.output_dir = output_dir
        self.target_header = target_header.upper()
        os.makedirs(self.output_dir, exist_ok=True)

    def find_header_split_y(
        self, battle_log_frame_bgr: np.ndarray
    ) -> Tuple[int, str, float]:
        """
        Runs OCR on frame 0 of the right half (960x1080) to fuzzy search for 'BATTLE LOG'.
        Returns the Y position BELOW the header (where team slots end and events begin).

        :param battle_log_frame_bgr: Right 960x1080 sub-frame image
        :return: Tuple of (split_y, header_match_text, score)
        """
        # Search in vertical band y=180 to y=450 where BATTLE LOG header sits
        search_zone = battle_log_frame_bgr[180:450, :]
        ocr_results = self.reader.readtext(search_zone)

        best_text = ""
        best_score = 0.0
        split_y = 315  # Default split y fallback
        header_y = 315

        for bbox, text, conf in ocr_results:
            clean_text = text.upper().strip()
            ratio = SequenceMatcher(None, self.target_header, clean_text).ratio()

            if ratio > best_score:
                best_score = ratio
                best_text = text
                # Absolute y-coordinate of BATTLE LOG header TOP edge
                header_y = 180 + int(bbox[0][1])
                # Get bottom edge of the text box (bbox[2] is bottom-right corner)
                header_bottom = 180 + int(bbox[2][1])
                # Cut BELOW the header text, with some padding
                split_y = header_bottom + 10

        if best_score < 0.60:
            split_y = 315

        return split_y, best_text, round(best_score, 4)

    def find_timer_split_y(
        self, battle_log_frame_bgr: np.ndarray, header_split_y: int
    ) -> Tuple[int, str, float]:
        """
        Runs OCR on the lower portion of battle log frame to detect timer (MM:SS format).
        Searches below the header for the timer and cuts BEFORE it to exclude timer from team slots.

        :param battle_log_frame_bgr: Right 960x1080 sub-frame image
        :param header_split_y: Y coordinate of the BATTLE LOG header (upper cut point)
        :return: Tuple of (timer_split_y, timer_text, score)
        """
        h_img = battle_log_frame_bgr.shape[0]
        
        # Search in vertical band from header down to 70% of frame
        # Timer typically appears in lower portion of the screen
        search_start = min(header_split_y + 50, int(h_img * 0.3))  # Start after header
        search_end = int(h_img * 0.7)  # Search down to 70% of frame
        
        if search_end <= search_start:
            # Fallback if geometry doesn't make sense
            return int(h_img * 0.8), "", 0.0
        
        search_zone = battle_log_frame_bgr[search_start:search_end, :]
        ocr_results = self.reader.readtext(search_zone)

        best_text = ""
        best_score = 0.0
        timer_y = int(h_img * 0.8)  # Default fallback: 80% down
        timer_found = False

        # Look for MM:SS pattern (timer format: 00:00, 23:45, etc.)
        import re
        timer_pattern = r'^\d{1,2}:\d{2}$'
        
        for bbox, text, conf in ocr_results:
            clean_text = text.strip()
            
            # Check if text matches timer pattern (MM:SS)
            if re.match(timer_pattern, clean_text) and conf > 0.4:
                # Found a timer! Use its position as the split point
                abs_y_top = search_start + int(bbox[0][1])
                
                # We want to cut BEFORE the timer, so use the top edge
                if abs_y_top < timer_y:
                    timer_y = abs_y_top
                    best_text = clean_text
                    best_score = conf
                    timer_found = True

        if not timer_found:
            # No timer found, use default fallback
            return int(h_img * 0.8), "", 0.0

        return timer_y, best_text, round(best_score, 4)

    def process_normalized_video(
        self, video_dir_step1: str, fps: float = 5.0, save_frames: bool = True
    ) -> RegionBounds:
        """
        Process a Step 1 normalized video output directory and extract separate 3 video streams:
        1. Map Path (`map/map_path_video.mp4`)
        2. Team Slots (`team_slots/team_slots_video.mp4`)
        3. Battle Log Event Feed (`battle_log/battle_log_video.mp4`)

        :param video_dir_step1: Path to step1 output folder (e.g. Backend/video/output/step1/1)
        :param fps: Target video FPS (default 5.0)
        :param save_frames: Save individual PNG frame crops if True
        :return: RegionBounds dataclass object
        """
        video_name = os.path.basename(video_dir_step1.rstrip("/"))
        normalized_mp4 = os.path.join(
            video_dir_step1, f"{video_name}_normalized.mp4"
        )

        if not os.path.exists(normalized_mp4):
            raise FileNotFoundError(f"Normalized video not found: {normalized_mp4}")

        step2_out_dir = os.path.join(self.output_dir, video_name)
        map_out_dir = os.path.join(step2_out_dir, "map")
        team_out_dir = os.path.join(step2_out_dir, "team_slots")
        log_out_dir = os.path.join(step2_out_dir, "battle_log")

        os.makedirs(step2_out_dir, exist_ok=True)
        os.makedirs(map_out_dir, exist_ok=True)
        os.makedirs(team_out_dir, exist_ok=True)
        os.makedirs(log_out_dir, exist_ok=True)

        if save_frames:
            os.makedirs(os.path.join(map_out_dir, "frames"), exist_ok=True)
            os.makedirs(os.path.join(team_out_dir, "frames"), exist_ok=True)
            os.makedirs(os.path.join(log_out_dir, "frames"), exist_ok=True)

        # Open normalized video cap to detect header and timer
        cap = cv2.VideoCapture(normalized_mp4)
        ret, frame_0 = cap.read()
        if not ret or frame_0 is None:
            cap.release()
            raise ValueError(f"Failed to read frame 0 from {normalized_mp4}")

        # Extract right half for header fuzzy search
        right_half_0 = frame_0[:, self.split_x :]
        split_y_header, header_text, header_score = self.find_header_split_y(right_half_0)
        
        # If frame 0 does not show a clear header (e.g. black/intro screen), sample early frames
        if header_score < 0.60:
            for check_idx in [5, 10, 15, 20]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, check_idx)
                c_ret, c_frame = cap.read()
                if c_ret and c_frame is not None:
                    c_right = c_frame[:, self.split_x :]
                    sy, ht, hs = self.find_header_split_y(c_right)
                    if hs > header_score:
                        split_y_header, header_text, header_score = sy, ht, hs
                        right_half_0 = c_right
                        if header_score >= 0.80:
                            break

        # Detect timer for lower split
        split_y_timer, timer_text, timer_score = self.find_timer_split_y(right_half_0, split_y_header)

        # Region bounds calculation
        map_w = self.split_x
        right_w = self.canvas_width - self.split_x

        bounds = RegionBounds(
            split_x=self.split_x,
            split_y_header=split_y_header,
            split_y_timer=split_y_timer,
            header_text_found=header_text,
            header_fuzzy_score=header_score,
            timer_text_found=timer_text,
            timer_score=timer_score,
            map_region={
                "x_min": 0,
                "y_min": 0,
                "x_max": map_w,
                "y_max": self.canvas_height,
                "width": map_w,
                "height": self.canvas_height,
            },
            team_slots_region={
                "x_min": self.split_x,
                "y_min": 0,
                "x_max": self.canvas_width,
                "y_max": split_y_header,
                "width": right_w,
                "height": split_y_header,
            },
            battle_log_region={
                "x_min": self.split_x,
                "y_min": split_y_header,
                "x_max": self.canvas_width,
                "y_max": split_y_timer,
                "width": right_w,
                "height": split_y_timer - split_y_header,
            },
        )

        with open(os.path.join(step2_out_dir, "region_bounds.json"), "w") as f:
            json.dump(asdict(bounds), f, indent=2)

        # Reset capture to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Output video writers
        map_temp_mp4 = os.path.join(map_out_dir, "map_temp.mp4")
        map_final_mp4 = os.path.join(map_out_dir, "map_path_video.mp4")

        team_temp_mp4 = os.path.join(team_out_dir, "team_temp.mp4")
        team_final_mp4 = os.path.join(team_out_dir, "team_slots_video.mp4")

        log_temp_mp4 = os.path.join(log_out_dir, "log_temp.mp4")
        log_final_mp4 = os.path.join(log_out_dir, "battle_log_video.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        map_writer = cv2.VideoWriter(
            map_temp_mp4, fourcc, fps, (map_w, self.canvas_height)
        )
        if not map_writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            map_writer = cv2.VideoWriter(
                map_temp_mp4, fourcc, fps, (map_w, self.canvas_height)
            )

        team_writer = cv2.VideoWriter(
            team_temp_mp4, fourcc, fps, (right_w, split_y_header)
        )
        log_writer = cv2.VideoWriter(
            log_temp_mp4, fourcc, fps, (right_w, split_y_timer - split_y_header)
        )

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            map_crop = frame[:, : self.split_x]
            right_crop = frame[:, self.split_x :]

            team_crop = right_crop[:split_y_header, :]
            log_crop = right_crop[split_y_header:split_y_timer, :]

            map_writer.write(map_crop)
            team_writer.write(team_crop)
            log_writer.write(log_crop)

            if save_frames:
                frame_name = f"frame_{frame_idx:04d}.png"
                cv2.imwrite(
                    os.path.join(map_out_dir, "frames", frame_name), map_crop
                )
                cv2.imwrite(
                    os.path.join(team_out_dir, "frames", frame_name), team_crop
                )
                cv2.imwrite(
                    os.path.join(log_out_dir, "frames", frame_name), log_crop
                )

            frame_idx += 1

        cap.release()
        map_writer.release()
        team_writer.release()
        log_writer.release()

        # Re-encode streams to H.264
        self._reencode_h264(map_temp_mp4, map_final_mp4)
        self._reencode_h264(team_temp_mp4, team_final_mp4)
        self._reencode_h264(log_temp_mp4, log_final_mp4)

        print(
            f"[Step 2 Complete] Video '{video_name}' split into:"
            f"\n  - Map ({map_w}x{self.canvas_height})"
            f"\n  - Team Slots ({right_w}x{split_y_header}) [Y: 0 to {split_y_header}]"
            f"\n  - Battle Log ({right_w}x{split_y_timer - split_y_header}) [Y: {split_y_header} to {split_y_timer}]"
            f"\n  Across {frame_idx} frames @ {fps} FPS"
        )
        return bounds

    def _reencode_h264(self, temp_path: str, final_path: str):
        """Helper to encode video with ffmpeg to standard libx264 yuv420p."""
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-loglevel",
            "error",
            final_path,
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            if os.path.exists(temp_path):
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(temp_path, final_path)
