"""
Player Path Normalizer (Step 1)

Normalizes match recordings to standard 1920x1080 canvas while preserving the
complete native framerate (no downsampling) for smooth player path tracking.
"""

from dataclasses import dataclass, asdict
import json
import logging
import os
import subprocess
from typing import Optional, Tuple
import cv2
import numpy as np

log = logging.getLogger("app.zone_tracker.normalizer")


@dataclass
class VideoMetadata:
    """Metadata describing input and normalized video."""
    filepath: str
    original_width: int
    original_height: int
    original_fps: float
    total_frames: int
    duration_seconds: float
    target_width: int
    target_height: int
    target_fps: float


class PlayerPathNormalizer:
    """Standardizes video canvas to 1920x1080 at full native framerate."""

    def __init__(
        self,
        target_width: int = 1920,
        target_height: int = 1080,
        target_fps: Optional[float] = None,
        keep_aspect_ratio: bool = True,
        output_dir: str = "video/output/zone/step1",
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps  # None means keep full native FPS
        self.keep_aspect_ratio = keep_aspect_ratio
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Inspect video file and extract properties."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        effective_target_fps = self.target_fps if (self.target_fps and self.target_fps > 0) else fps

        return VideoMetadata(
            filepath=video_path,
            original_width=w,
            original_height=h,
            original_fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            target_width=self.target_width,
            target_height=self.target_height,
            target_fps=effective_target_fps,
        )

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize and pad a single frame to 1920x1080 canvas."""
        if frame is None or frame.size == 0:
            raise ValueError("Empty or invalid frame.")

        h, w = frame.shape[:2]
        if not self.keep_aspect_ratio:
            return cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)

        scale = min(self.target_width / w, self.target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        top = (self.target_height - new_h) // 2
        left = (self.target_width - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def process(self, video_path: str, save_frames: bool = False) -> Tuple[str, VideoMetadata]:
        """
        Normalize video to target resolution at full native FPS.
        Returns path to normalized MP4 and metadata.
        """
        filename = os.path.basename(video_path)
        stem = os.path.splitext(filename)[0]
        video_out_dir = os.path.join(self.output_dir, stem)
        os.makedirs(video_out_dir, exist_ok=True)

        metadata = self.get_metadata(video_path)
        effective_fps = metadata.target_fps if metadata.target_fps > 0 else 30.0

        meta_json_path = os.path.join(video_out_dir, "metadata.json")
        with open(meta_json_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        normalized_video = os.path.join(video_out_dir, f"{stem}_normalized.mp4")

        if not save_frames:
            # Fast single-pass hardware-accelerated canvas normalization
            vcodec_args = (
                ["-c:v", "h264_videotoolbox", "-b:v", "4M"]
                if os.uname().sysname == "Darwin"
                else ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
            )
            vf = (
                f"scale={self.target_width}:{self.target_height}:force_original_aspect_ratio=decrease,"
                f"pad={self.target_width}:{self.target_height}:(ow-iw)/2:(oh-ih)/2"
                if self.keep_aspect_ratio
                else f"scale={self.target_width}:{self.target_height}"
            )
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", vf,
                "-r", str(effective_fps),
                *vcodec_args,
                "-loglevel", "error",
                normalized_video,
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                log.info(f"[PlayerPath:Step1] Normalized '{filename}' -> {metadata.total_frames} frames @ {effective_fps:.2f} FPS (fast-path)")
                return normalized_video, metadata
            except Exception as e:
                log.warning(f"[PlayerPath:Step1] Fast ffmpeg path failed ({e}); falling back to frame loop")

        # Fallback / save_frames path
        temp_video = os.path.join(video_out_dir, f"{stem}_temp.mp4")
        normalized_video = os.path.join(video_out_dir, f"{stem}_normalized.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(temp_video, fourcc, effective_fps, (self.target_width, self.target_height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_video, fourcc, effective_fps, (self.target_width, self.target_height))

        cap = cv2.VideoCapture(video_path)
        frame_interval = 1
        if self.target_fps and self.target_fps > 0 and metadata.original_fps > self.target_fps:
            frame_interval = max(1, int(round(metadata.original_fps / self.target_fps)))

        frames_dir = os.path.join(video_out_dir, "frames")
        if save_frames:
            os.makedirs(frames_dir, exist_ok=True)

        frame_idx = 0
        written_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_interval == 0:
                normalized = self.normalize_frame(frame)
                writer.write(normalized)
                if save_frames:
                    cv2.imwrite(os.path.join(frames_dir, f"frame_{written_count:05d}.png"), normalized)
                written_count += 1

            frame_idx += 1

        cap.release()
        writer.release()

        # Re-encode to standard web-compatible H.264
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_video,
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-loglevel", "error", normalized_video
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            if os.path.exists(temp_video):
                os.remove(temp_video)
        except Exception:
            if os.path.exists(temp_video):
                if os.path.exists(normalized_video):
                    os.remove(normalized_video)
                os.rename(temp_video, normalized_video)

        log.info(f"[PlayerPath:Step1] Normalized '{filename}' -> {written_count} frames @ {effective_fps:.2f} FPS")
        return normalized_video, metadata
