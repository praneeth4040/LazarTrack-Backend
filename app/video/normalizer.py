"""
Video Normalizer Module for LazarTrack Battle Log Extraction.

Normalizes input video streams from varying mobile device resolutions, aspect ratios,
and variable frame rates (VFR) to a standardized target resolution and frame rate.
Uses H.264 (avc1) codec for high browser & device web playback compatibility.
"""

from dataclasses import dataclass, asdict
import json
import os
import subprocess
from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Stores metadata for a video stream."""

    filepath: str
    original_width: int
    original_height: int
    original_fps: float
    total_frames: int
    duration_seconds: float
    target_width: int
    target_height: int
    target_fps: float


class VideoNormalizer:
    """Normalizes video frames to a standard resolution and FPS grid across all devices."""

    def __init__(
        self,
        target_width: int = 1920,
        target_height: int = 1080,
        target_fps: float = 5.0,
        keep_aspect_ratio: bool = True,
        output_dir: str = "Backend/video/output/step1",
    ):
        """
        Initialize the VideoNormalizer.

        :param target_width: Standardized width for downstream processing (default 1920)
        :param target_height: Standardized height for downstream processing (default 1080)
        :param target_fps: Frame rate at which to sample the video (default 5 FPS)
        :param keep_aspect_ratio: Maintain aspect ratio with letterboxing/black bars if True
        :param output_dir: Directory where Step 1 normalized outputs/frames/metadata are saved
        """
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.keep_aspect_ratio = keep_aspect_ratio
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Inspect video file and return metadata."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        cap.release()

        return VideoMetadata(
            filepath=video_path,
            original_width=w,
            original_height=h,
            original_fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            target_width=self.target_width,
            target_height=self.target_height,
            target_fps=self.target_fps,
        )

    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize and pad a single frame to target dimensions.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Empty or invalid frame provided.")

        h, w = frame.shape[:2]

        if not self.keep_aspect_ratio:
            return cv2.resize(
                frame,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA,
            )

        # Calculate scale keeping aspect ratio
        scale = min(self.target_width / w, self.target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        # Create canvas and place resized image in center
        canvas = np.zeros(
            (self.target_height, self.target_width, 3), dtype=np.uint8
        )
        top = (self.target_height - new_h) // 2
        left = (self.target_width - new_w) // 2

        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def process_and_save_video(
        self, video_path: str, save_frames: bool = True
    ) -> str:
        """
        Process an input video, save normalized video (and optional sample frame images + metadata)
        to the step1 output directory using H.264 encoding for max web/device compatibility.
        """
        filename = os.path.basename(video_path)
        name_without_ext = os.path.splitext(filename)[0]
        video_out_dir = os.path.join(self.output_dir, name_without_ext)
        os.makedirs(video_out_dir, exist_ok=True)

        frames_out_dir = os.path.join(video_out_dir, "frames")
        if save_frames:
            os.makedirs(frames_out_dir, exist_ok=True)

        metadata = self.get_metadata(video_path)

        # Save metadata JSON
        meta_json_path = os.path.join(video_out_dir, "metadata.json")
        with open(meta_json_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Temp raw video path
        temp_video_path = os.path.join(
            video_out_dir, f"{name_without_ext}_temp.mp4"
        )
        normalized_video_path = os.path.join(
            video_out_dir, f"{name_without_ext}_normalized.mp4"
        )

        # Try avc1 first, fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(
            temp_video_path,
            fourcc,
            self.target_fps,
            (self.target_width, self.target_height),
        )

        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                temp_video_path,
                fourcc,
                self.target_fps,
                (self.target_width, self.target_height),
            )

        cap = cv2.VideoCapture(video_path)
        if metadata.original_fps <= 0:
            frame_interval = 1
        else:
            frame_interval = max(
                1, int(round(metadata.original_fps / self.target_fps))
            )

        frame_idx = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_interval == 0:
                timestamp_ms = (
                    (frame_idx / metadata.original_fps) * 1000.0
                    if metadata.original_fps > 0
                    else 0.0
                )
                normalized = self.normalize_frame(frame)

                # Write frame to video
                writer.write(normalized)

                # Save sample frame image if requested
                if save_frames:
                    frame_name = f"frame_{saved_count:04d}_ts_{int(timestamp_ms)}ms.png"
                    cv2.imwrite(
                        os.path.join(frames_out_dir, frame_name), normalized
                    )

                saved_count += 1

            frame_idx += 1

        cap.release()
        writer.release()

        # Re-encode with ffmpeg to standard H.264 (yuv420p) if ffmpeg is installed
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-loglevel",
            "error",
            normalized_video_path,
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except Exception:
            # If ffmpeg is missing or fails, rename temp file as normalized video
            if os.path.exists(temp_video_path):
                if os.path.exists(normalized_video_path):
                    os.remove(normalized_video_path)
                os.rename(temp_video_path, normalized_video_path)

        print(
            f"[Step 1 Complete] {filename} -> {video_out_dir} ({saved_count} frames @ {self.target_fps} FPS H.264)"
        )
        return video_out_dir

    def stream_normalized_frames(
        self, video_path: str
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield normalized frames sampled at target_fps along with frame index and timestamp (ms)."""
        metadata = self.get_metadata(video_path)
        cap = cv2.VideoCapture(video_path)

        if metadata.original_fps <= 0:
            frame_interval = 1
        else:
            frame_interval = max(
                1, int(round(metadata.original_fps / self.target_fps))
            )

        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_interval == 0:
                timestamp_ms = (
                    (frame_idx / metadata.original_fps) * 1000.0
                    if metadata.original_fps > 0
                    else 0.0
                )
                normalized = self.normalize_frame(frame)
                yield frame_idx, timestamp_ms, normalized

            frame_idx += 1

        cap.release()
