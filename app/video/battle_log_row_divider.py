"""
Battle Log Row Divider for LazarTrack (Step 4.1)
Divides battle_log frames into 8 rows and creates individual row videos.
Each row represents a single battle event line (kill, assist, etc).
"""

import json
import os
import subprocess
from typing import Dict, List, Tuple
import cv2
import numpy as np


class BattleLogRowDivider:
    """Divides battle_log frames into 8 equal rows."""

    def __init__(
        self,
        num_rows: int = 8,
        target_fps: float = 5.0,
        output_dir: str = "Backend/video/output/step4",
    ):
        """
        Initialize BattleLogRowDivider.

        :param num_rows: Maximum number of rows (0-8)
        :param target_fps: Frame rate for output videos
        :param output_dir: Step 4 output directory
        """
        self.num_rows = num_rows
        self.target_fps = target_fps
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def divide_frame_into_rows(
        self,
        frame: np.ndarray,
        frame_index: int,
        draw_lines: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Divide a single battle_log frame into rows.

        :param frame: Input battle_log frame
        :param frame_index: Index of the frame
        :param draw_lines: Whether to draw cut lines on frame
        :return: Tuple of (annotated_frame, row_metadata)
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        h, w = frame.shape[:2]
        row_height = h / self.num_rows

        # Create a copy for annotation
        annotated = frame.copy()

        # Draw horizontal lines for row divisions
        row_info = {
            "frame_index": frame_index,
            "frame_height": h,
            "frame_width": w,
            "num_rows": self.num_rows,
            "row_height": row_height,
            "rows": []
        }

        for row_idx in range(1, self.num_rows):
            y = int(row_idx * row_height)
            
            # Draw white line with text label
            if draw_lines:
                cv2.line(annotated, (0, y), (w, y), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"Row {row_idx}",
                    (10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            row_info["rows"].append({
                "row_index": row_idx,
                "y_pixel": y,
                "description": f"Divider between row {row_idx-1} and row {row_idx}"
            })

        # Add row boundaries metadata
        for i in range(self.num_rows):
            y_start = int(i * row_height)
            y_end = int((i + 1) * row_height) if i < self.num_rows - 1 else h

            if draw_lines and i == 0:
                cv2.putText(
                    annotated,
                    f"Row 0",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 100, 100),
                    1
                )

        # Add header info
        header_text = f"Frame {frame_index} | {self.num_rows} Rows | H:{h}px W:{w}px"
        cv2.rectangle(annotated, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            header_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1
        )

        return annotated, row_info

    def extract_rows_from_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> Dict[int, np.ndarray]:
        """
        Extract individual row images from a battle_log frame.

        :param frame: Input battle_log frame
        :param frame_index: Index of the frame
        :return: Dict mapping row_index -> row_image
        """
        if frame is None:
            raise ValueError("Frame cannot be None")

        h, w = frame.shape[:2]
        row_height = h / self.num_rows
        rows = {}

        for row_idx in range(self.num_rows):
            y_start = int(row_idx * row_height)
            y_end = int((row_idx + 1) * row_height) if row_idx < self.num_rows - 1 else h
            
            row_image = frame[y_start:y_end, :]
            rows[row_idx] = row_image

        return rows

    def process_video_frames(
        self,
        video_id: str,
        battle_log_frames_dir: str,
        output_dir: str = None,
    ) -> Dict:
        """
        Process all battle_log frames for a video.

        :param video_id: ID of the video
        :param battle_log_frames_dir: Directory with battle_log frames
        :param output_dir: Output directory for results
        :return: Summary of processed frames
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(battle_log_frames_dir),
                "..",
                f"battle_log_rows_{video_id}"
            )

        os.makedirs(output_dir, exist_ok=True)
        
        # Get frame files
        frame_files = sorted([f for f in os.listdir(battle_log_frames_dir) 
                             if f.endswith(".png")])
        
        if not frame_files:
            raise ValueError(f"No frames found in {battle_log_frames_dir}")

        print(f"[Step 4.2] Processing battle_log rows for video {video_id}")
        print(f"  Total frames: {len(frame_files)}")
        print(f"  Rows per frame: {self.num_rows}")

        # Process each frame
        frame_metadata = []
        
        for frame_idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(battle_log_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"  Warning: Could not read {frame_file}")
                continue

            # Get annotated frame with lines
            annotated_frame, row_info = self.divide_frame_into_rows(
                frame, frame_idx, draw_lines=True
            )

            # Extract individual rows
            rows = self.extract_rows_from_frame(frame, frame_idx)

            # Save annotated frame with grid lines
            annotated_path = os.path.join(
                output_dir,
                f"frame_{frame_idx:04d}_annotated.png"
            )
            cv2.imwrite(annotated_path, annotated_frame)

            # Save individual rows
            rows_subdir = os.path.join(output_dir, f"frame_{frame_idx:04d}_rows")
            os.makedirs(rows_subdir, exist_ok=True)
            
            row_paths = {}
            for row_idx, row_image in rows.items():
                row_path = os.path.join(rows_subdir, f"row_{row_idx:01d}.png")
                cv2.imwrite(row_path, row_image)
                row_paths[row_idx] = row_path

            # Store metadata
            frame_metadata.append({
                "frame_index": frame_idx,
                "original_file": frame_file,
                "annotated_output": annotated_path,
                "row_indices": list(rows.keys()),
                "row_outputs": row_paths,
                "frame_info": row_info
            })

            if (frame_idx + 1) % max(1, len(frame_files) // 5) == 0:
                print(f"  Progress: {frame_idx + 1}/{len(frame_files)} frames processed")

        # Save frame index metadata
        metadata_path = os.path.join(output_dir, "frame_index.json")
        with open(metadata_path, "w") as f:
            json.dump(frame_metadata, f, indent=2)

        print(f"  ✓ All frames processed and indexed")
        print(f"  Metadata saved: {metadata_path}")

        summary = {
            "video_id": video_id,
            "total_frames": len(frame_files),
            "rows_per_frame": self.num_rows,
            "output_dir": output_dir,
            "metadata_file": metadata_path,
            "frame_count": len(frame_metadata)
        }

        return summary

    def create_row_videos(
        self,
        video_id: str,
        rows_output_dir: str,
        output_video_dir: str = None,
    ) -> Dict[int, str]:
        """
        Create individual videos for each row (0-7).

        :param video_id: ID of the video
        :param rows_output_dir: Directory with frame_{idx}_rows subdirectories
        :param output_video_dir: Directory to save row videos
        :return: Dict mapping row_index -> video_path
        """
        if output_video_dir is None:
            output_video_dir = os.path.join(
                os.path.dirname(rows_output_dir),
                "row_videos"
            )
        
        os.makedirs(output_video_dir, exist_ok=True)

        print(f"  Creating row videos for video {video_id}...")

        # Get all frame directories
        frame_dirs = sorted([d for d in os.listdir(rows_output_dir)
                           if d.startswith("frame_") and d.endswith("_rows")
                           and os.path.isdir(os.path.join(rows_output_dir, d))])

        if not frame_dirs:
            raise ValueError(f"No frame row directories found in {rows_output_dir}")

        # Get dimensions from first row
        first_frame_dir = os.path.join(rows_output_dir, frame_dirs[0])
        first_row_path = os.path.join(first_frame_dir, "row_0.png")
        
        if not os.path.exists(first_row_path):
            raise ValueError(f"Could not find sample row: {first_row_path}")

        sample_row = cv2.imread(first_row_path)
        h, w = sample_row.shape[:2]

        # Create video writer for each row
        row_videos = {}
        writers = {}

        print(f"    Row dimensions: {w}x{h}px")
        print(f"    Total frames: {len(frame_dirs)}")
        print(f"    Creating 8 row videos...")

        # Initialize writers for all 8 rows
        for row_idx in range(self.num_rows):
            video_path = os.path.join(output_video_dir, f"row_{row_idx:01d}_video.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                video_path,
                fourcc,
                self.target_fps,
                (w, h),
            )

            if not writer.isOpened():
                raise RuntimeError(f"Failed to create VideoWriter for row {row_idx}")

            writers[row_idx] = (writer, video_path)
            row_videos[row_idx] = video_path

        # Write rows to videos
        for frame_idx, frame_dir_name in enumerate(frame_dirs):
            frame_dir = os.path.join(rows_output_dir, frame_dir_name)

            for row_idx in range(self.num_rows):
                row_path = os.path.join(frame_dir, f"row_{row_idx:01d}.png")

                if not os.path.exists(row_path):
                    print(f"    Warning: Missing {row_path}")
                    continue

                row_image = cv2.imread(row_path)
                if row_image is None:
                    print(f"    Warning: Could not read {row_path}")
                    continue

                writer = writers[row_idx][0]
                writer.write(row_image)

            if (frame_idx + 1) % max(1, len(frame_dirs) // 5) == 0:
                print(f"    Progress: {frame_idx + 1}/{len(frame_dirs)} frames written")

        # Release all writers and re-encode with ffmpeg
        for row_idx in range(self.num_rows):
            writer, temp_path = writers[row_idx]
            writer.release()

            # Re-encode with ffmpeg for compatibility
            final_path = temp_path.replace(".mp4", "_final.mp4")
            
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", temp_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-loglevel", "error",
                final_path,
            ]

            try:
                subprocess.run(ffmpeg_cmd, check=True)
                os.remove(temp_path)
                row_videos[row_idx] = final_path
            except Exception as e:
                print(f"    Warning: ffmpeg re-encoding failed for row {row_idx}: {e}")
                # Keep original if ffmpeg fails
                os.rename(temp_path, temp_path.replace("_final", ""))

        print(f"  ✓ Row videos created")

        return row_videos

    def batch_process_videos_with_rows(
        self,
        video_ids: List[str] = None,
        step4_1_output_dir: str = "Backend/video/output/step4_2_direct",
    ) -> Dict[str, Dict]:
        """
        Process all battle_log videos and create both frames and row videos.

        :param video_ids: List of video IDs. If None, auto-detect.
        :param step4_1_output_dir: Step 4.1 output directory
        :return: Dict mapping video_id -> processing_summary
        """
        if video_ids is None:
            if os.path.exists(step4_1_output_dir):
                video_ids = sorted([d for d in os.listdir(step4_1_output_dir)
                                  if os.path.isdir(os.path.join(step4_1_output_dir, d))])
            else:
                print(f"Error: Step 4.1 directory not found: {step4_1_output_dir}")
                return {}

        results = {}
        
        for video_id in video_ids:
            try:
                battle_log_dir = os.path.join(step4_1_output_dir, video_id, "battle_log_filtered")
                
                if not os.path.exists(battle_log_dir):
                    print(f"[Warning] Battle log directory not found: {battle_log_dir}")
                    continue

                rows_output_dir = os.path.join(step4_1_output_dir, video_id, "battle_log_rows")
                
                summary = self.process_video_frames(
                    video_id=video_id,
                    battle_log_frames_dir=battle_log_dir,
                    output_dir=rows_output_dir
                )
                
                # Create row videos
                row_videos = self.create_row_videos(
                    video_id=video_id,
                    rows_output_dir=rows_output_dir,
                    output_video_dir=os.path.join(step4_1_output_dir, video_id, "row_videos")
                )
                
                summary["row_videos"] = row_videos
                results[video_id] = summary
                print()
                
            except Exception as e:
                print(f"[Error] Failed to process video {video_id}: {e}\n")
                continue
        
        return results
