#!/usr/bin/env python3
"""
Step 4: Divide Battle Log into 8 Rows

Purpose: Split each battle log frame into 8 individual row frames.
Each row represents one line of the battle log.

Input: Step 2 output (battle log frames - 960×499)
Output: 8 rows per frame, organized by row (rows 0-7)

Structure:
  video/output/step4/
  ├── 1/
  │   ├── row_0_frames/
  │   ├── row_1_frames/
  │   ├── row_2_frames/
  │   ├── row_3_frames/
  │   ├── row_4_frames/
  │   ├── row_5_frames/
  │   ├── row_6_frames/
  │   └── row_7_frames/
  ├── 2/
  └── ... through video 6

Event Flow:
  - Row 0: Newest events (entry point)
  - Rows 1-6: Events aging as they scroll down
  - Row 7: Oldest events (exit point - data loss if not captured)
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Step4RowDivider:
    """Divide battle log frames into 8 rows."""
    
    def __init__(self):
        self.num_rows = 8
        self.frame_width = 960
        self.frame_height = 499
        self.row_height = self.frame_height // self.num_rows
    
    def divide_frame_into_rows(self, frame):
        """
        Divide a frame into 8 row frames.
        
        Handles non-evenly-divisible heights by distributing remainder.
        
        :param frame: Input frame (960×499)
        :return: List of 8 row frames (height varies by ~1 pixel per row)
        """
        frame_height = frame.shape[0]
        base_height = frame_height // self.num_rows
        remainder = frame_height % self.num_rows
        
        rows = []
        start_y = 0
        
        for i in range(self.num_rows):
            # Distribute remainder across first N rows
            row_h = base_height + (1 if i < remainder else 0)
            end_y = start_y + row_h
            row_frame = frame[start_y:end_y, :]
            rows.append(row_frame)
            start_y = end_y
        
        return rows
    
    def process_video(self, video_id, battle_log_frames_dir, output_base_dir):
        """
        Process all frames for a video.
        
        :param video_id: Video ID (1-6)
        :param battle_log_frames_dir: Directory with battle log frames
        :param output_base_dir: Output base directory
        :return: Processing summary
        """
        print(f"\n[Video {video_id}] Processing...")
        
        # Get all frames
        if not os.path.exists(battle_log_frames_dir):
            return {
                "status": "failed",
                "reason": f"Frame directory not found: {battle_log_frames_dir}",
                "video_id": video_id,
            }
        
        frame_files = sorted([
            f for f in os.listdir(battle_log_frames_dir)
            if f.endswith('.png')
        ])
        
        if not frame_files:
            return {
                "status": "failed",
                "reason": "No frames found",
                "video_id": video_id,
            }
        
        print(f"  Found {len(frame_files)} frames")
        
        # Create output directories for each row
        video_output_dir = os.path.join(output_base_dir, str(video_id))
        os.makedirs(video_output_dir, exist_ok=True)
        
        row_output_dirs = {}
        for row_idx in range(self.num_rows):
            row_dir = os.path.join(video_output_dir, f"row_{row_idx}_frames")
            os.makedirs(row_dir, exist_ok=True)
            row_output_dirs[row_idx] = row_dir
        
        # Process each frame
        row_frame_counts = {i: 0 for i in range(self.num_rows)}
        
        for frame_idx, frame_file in enumerate(frame_files):
            frame_path = os.path.join(battle_log_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                continue
            
            # Divide frame into rows
            rows = self.divide_frame_into_rows(frame)
            
            # Save each row
            for row_idx, row_frame in enumerate(rows):
                output_filename = f"row_{row_idx}_{frame_idx:06d}.png"
                output_path = os.path.join(row_output_dirs[row_idx], output_filename)
                cv2.imwrite(output_path, row_frame)
                row_frame_counts[row_idx] += 1
            
            if (frame_idx + 1) % max(1, len(frame_files) // 5) == 0:
                print(f"    Processed {frame_idx + 1}/{len(frame_files)} frames")
        
        # Print summary
        print(f"  ✓ Completed")
        total_frames = len(frame_files)
        for row_idx in range(self.num_rows):
            count = row_frame_counts[row_idx]
            print(f"    Row {row_idx}: {count} frames")
        
        return {
            "status": "success",
            "video_id": video_id,
            "total_frames": total_frames,
            "total_rows": self.num_rows,
            **{f"row_{i}_frames": row_frame_counts[i] for i in range(self.num_rows)},
        }


def main():
    """Run Step 4: Divide battle log frames into 8 rows."""
    import argparse
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Step 4: Divide Battle Log into 8 Rows")
    parser.add_argument("--step2", default=os.path.join(base_dir, "video", "output", "step2"), help="Step 2 output directory")
    parser.add_argument("--out", default=os.path.join(base_dir, "video", "output", "step4"), help="Step 4 output directory")
    parser.add_argument("--video", default=None, help="Specific video ID or directory name (e.g. 1)")
    parser.add_argument("--videos", nargs="*", default=None, help="List of video IDs to process")
    args = parser.parse_args()

    step2_input_base = os.path.abspath(args.step2)
    step4_output_base = os.path.abspath(args.out)
    
    print("\n" + "=" * 90)
    print("STEP 4: DIVIDE BATTLE LOG INTO 8 ROWS")
    print("=" * 90)
    print()
    print("Purpose: Split each battle log frame (960×499) into 8 rows (960×62 each)")
    print()
    print("Process:")
    print("  1. For each video:")
    print("     - Read all battle log frames from Step 2")
    print("     - Divide each frame horizontally into 8 equal rows")
    print("  2. Organize output:")
    print("     - Create row_0_frames/, row_1_frames/, ..., row_7_frames/")
    print("     - Save individual row frames as PNG images")
    print("  3. Structure:")
    print("     - Row 0: Newest events (entry point)")
    print("     - Row 7: Oldest events (exit point)")
    print()
    print("Event Flow Each Frame:")
    print("  - New event appears at Row 0")
    print("  - All events shift down one row")
    print("  - Event at Row 7 exits screen → data loss if not captured")
    print()
    print(f"Input:  {step2_input_base}")
    print(f"Output: {step4_output_base}")
    print("=" * 90)
    
    # Check if input directory exists
    if not os.path.exists(step2_input_base):
        print(f"\n❌ Input directory not found: {step2_input_base}")
        print("Please run Step 2 first.\n")
        return
    
    # Create output directory
    os.makedirs(step4_output_base, exist_ok=True)
    
    # Initialize row divider
    divider = Step4RowDivider()
    
    # Get video IDs
    if args.video:
        video_ids = [str(args.video)]
    elif args.videos:
        video_ids = [str(v) for v in args.videos]
    else:
        video_ids = sorted([
            d for d in os.listdir(step2_input_base)
            if os.path.isdir(os.path.join(step2_input_base, d)) and not d.startswith(".")
        ])
    
    print(f"\nFound {len(video_ids)} videos to process: {video_ids}\n")
    
    # Process all videos
    results = {}
    for video_id in video_ids:
        battle_log_frames_dir = os.path.join(step2_input_base, video_id, "battle_log", "frames")
        result = divider.process_video(
            video_id=video_id,
            battle_log_frames_dir=battle_log_frames_dir,
            output_base_dir=step4_output_base,
        )
        results[video_id] = result
    
    # Print summary
    print()
    print("=" * 90)
    print("STEP 4 COMPLETE: ROW DIVISION")
    print("=" * 90)
    print()
    
    successful = 0
    failed = 0
    total_frames = 0
    
    for video_id in sorted(results.keys()):
        result = results[video_id]
        
        if result.get("status") == "success":
            successful += 1
            frames = result.get("total_frames", 0)
            total_frames += frames
            
            print(f"Video {video_id}: ✓ {frames} frames → 8 rows")
        else:
            failed += 1
            print(f"Video {video_id}: ✗ Failed - {result.get('reason', 'Unknown error')}")
    
    print()
    print("=" * 90)
    print(f"Summary: {successful} successful, {failed} failed")
    print(f"Total frames processed: {total_frames}")
    print("=" * 90)
    
    # Save results to JSON
    results_file = os.path.join(step4_output_base, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {results_file}")
    
    # Create summary report
    summary = {
        "step": "4",
        "name": "Row Division",
        "purpose": "Divide battle log frames into 8 rows",
        "total_videos": len(results),
        "successful": successful,
        "failed": failed,
        "total_frames_processed": total_frames,
        "rows_per_frame": 8,
        "row_dimensions": {
            "width": 960,
            "height": 62,
        },
        "output_structure": {
            "location": "Backend/video/output/step4/",
            "format": "{video_id}/row_{row_idx}_frames/",
            "rows": "0-7 (0=newest entry, 7=oldest exit)",
        },
        "videos": results,
    }
    
    summary_file = os.path.join(step4_output_base, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved: {summary_file}")
    
    print()
    print("=" * 90)
    print("Directory structure created:")
    print(f"  {step4_output_base}/")
    for vid in sorted(results.keys()):
        if results[vid].get("status") == "success":
            print(f"  ├── {vid}/")
            for row in range(8):
                row_count = results[vid].get(f"row_{row}_frames", 0)
                print(f"  │   ├── row_{row}_frames/ ({row_count} frames)")
    
    print()
    print("=" * 90)
    print()
    print("✅ STEP 4 COMPLETE - Ready for Step 5\n")


if __name__ == "__main__":
    main()
