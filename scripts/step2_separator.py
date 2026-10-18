"""
Runner script for Step 2: Unified 3-Way Region Separation (Map, Team Slots, Battle Log Feed).
Fuzzy matches 'BATTLE LOG' header on frame 0 to dynamically separate streams.
Outputs separated videos and frames to Backend/video/output/step2/
"""

import argparse
import glob
import os
import sys
import easyocr
from app.video.separator import RegionSeparator

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Step 2: Unified 3-Way Region Separation")
    parser.add_argument("--step1", default=os.path.join(base_dir, "video", "output", "step1"), help="Step 1 normalized output directory")
    parser.add_argument("--out", default=os.path.join(base_dir, "video", "output", "step2"), help="Step 2 output directory")
    parser.add_argument("--video", default=None, help="Specific video folder or ID to process (e.g. 1)")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for output videos")
    parser.add_argument("--save-frames", action="store_true", default=True, help="Save individual frames")
    args = parser.parse_args()

    step1_dir = os.path.abspath(args.step1)
    output_dir = os.path.abspath(args.out)
    os.makedirs(output_dir, exist_ok=True)

    if args.video:
        target_dir = os.path.join(step1_dir, str(args.video))
        if os.path.isdir(target_dir):
            video_dirs = [target_dir]
        else:
            print(f"Error: Step 1 output for video '{args.video}' not found in {step1_dir}")
            sys.exit(1)
    else:
        video_dirs = sorted([d for d in glob.glob(os.path.join(step1_dir, "*")) if os.path.isdir(d)])

    if not video_dirs:
        print(f"No Step 1 video output directories found in {step1_dir}")
        sys.exit(1)

    print(f"Starting Step 2 unified 3-way region separation for {len(video_dirs)} video(s) -> {output_dir}")

    reader = easyocr.Reader(["en"])
    separator = RegionSeparator(
        reader=reader,
        canvas_width=1920,
        canvas_height=1080,
        split_x=960,
        output_dir=output_dir,
    )

    for v_dir in video_dirs:
        separator.process_normalized_video(v_dir, fps=args.fps, save_frames=args.save_frames)

    print(f"\n[Step 2 Finished] Map, Team Slots, and Battle Log Feed videos stored in {output_dir}")

if __name__ == "__main__":
    main()
