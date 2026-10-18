"""
Runner script for Step 1: Video Normalization on all input videos.
Reads videos from Backend/video/input/* and outputs normalized streams and metadata to Backend/video/output/step1/
"""

import argparse
import glob
import os
import sys
from app.video.normalizer import VideoNormalizer

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Step 1: Video Normalization")
    parser.add_argument("--input", default=os.path.join(base_dir, "video", "input"), help="Input directory or file")
    parser.add_argument("--output", default=os.path.join(base_dir, "video", "output", "step1"), help="Output directory")
    parser.add_argument("--video", default=None, help="Specific video filename or ID (e.g. 1 or 1.mp4)")
    parser.add_argument("--save-frames", action="store_true", default=True, help="Save sample frames as images")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(args.input):
        video_files = [os.path.abspath(args.input)]
        input_dir = os.path.dirname(os.path.abspath(args.input))
    elif args.video:
        if os.path.isabs(args.video) and os.path.isfile(args.video):
            video_files = [args.video]
        elif os.path.isfile(os.path.join(input_dir, args.video)):
            video_files = [os.path.join(input_dir, args.video)]
        elif os.path.isfile(os.path.join(input_dir, f"{args.video}.mp4")):
            video_files = [os.path.join(input_dir, f"{args.video}.mp4")]
        else:
            print(f"Error: Specified video not found: {args.video} in {input_dir}")
            sys.exit(1)
    else:
        video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))

    if not video_files:
        print(f"No video files found in {input_dir}")
        sys.exit(1)

    print(f"Starting Step 1 normalization on {len(video_files)} video(s) -> {output_dir}")

    normalizer = VideoNormalizer(
        target_width=1920,
        target_height=1080,
        target_fps=5.0,
        output_dir=output_dir,
    )

    for video_path in video_files:
        normalizer.process_and_save_video(video_path, save_frames=args.save_frames)

    print(f"\n[Step 1 Finished] Normalized videos stored in {output_dir}")

if __name__ == "__main__":
    main()
