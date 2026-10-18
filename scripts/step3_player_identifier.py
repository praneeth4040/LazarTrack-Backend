"""
Runner script for Step 3: Player Identification using Vision Language Model.

Reads frame_0000.png from Backend/video/output/step2/*/team_slots/frames/
Outputs player mapping JSONs to Backend/video/output/step3/*/players.json
"""

import glob
import os
import sys

from app.video.player_identifier import PlayerIdentifier, VisionLanguageModel


import argparse
import glob
import os
import sys

from app.video.player_identifier import PlayerIdentifier, VisionLanguageModel


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Step 3: Player Identification via VL Model")
    parser.add_argument("--step2", default=os.path.join(base_dir, "video", "output", "step2"), help="Step 2 output directory")
    parser.add_argument("--out", default=os.path.join(base_dir, "video", "output", "step3"), help="Step 3 output directory")
    parser.add_argument("--video", default=None, help="Specific video ID or directory (e.g. 1)")
    args = parser.parse_args()

    step2_dir = os.path.abspath(args.step2)
    step3_dir = os.path.abspath(args.out)
    os.makedirs(step3_dir, exist_ok=True)

    if args.video:
        target_dir = os.path.join(step2_dir, str(args.video))
        if os.path.isdir(target_dir):
            video_dirs = [target_dir]
        else:
            print(f"Error: Step 2 output for video '{args.video}' not found in {step2_dir}")
            sys.exit(1)
    else:
        video_dirs = sorted([d for d in glob.glob(os.path.join(step2_dir, "*")) if os.path.isdir(d)])

    if not video_dirs:
        print(f"No Step 2 video output directories found in {step2_dir}")
        sys.exit(1)

    print(
        f"Starting Step 3 player identification for {len(video_dirs)} video(s) -> {step3_dir}"
    )

    try:
        vl_model = VisionLanguageModel()
        print(f"✓ VL Model initialized: {vl_model.model}")
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    identifier = PlayerIdentifier(vl_model=vl_model, output_dir=step3_dir)

    for v_dir in video_dirs:
        try:
            identifier.process_video_step2_output(v_dir)
        except Exception as e:
            print(f"✗ Error processing {v_dir}: {e}")
            continue

    vl_model.close()
    print(
        f"\n[Step 3 Finished] Player slot mappings extracted and stored in {step3_dir}"
    )


if __name__ == "__main__":
    main()
