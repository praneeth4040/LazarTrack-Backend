"""
Zone Tracker Pipeline Orchestrator

Complete end-to-end pipeline for Zone Tracking across Free Fire maps:
- Step 1: Standardizes canvas to 1920x1080 at full native framerate.
- Step 2: Isolates the 960x1080 Map stream and extracts team roster frame.
- Step 3: Detects exact map boundary using UI anchors and crops clean map stream.
- Step 4: Aligns clean map with reference blank map, computes homography matrix.
- Step 5: Frame registrar tracks camera motion across frames (transforms.npz).
- Step 6: Zone detector extracts per-frame safe zone and shrink boundary circle observations.
- Step 7: Zone clusterer aggregates observations into discrete safe zone phases.
- Step 8: Zone renderer draws complete safe zone circles onto the reference base map.

Output organized into step1 through step8 directories.
"""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict
import cv2

from app.player_path.normalizer import PlayerPathNormalizer
from app.player_path.map_separator import PlayerPathMapSeparator
from app.player_path.map_cleaner import PlayerPathMapCleaner
from app.player_path.map_aligner import PlayerPathMapAligner
from app.player_path.frame_registrar import run as run_frame_registrar, RegistrarConfig
from app.player_path.zone_detector import run as run_zone_detector
from app.player_path.zone_clusterer import run as run_zone_clusterer
from app.player_path.zone_renderer import render_zones_map

log = logging.getLogger("app.player_path.pipeline")


class PlayerPathPipelineError(Exception):
    """Custom exception for player path pipeline errors."""
    pass


def run_player_path_pipeline(
    video_path: str,
    video_id: str,
    output_base_dir: str,
    reference_map_path: str = "assets/maps/bermuda.jpeg",
    save_frames: bool = False,
    cleanup_intermediate: bool = True,
    crop_mode: str = "full_map",
) -> Dict[str, Any]:
    """
    Execute full Zone Tracker Pipeline (Steps 1 to 8).

    Args:
        video_path: Absolute or relative path to the input video.
        video_id: Unique session identifier for the run.
        output_base_dir: Final destination directory for deliverables (e.g. video/output/zone/1).
        reference_map_path: Path to the blank reference map image (e.g. assets/maps/bermuda.jpeg).
        save_frames: Whether to dump intermediate PNG frames.
        cleanup_intermediate: Whether to delete temporary scratch folders after completion.
        crop_mode: 'full_map' (864x858) or 'inner_terrain' (770x755).

    Returns:
        Dictionary containing deliverables across step1 to step8.
    """
    log.info(f"[ZonePipeline] Starting pipeline for video_id={video_id} (path={video_path}, map={reference_map_path})")

    session_dir = Path(output_base_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    # Step-specific directories under output_base_dir
    step1_dir = session_dir / "step1"
    step2_dir = session_dir / "step2"
    step3_dir = session_dir / "step3"
    step4_dir = session_dir / "step4"
    step5_dir = session_dir / "step5"
    step6_dir = session_dir / "step6"
    step7_dir = session_dir / "step7"
    step8_dir = session_dir / "step8"

    for d in [step1_dir, step2_dir, step3_dir, step4_dir, step5_dir, step6_dir, step7_dir, step8_dir]:
        d.mkdir(parents=True, exist_ok=True)

    temp_intermediate = session_dir / "_temp"
    temp_intermediate.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Normalizer at full native FPS
        log.info(f"[{video_id}] Step 1: Normalizing canvas to 1920x1080 at full native FPS")
        normalizer = PlayerPathNormalizer(
            target_width=1920,
            target_height=1080,
            target_fps=None,
            output_dir=str(temp_intermediate / "step1"),
        )
        normalized_video_path, metadata = normalizer.process(video_path, save_frames=save_frames)
        final_norm_video = step1_dir / "normalized_video.mp4"
        shutil.copy(normalized_video_path, final_norm_video)
        with open(step1_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        # Step 2: Map Separator at full native FPS
        log.info(f"[{video_id}] Step 2: Separating 960x1080 map stream at full native FPS")
        separator = PlayerPathMapSeparator(
            split_x=960,
            output_dir=str(temp_intermediate / "step2"),
        )
        step2_result = separator.process(
            normalized_video_path=normalized_video_path,
            video_id=video_id,
            save_map_frames=save_frames,
        )
        final_raw_map = step2_dir / "map_video.mp4"
        shutil.copy(step2_result.map_video_path, final_raw_map)
        final_team_image = step2_dir / "team_image.png"
        shutil.copy(step2_result.team_image_path, final_team_image)
        with open(step2_dir / "region_bounds.json", "w") as f:
            json.dump(step2_result.region_bounds, f, indent=2)

        # Step 3: Anchor-based Map Boundary Detection & Clean Map Extraction
        log.info(f"[{video_id}] Step 3: Detecting anchors & cropping exact clean map stream")
        cleaner = PlayerPathMapCleaner(
            output_dir=str(temp_intermediate / "step3"),
        )
        clean_result = cleaner.process(
            map_video_path=step2_result.map_video_path,
            video_id=video_id,
            crop_mode=crop_mode,
        )
        final_clean_map = step3_dir / "clean_map_video.mp4"
        shutil.copy(clean_result.clean_map_video_path, final_clean_map)
        final_sample_frame = step3_dir / "clean_map_frame_0.png"
        shutil.copy(clean_result.sample_frame_path, final_sample_frame)
        final_bounds_json = step3_dir / "map_bounds.json"
        shutil.copy(clean_result.bounds_json_path, final_bounds_json)

        # Step 4: Map Alignment & Homography Projection Matrix
        log.info(f"[{video_id}] Step 4: Aligning clean map stream with reference map ({reference_map_path})")
        aligner = PlayerPathMapAligner(
            reference_map_path=reference_map_path,
            output_dir=str(temp_intermediate / "step4"),
        )
        align_result = aligner.align(
            clean_map_image_path=clean_result.sample_frame_path,
            video_id=video_id,
        )
        final_overlay = step4_dir / "alignment_overlay.png"
        shutil.copy(align_result.alignment_overlay_path, final_overlay)
        final_homography_json = step4_dir / "homography.json"
        shutil.copy(align_result.homography_json_path, final_homography_json)

        # Step 5: Frame Registrar (Camera motion registration)
        log.info(f"[{video_id}] Step 5: Registering per-frame camera transforms")
        run_frame_registrar(
            video=str(final_clean_map),
            homography_json=str(final_homography_json),
            out_dir=str(step5_dir),
            base_image=reference_map_path,
            cfg=RegistrarConfig(max_register_fps=10.0),
        )

        # Step 6: Zone Detector (Per-frame circle observations)
        log.info(f"[{video_id}] Step 6: Detecting per-frame zone observations")
        obs_json = step6_dir / "zone_observations.json"
        run_zone_detector(
            video=str(final_clean_map),
            transforms=str(step5_dir / "transforms.npz"),
            ui_mask=str(step5_dir / "ui_mask.png"),
            out_json=str(obs_json),
            step=12,
        )

        # Step 7: Zone Clusterer (Aggregate into discrete safe zones)
        log.info(f"[{video_id}] Step 7: Clustering observations into safe zone phases")
        zones_json = step7_dir / "zones.json"
        ref_im = cv2.imread(reference_map_path)
        ref_dim = max(ref_im.shape[:2]) if ref_im is not None else 1024.0
        zones, extra = run_zone_clusterer(
            obs_json=str(obs_json),
            out_json=str(zones_json),
            map_dim=ref_dim,
        )

        # Step 8: Zone Renderer (Render complete zones on base map)
        log.info(f"[{video_id}] Step 8: Rendering safe zones on base map")
        zones_map_png = step8_dir / "zones_map.png"
        render_zones_map(
            zones_json_path=str(zones_json),
            base_map_path=reference_map_path,
            out_png_path=str(zones_map_png),
        )

        # Root convenience copies
        shutil.copy(zones_json, session_dir / "zones.json")
        shutil.copy(zones_map_png, session_dir / "zones_map.png")

        summary = {
            "success": True,
            "pipeline": "zone_tracker",
            "video_id": video_id,
            "zones_count": len(zones),
            "zones_map_png": str(session_dir / "zones_map.png"),
            "zones_json": str(session_dir / "zones.json"),
            "reference_map": reference_map_path,
            "fps": clean_result.fps,
            "total_frames": clean_result.total_frames,
            "duration_seconds": metadata.duration_seconds,
            "resolution": clean_result.map_bounds["dimensions"],
            "artifacts": {
                "step1": {
                    "normalized_video": str(final_norm_video),
                    "metadata": str(step1_dir / "metadata.json"),
                },
                "step2": {
                    "map_video": str(final_raw_map),
                    "team_image": str(final_team_image),
                    "region_bounds": str(step2_dir / "region_bounds.json"),
                },
                "step3": {
                    "clean_map_video": str(final_clean_map),
                    "clean_map_frame_0": str(final_sample_frame),
                    "map_bounds": str(final_bounds_json),
                },
                "step4": {
                    "homography": str(final_homography_json),
                    "alignment_overlay": str(final_overlay),
                },
                "step5": {
                    "transforms": str(step5_dir / "transforms.npz"),
                    "registration": str(step5_dir / "registration.json"),
                    "ui_mask": str(step5_dir / "ui_mask.png"),
                },
                "step6": {
                    "zone_observations": str(obs_json),
                },
                "step7": {
                    "zones": str(zones_json),
                },
                "step8": {
                    "zones_map": str(zones_map_png),
                },
            },
            "zones": zones,
            "extra_plateaus": extra,
            "homography_matrix": align_result.homography_matrix,
            "alignment_metrics": asdict(align_result.metrics),
            "map_bounds": clean_result.map_bounds,
            "metadata": asdict(metadata),
        }

        with open(session_dir / "session.json", "w") as f:
            json.dump(summary, f, indent=2)

        log.info(
            f"[ZonePipeline] Completed video_id={video_id}: all 8 steps finished. "
            f"Detected {len(zones)} safe zones."
        )
        return summary

    except Exception as e:
        log.exception(f"[ZonePipeline] Pipeline execution failed: {e}")
        raise PlayerPathPipelineError(str(e))

    finally:
        if cleanup_intermediate and temp_intermediate.exists():
            shutil.rmtree(temp_intermediate, ignore_errors=True)
            log.info(f"[{video_id}] Cleaned temporary directory: {temp_intermediate}")


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
    ap = argparse.ArgumentParser(description="Run complete Zone Tracker Pipeline (Steps 1-8)")
    ap.add_argument("--video", required=True, help="Input replay video path")
    ap.add_argument("--id", default="1", help="Video / session ID")
    ap.add_argument("--out", default=None, help="Output directory (defaults to video/output/zone/<id>)")
    ap.add_argument("--base", default="assets/maps/bermuda.jpeg", help="Reference base map")
    ap.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate step files")
    args = ap.parse_args()

    out_dir = args.out or f"video/output/zone/{args.id}"

    run_player_path_pipeline(
        video_path=args.video,
        video_id=args.id,
        output_base_dir=out_dir,
        reference_map_path=args.base,
        cleanup_intermediate=not args.keep_intermediate,
    )


if __name__ == "__main__":
    main()
