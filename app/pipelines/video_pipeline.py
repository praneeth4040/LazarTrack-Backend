#!/usr/bin/env python3
"""
Complete Video Battle Log Extraction Pipeline

Runs all 7 steps using the dedicated pipeline scripts in sequence.
Preserves deliverables in the session export directory for production serving.
"""

import os
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger("app.pipelines.video_pipeline")


class VideoPipelineError(Exception):
    """Custom exception for pipeline errors"""
    pass


def run_complete_video_pipeline(
    video_path: str,
    video_id: str,
    output_base_dir: str,
    cleanup_intermediate: bool = True,
    run_step7: bool = False,
) -> Dict[str, Any]:
    """
    Run video battle log extraction pipeline up to Step 6 by default.
    Step 7 (Vision Language model extraction) is deferred unless run_step7=True.
    
    Args:
        video_path: Path to input video file
        video_id: Unique session identifier for this video (e.g. UUID)
        output_base_dir: Directory where final session artifacts are stored
        cleanup_intermediate: If True, remove intermediate frame folders after copying final artifacts
        run_step7: If False (default), stop at Step 6 group image packing. If True, run Step 7.
        
    Returns:
        Dictionary containing group images, map video, team slots, metadata, and optional events.
    """
    log.info(f"Starting targeted pipeline for video_id={video_id} (path={video_path})")
    
    backend_dir = Path(__file__).parent.parent.parent.resolve()
    video_input_dir = backend_dir / "video" / "input"
    video_output_base = backend_dir / "video" / "output"
    
    video_input_dir.mkdir(parents=True, exist_ok=True)
    video_output_base.mkdir(parents=True, exist_ok=True)
    
    # Destination session export folder
    session_dir = Path(output_base_dir).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Use video_id as the primary identifier (alphanumeric supported)
    # Also find an integer video_num if required by legacy scripts
    existing_videos = [int(f.stem) for f in video_input_dir.glob("*.mp4") if f.stem.isdigit()]
    video_num = max(existing_videos) + 1 if existing_videos else 1
    
    input_video_path = video_input_dir / f"{video_num}.mp4"
    shutil.copy(video_path, input_video_path)
    log.info(f"[{video_id}] Staged video as {input_video_path.name}")
    
    pipeline_env = {**os.environ, "PYTHONPATH": str(backend_dir)}
    
    try:
        # Step 1: Normalization (1920x1080 @ 5 FPS, H.264)
        log.info(f"[{video_id}] Running Step 1: Normalization")
        step1_script = backend_dir / "scripts" / "step1_normalizer.py"
        res1 = subprocess.run([
            sys.executable, str(step1_script),
            "--input", str(video_input_dir),
            "--output", str(video_output_base / "step1"),
            "--video", str(video_num),
        ], capture_output=True, text=True, cwd=str(backend_dir), env=pipeline_env)
        if res1.returncode != 0:
            raise VideoPipelineError(f"Step 1 failed: {res1.stderr or res1.stdout}")
        
        # Step 2: Battle log & Region separation
        log.info(f"[{video_id}] Running Step 2: Region separation")
        step2_script = backend_dir / "scripts" / "step2_separator.py"
        res2 = subprocess.run([
            sys.executable, str(step2_script),
            "--step1", str(video_output_base / "step1"),
            "--out", str(video_output_base / "step2"),
            "--video", str(video_num),
        ], capture_output=True, text=True, cwd=str(backend_dir), env=pipeline_env)
        if res2.returncode != 0:
            raise VideoPipelineError(f"Step 2 failed: {res2.stderr or res2.stdout}")
        
        # Step 3: Player identification (Roster via Qwen VL)
        log.info(f"[{video_id}] Running Step 3: Player identification")
        step3_script = backend_dir / "scripts" / "step3_player_identifier.py"
        res3 = subprocess.run([
            sys.executable, str(step3_script),
            "--step2", str(video_output_base / "step2"),
            "--out", str(video_output_base / "step3"),
            "--video", str(video_num),
        ], capture_output=True, text=True, cwd=str(backend_dir), env=pipeline_env)
        if res3.returncode != 0:
            raise VideoPipelineError(f"Step 3 failed: {res3.stderr or res3.stdout}")
        
        # Step 4: Row division (8 horizontal slices)
        log.info(f"[{video_id}] Running Step 4: Row division")
        step4_script = backend_dir / "scripts" / "step4_row_divider.py"
        res4 = subprocess.run([
            sys.executable, str(step4_script),
            "--step2", str(video_output_base / "step2"),
            "--out", str(video_output_base / "step4"),
            "--video", str(video_num),
        ], capture_output=True, text=True, cwd=str(backend_dir), env=pipeline_env)
        if res4.returncode != 0:
            raise VideoPipelineError(f"Step 4 failed: {res4.stderr or res4.stdout}")
        
        # Step 5: Event tracking & Deduplication
        log.info(f"[{video_id}] Running Step 5: Event tracking & deduplication")
        step5_script = backend_dir / "scripts" / "step5_event_tracker.py"
        res5 = subprocess.run([
            sys.executable, str(step5_script),
            "--step4", str(video_output_base / "step4"),
            "--out", str(video_output_base / "step5"),
            "--videos", str(video_num),
        ], capture_output=True, text=True, cwd=str(backend_dir), env=pipeline_env)
        if res5.returncode != 0:
            raise VideoPipelineError(f"Step 5 failed: {res5.stderr or res5.stdout}")
        
        # Step 6: Group images packing (8 events per image)
        log.info(f"[{video_id}] Running Step 6: Group images packing")
        step6_script = backend_dir / "scripts" / "step6_group_packer.py"
        res6 = subprocess.run([
            sys.executable, str(step6_script),
            "--step5", str(video_output_base / "step5"),
            "--out", str(video_output_base / "step6"),
            "--videos", str(video_num),
            "--rows", "8",
        ], capture_output=True, text=True, cwd=str(backend_dir), env=pipeline_env)
        if res6.returncode != 0:
            raise VideoPipelineError(f"Step 6 failed: {res6.stderr or res6.stdout}")
        
        # Step 7: Vision model extraction & roster reconciliation (runs ONLY when run_step7=True)
        if run_step7:
            log.info(f"[{video_id}] Running Step 7: Vision model extraction")
            step7_script = backend_dir / "scripts" / "step7_vision_extractor.py"
            res7 = subprocess.run([
                sys.executable, str(step7_script),
                "--step6", str(video_output_base / "step6"),
                "--out", str(video_output_base / "step7"),
                "--step3", str(video_output_base / "step3"),
                "--videos", str(video_num),
                "--workers", "3",
            ], capture_output=True, text=True, cwd=str(backend_dir), env=pipeline_env)
            # Step 7 warnings are non-fatal (retries are handled internally)
            if res7.returncode != 0:
                log.warning(f"[{video_id}] Step 7 warnings: {(res7.stderr or res7.stdout)[:500]}")
        else:
            log.info(f"[{video_id}] Step 7 deferred (pipeline stopping at Step 6 as requested)")
        
        # -------------------------------------------------------------
        # Collect & Persist Deliverables to session_dir
        # -------------------------------------------------------------
        step1_dir = video_output_base / "step1" / str(video_num)
        step2_dir = video_output_base / "step2" / str(video_num)
        step3_dir = video_output_base / "step3" / str(video_num)
        step6_dir = video_output_base / "step6" / str(video_num)
        step7_dir = video_output_base / "step7" / str(video_num)
        
        # 1. Map Video — use the Step 2 separated map region (left 960px) NOT the full Step 1 video
        final_map_video = None
        step2_map_video = step2_dir / "map" / "map_path_video.mp4"
        if step2_map_video.exists():
            dest_map = session_dir / "map_video.mp4"
            shutil.copy(step2_map_video, dest_map)
            final_map_video = str(dest_map)
            log.info(f"[{video_id}] Collected map video from step2 ({step2_map_video.stat().st_size // 1024} KB)")
        else:
            # Fallback: step1 normalized (full frame) if step2 map not found
            if step1_dir.exists():
                for f in step1_dir.glob("*_normalized.mp4"):
                    dest_map = session_dir / "map_video.mp4"
                    shutil.copy(f, dest_map)
                    final_map_video = str(dest_map)
                    log.warning(f"[{video_id}] step2 map video not found, falling back to step1 full video")
                    break
        
        # 2. Team Image (roster frame)
        final_team_image = None
        source_team_frame = step2_dir / "team_slots" / "frames" / "frame_0000.png"
        if source_team_frame.exists():
            dest_team = session_dir / "team_image.png"
            shutil.copy(source_team_frame, dest_team)
            final_team_image = str(dest_team)
        
        # 3. Group Images
        final_group_images = []
        if step6_dir.exists():
            dest_group_dir = session_dir / "group_images"
            dest_group_dir.mkdir(parents=True, exist_ok=True)
            for g_img in sorted(step6_dir.glob("group_*.png")):
                dest_file = dest_group_dir / g_img.name
                shutil.copy(g_img, dest_file)
                final_group_images.append(str(dest_file))
            # Also copy manifest if present
            manifest_file = step6_dir / "manifest.json"
            if manifest_file.exists():
                shutil.copy(manifest_file, dest_group_dir / "manifest.json")
        
        # 4. Events JSON
        events = []
        final_events_file = session_dir / "events.json"
        if run_step7:
            source_events_file = step7_dir / "events.json"
            if source_events_file.exists():
                shutil.copy(source_events_file, final_events_file)
                with open(final_events_file, "r") as f:
                    events_data = json.load(f)
                    events = events_data.get("events", [])
            else:
                with open(final_events_file, "w") as f:
                    json.dump({"video_id": video_id, "events": []}, f, indent=2)
        else:
            if not final_events_file.exists():
                with open(final_events_file, "w") as f:
                    json.dump({"video_id": video_id, "events": [], "step7_status": "deferred"}, f, indent=2)
        
        # 5. Team Slots Roster
        players_data = {}
        source_players = step3_dir / "players.json"
        if source_players.exists():
            shutil.copy(source_players, session_dir / "players.json")
            with open(source_players, "r") as f:
                players_data = json.load(f)
        
        # 6. Cleanup intermediate heavy frame folders if requested
        # NOTE: step3 is intentionally excluded — its players.json is tiny and was
        # already used by step7; removing it retroactively causes review flags.
        if cleanup_intermediate:
            for step in ["step1", "step2", "step4", "step5", "step6", "step7"]:
                step_dir = video_output_base / step / str(video_num)
                if step_dir.exists():
                    shutil.rmtree(step_dir)
        
        # Cleanup uploaded temporary input
        if input_video_path.exists():
            input_video_path.unlink()
        
        return {
            "success": True,
            "video_id": video_id,
            "video_number": video_num,
            "events": events,
            "group_images": final_group_images,
            "map_video_path": final_map_video,
            "team_slots": players_data,
            "team_image": final_team_image,
            "events_file_path": str(final_events_file),
            "output_dir": str(session_dir),
            "metadata": {
                "event_count": len(events),
                "group_count": len(final_group_images),
                "player_count": len(players_data),
                "pipeline_steps": {
                    "step1": "completed",
                    "step2": "completed",
                    "step3": "completed",
                    "step4": "completed",
                    "step5": "completed",
                    "step6": "completed",
                    "step7": "completed" if run_step7 else "deferred",
                },
            },
        }
        
    except Exception as e:
        if input_video_path.exists():
            input_video_path.unlink()
        log.error(f"Pipeline failed for video_id={video_id}: {str(e)}", exc_info=True)
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
        }


def run_step7_on_session(session_dir_path: str, video_id: str) -> Dict[str, Any]:
    """
    Runs Step 7 on demand on an existing session (for admin portal / background jobs).
    Reads the group_images, manifest.json, and players.json already saved in the session directory.
    """
    backend_dir = Path(__file__).parent.parent.parent.resolve()
    session_dir = Path(session_dir_path).resolve()
    group_dir = session_dir / "group_images"
    players_file = session_dir / "players.json"

    if not group_dir.exists():
        raise VideoPipelineError(f"Group images directory not found in {session_dir}")

    from scripts.step7_vision_extractor import process_video
    from app.core.config import settings

    api_key = os.getenv("OPENROUTER_API_KEY", "") or getattr(settings, "OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not configured")

    log.info(f"[{video_id}] Running on-demand Step 7 extraction directly...")
    summary = process_video(
        video_id=video_id,
        vdir=str(group_dir),
        out_dir=str(session_dir),
        api_key=api_key,
        model="qwen/qwen3.7-flash",
        roster_dir=str(session_dir),
        workers=2,
        max_attempts=3,
        max_tokens=1500,
        force=True,
    )

    final_events_file = session_dir / "events.json"
    events = []
    if final_events_file.exists():
        with open(final_events_file, "r") as f:
            events_data = json.load(f)
            events = events_data.get("events", [])

    return {
        "success": True,
        "video_id": video_id,
        "events": events,
        "event_count": len(events),
        "summary": summary,
    }
