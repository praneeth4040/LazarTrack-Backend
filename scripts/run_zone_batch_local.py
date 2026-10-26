"""One-off local runner: process all test videos through the 8-step zone pipeline.

Equivalent to POST /api/v1/zone/process for each video (same function the
endpoint calls), with the per-video reference maps the dev runner used.
Results land in video/output/zone/{id}/ and are viewable via
GET /api/v1/zone/{id}/visual on the running backend.
"""
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEOS = [
    {"id": "1", "mp4": "video/input/1.mp4", "map": "assets/maps/bermuda.jpeg"},
    {"id": "2", "mp4": "video/input/2.mp4", "map": "assets/maps/kalahari.jpg"},
    {"id": "3", "mp4": "video/input/3.mp4", "map": "assets/maps/nexterra.png"},
    {"id": "4", "mp4": "video/input/4.mp4", "map": "assets/maps/solara.jpg"},
    {"id": "5", "mp4": "video/input/5.mp4", "map": "assets/maps/bermuda.jpeg"},
    {"id": "6", "mp4": "video/input/6.mp4", "map": "assets/maps/nexterra.png"},
]


def main() -> None:
    # MUST stay inside this guard: the zone detector's spawn-based process pool
    # re-imports __main__ in each worker — without the guard the whole batch
    # would recursively re-execute.
    from app.zone_tracker.pipeline import run_player_path_pipeline

    backend_root = Path(__file__).resolve().parent.parent
    summary = []

    for v in VIDEOS:
        video_path = backend_root / v["mp4"]
        ref_map = backend_root / v["map"]
        if not video_path.exists():
            print(f"[batch] SKIP {v['id']}: {video_path} missing")
            continue
        if not ref_map.exists():
            print(f"[batch] SKIP {v['id']}: map {ref_map} missing")
            continue

        print(f"\n[batch] ===== video {v['id']} ({v['mp4']}, map={v['map']}) =====", flush=True)
        t0 = time.time()
        try:
            result = run_player_path_pipeline(
                video_path=str(video_path),
                video_id=v["id"],
                output_base_dir=str(backend_root / "video" / "output" / "zone" / v["id"]),
                reference_map_path=str(ref_map),
                cleanup_intermediate=False,
            )
            elapsed = round(time.time() - t0, 1)
            zones = result.get("zones", [])
            print(f"[batch] DONE video {v['id']}: {len(zones)} zones in {elapsed}s", flush=True)
            summary.append({
                "video_id": v["id"],
                "map": Path(v["map"]).stem,
                "zones": len(zones),
                "elapsed_s": elapsed,
                "ok": True,
                "zones_detail": [
                    {k: z.get(k) for k in ("name", "cx", "cy", "r", "confidence")}
                    for z in zones
                ],
            })
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            print(f"[batch] FAILED video {v['id']} after {elapsed}s: {e}", flush=True)
            logging.exception("zone pipeline failed")
            summary.append({"video_id": v["id"], "ok": False, "error": str(e), "elapsed_s": elapsed})

    print("\n[batch] ===== SUMMARY =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
