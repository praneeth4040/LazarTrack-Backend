"""
Run full Zone Tracker Pipeline (Steps 1-8) for all 6 videos.
"""
import sys, os, time, shutil, json, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run_all_pipeline.log", mode="w"),
    ],
)
log = logging.getLogger("run_all")

VIDEOS = [
    {"id": "1", "mp4": "video/input/1.mp4", "map": "assets/maps/bermuda.jpeg"},
    {"id": "2", "mp4": "video/input/2.mp4", "map": "assets/maps/kalahari.jpg"},
    {"id": "3", "mp4": "video/input/3.mp4", "map": "assets/maps/nexterra.png"},
    {"id": "4", "mp4": "video/input/4.mp4", "map": "assets/maps/solara.jpg"},
    {"id": "5", "mp4": "video/input/5.mp4", "map": "assets/maps/bermuda.jpeg"},
    {"id": "6", "mp4": "video/input/6.mp4", "map": "assets/maps/nexterra.png"},
]

DEBUG_DIR = Path("video/output/assets_debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

from app.player_path.pipeline import run_player_path_pipeline

results = {}

for v in VIDEOS:
    vid_id  = v["id"]
    mp4     = v["mp4"]
    ref_map = v["map"]
    map_name = Path(ref_map).stem

    print(f"\n{'='*60}", flush=True)
    print(f"VIDEO {vid_id} — {mp4}  ->  {ref_map}", flush=True)
    print(f"{'='*60}", flush=True)

    if not Path(mp4).exists():
        print(f"  [SKIP] {mp4} not found", flush=True)
        results[vid_id] = {"success": False, "reason": "video not found"}
        continue

    out_dir = f"video/output/zone/{vid_id}"
    # Clean previous run
    if Path(out_dir).exists():
        shutil.rmtree(out_dir)

    t0 = time.time()
    try:
        summary = run_player_path_pipeline(
            video_path=mp4,
            video_id=vid_id,
            output_base_dir=out_dir,
            reference_map_path=ref_map,
            save_frames=False,
            cleanup_intermediate=True,
        )
        elapsed = time.time() - t0
        n_zones = len(summary.get("zones", []))
        print(f"  Done in {elapsed:.0f}s — {n_zones} zones detected", flush=True)
        results[vid_id] = {"success": True, "elapsed": round(elapsed, 1), "zones": n_zones}

        # Copy debug overlays
        overlay_src = Path(out_dir) / "step4" / "alignment_overlay.png"
        if overlay_src.exists():
            shutil.copy(overlay_src, DEBUG_DIR / f"alignment_overlay_vid{vid_id}_{map_name}.png")

        zones_map_src = Path(out_dir) / "zones_map.png"
        if zones_map_src.exists():
            shutil.copy(zones_map_src, DEBUG_DIR / f"zones_map_vid{vid_id}_{map_name}.png")

        for z in summary.get("zones", []):
            idx = z.get("index", z.get("phase", "?"))
            cx, cy, r = z.get("cx", 0), z.get("cy", 0), z.get("r", 0)
            t_s, t_e = z.get("t_start", 0), z.get("t_end", 0)
            print(f"    Zone {idx}: cx={cx:.0f} cy={cy:.0f} r={r:.0f}  [{t_s:.0f}s -> {t_e:.0f}s]", flush=True)

    except Exception as e:
        elapsed = time.time() - t0
        import traceback; traceback.print_exc()
        print(f"  FAILED after {elapsed:.0f}s: {e}", flush=True)
        results[vid_id] = {"success": False, "elapsed": round(elapsed, 1), "error": str(e)}

print(f"\n{'='*60}", flush=True)
print("FINAL SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
for vid_id, r in results.items():
    vv = next(x for x in VIDEOS if x["id"] == vid_id)
    status = "OK" if r.get("success") else "FAIL"
    detail = f"{r.get('zones','?')} zones  {r.get('elapsed','?')}s" if r.get("success") else r.get("error", r.get("reason",""))
    print(f"  Vid {vid_id} ({Path(vv['map']).stem:10s})  {status}  {detail}", flush=True)

json.dump(results, open("run_all_pipeline_results.json", "w"), indent=2)
print("\nDone. Results -> run_all_pipeline_results.json", flush=True)
