import asyncio
import os
import json
from httpx import AsyncClient, ASGITransport
from app.main import app

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "img", "input")
IMAGE_PATH = os.path.join(IMAGE_DIR, "image1.png")
if not os.path.exists(IMAGE_PATH):
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "img", "image1.png")

BASE_URL = "http://test"
API = "/api/v1"


async def test_backend():
    print("=" * 70)
    print("TESTING LAZARTRACK /pipeline/match_stats ENDPOINTS")
    print("=" * 70)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE_URL) as client:

        # 1. Health check
        print(f"\n1. Health Check ({API}/health)...")
        r = await client.get(API + "/health")
        print(f"   Status: {r.status_code}")
        print(f"   Body:   {json.dumps(r.json(), indent=2)}")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"

        if not os.path.exists(IMAGE_PATH):
            print(f"\n Sample image not found at {IMAGE_PATH} — skipping upload tests.")
            return

        with open(IMAGE_PATH, "rb") as f:
            img_bytes = f.read()
        fname = os.path.basename(IMAGE_PATH)
        ftype = "image/png" if fname.lower().endswith(".png") else "image/jpeg"

        # 2. Main pipeline endpoint
        print(f"\n2. Main Pipeline ({API}/pipeline/match_stats) — {fname}...")
        print("   (this may take 10–30 seconds on first run while EasyOCR model loads)")
        r = await client.post(
            API + "/pipeline/match_stats",
            files={"file": (fname, img_bytes, ftype)},
        )
        print(f"   Status: {r.status_code}")
        if r.status_code != 200:
            print(f"   Error: {r.text[:1500]}")
            return
        body = r.json()
        print(f"   pipeline_id:   {body.get('pipeline_id')}")
        print(f"   pipeline_name: {body.get('pipeline_name')}")
        print(f"   confidence:    {body.get('confidence')}")
        data = body.get("data", {})
        print(f"   map:           {data.get('map')}")
        print(f"   rank:          {data.get('rank')}")
        print(f"   total_teams:   {data.get('total_teams')}")
        players = data.get("players", [])
        print(f"   players:       {len(players)}")
        for idx, p in enumerate(players, 1):
            print(f"     #{idx} {p.get('inGameName'):<18} "
                  f"K={p.get('kills')} A={p.get('assists')} "
                  f"Dmg={p.get('damage'):<5} Rev={p.get('revival')} "
                  f"Knock={p.get('knockedDown')} HSR={p.get('headShotRate')}%")
        durations = data.get("metadata", {}).get("step_durations", {})
        total_dur = sum(durations.values()) if durations else 0.0
        print(f"   step durations (s): step1={durations.get('step1_normalization')}, "
              f"step2={durations.get('step2_smart_roi')}, "
              f"step3={durations.get('step3_section_split')}, "
              f"step4={durations.get('step4_column_slice')}, "
              f"step5={durations.get('step5_per_cell_ocr')}, total≈{round(total_dur, 2)}")

        # 3. Debug endpoint
        print(f"\n3. Debug Pipeline ({API}/pipeline/match_stats/debug)...")
        r2 = await client.post(
            API + "/pipeline/match_stats/debug",
            files={"file": (fname, img_bytes, ftype)},
        )
        print(f"   Status: {r2.status_code}")
        if r2.status_code != 200:
            print(f"   Error: {r2.text[:1500]}")
            return
        d = r2.json()
        steps = d.get("steps", [])
        print(f"   debug steps returned: {len(steps)}")
        for s in steps:
            step_num = s.get("step")
            step_name = s.get("name")
            step_dur = s.get("duration_seconds")
            image_keys = []
            if "image" in s and s["image"]:
                image_keys.append("image (base64)")
            if "images" in s:
                for k in s["images"].keys():
                    image_keys.append(f"images.{k} (base64)")
            if "column_images" in s:
                for k in s["column_images"].keys():
                    image_keys.append(f"col_img:{k} (base64)")
            extras = f"  | assets: {', '.join(image_keys)}" if image_keys else ""
            if step_num == 5:
                extras += f"  | players debug rows: {len(s.get('players', []))}"
            print(f"     Step {step_num}: {step_name:<38} {step_dur}s{extras}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_backend())
