#!/usr/bin/env python3
"""
STEP 6: pack event crops from step 5 into group images (8 rows per image) for the VL model.

INPUT (from step 5), per video:  <step5>/<video_id>/step5.json  and  crops/*.png

OUTPUT, per video:               <out>/<video_id>/group_000.png, group_001.png, ...
                                 <out>/<video_id>/manifest.json   (which event is in which slot)
                                 <out>/all_videos_manifest.json

Each group image has up to 8 rows. Every row shows one event crop, with its event id
printed in a left margin ("#12") so the model's answers can be matched back to events.
The last image is padded with empty rows so every image has exactly 8 rows.

USAGE
  python step6.py --step5 /path/step5 --out /path/step6
  python step6.py --step5 ... --out ... --videos 1 3 --rows 8
"""

import argparse, json, os
import cv2, numpy as np

ROW_H = 62            # every row is resized to this height
CROP_X = (0, 820)     # keep this part of the 960 px wide row (the log box ends near x=795)
GUTTER = 0            # left margin that holds the event id (removed as per user request)
GAP = 4               # gap between rows
BG = (30, 24, 20)     # image background (BGR)

def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())

def best_crop(vdir, ev):
    """Pick the sharpest of the event's saved crops."""
    best = None
    for f in ev["crops"]:
        im = cv2.imread(f"{vdir}/{f}")
        if im is None:
            continue
        sh = sharpness(im)
        if best is None or sh > best[0]:
            best = (sh, f, im)
    return best

def make_row(label, im):
    w = CROP_X[1] - CROP_X[0]
    row = np.full((ROW_H, GUTTER + w, 3), BG, np.uint8)
    if im is not None:
        im = cv2.resize(im, (960, ROW_H))[:, CROP_X[0]:CROP_X[1]]
        row[:, GUTTER:] = im
    # Label removed as per user request
    # cv2.putText(row, label, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return row

def process(vdir, vout, rows_per_image):
    os.makedirs(vout, exist_ok=True)
    
    data = json.load(open(f"{vdir}/step5.json"))
    events = data["events"]
    
    picked = []
    for ev in events:
        b = best_crop(vdir, ev)
        if b is not None:
            picked.append((ev["id"], b[1], b[2]))
    
    manifest = []
    for g, start in enumerate(range(0, len(picked), rows_per_image)):
        chunk = picked[start:start + rows_per_image]
        rows, slots = [], []
        
        for slot in range(rows_per_image):
            if slot < len(chunk):
                eid, f, im = chunk[slot]
                rows.append(make_row(f"#{eid}", im))
                slots.append({"slot": slot, "event_id": eid, "crop": f})
            else:
                rows.append(make_row("", None))
                slots.append({"slot": slot, "event_id": None, "crop": None})
        
        gap = np.full((GAP, rows[0].shape[1], 3), BG, np.uint8)
        stacked = []
        for r in rows:
            stacked += [r, gap]
        
        img = np.vstack(stacked[:-1])
        name = f"group_{g:03d}.png"
        cv2.imwrite(f"{vout}/{name}", img)
        manifest.append({"image": name, "slots": slots})
    
    json.dump(manifest, open(f"{vout}/manifest.json", "w"), indent=1)
    
    return {"events": len(picked), "images": len(manifest), "rows_per_image": rows_per_image}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--rows", type=int, default=8)
    a = ap.parse_args()
    
    vids = a.videos or sorted(d for d in os.listdir(a.step5)
                              if os.path.isfile(f"{a.step5}/{d}/step5.json"))
    
    allm = {}
    for v in vids:
        s = process(f"{a.step5}/{v}", f"{a.out}/{v}", a.rows)
        allm[v] = s
        print(f"video {v}: {s['events']} events -> {s['images']} images of {s['rows_per_image']} rows")
    
    os.makedirs(a.out, exist_ok=True)
    json.dump(allm, open(f"{a.out}/all_videos_manifest.json", "w"), indent=1)

if __name__ == "__main__":
    main()
