#!/usr/bin/env python3
"""
STEP 5: find every battle-log event, follow it down the log, save crops.

INPUT (from step 4), per video:
    <step4>/<video_id>/row_<r>_frames/row_<r>_<frame>.png     r = 0..7, row 0 = top
  - every PNG is one row of one frame (about 960x62), full width, not resized
  - all 8 folders contain the same frame numbers
  - frames come from the same fps (about 5 fps was tested)

OUTPUT, per video:   <out>/<video_id>/step5.json, crops/*.png, overview.png
        and          <out>/all_videos_summary.json

USAGE
  python step5.py --step4 /path/step4 --out /path/step5
  python step5.py --step4 ... --out ... --videos 2 --expected 2=21
"""

import argparse, glob, json, os, re
import cv2, numpy as np

SIG_X = (60, 260)     # icon + timestamp area (does not scroll). 960 px wide rows.
STABLE_MSE = 3.0      # frame is "settled" if it matches the next frame (0-255 pixel scale)
AMBIG = 0.6           # best/second-best shift error above this -> flag
CROPS = 3             # crops saved per event
NEW_COST = 700.0      # cost of explaining a row as a brand-new event (between a good match ~100-300 and a wrong one ~1000+)
OCC_DEFAULT = 12.0    # used only if auto calibration fails (0-255 scale)

def mse(a, b):
    h = min(a.shape[0], b.shape[0])      # step 4 rows are 63 px (rows 0-2) or 62 px tall
    return float(((a[:h].astype(np.float32) - b[:h].astype(np.float32)) ** 2).mean())

def frame_no(p):
    return int(re.findall(r"(\d+)\.png$", os.path.basename(p))[-1])

def load_video(vdir):
    maps = []
    for r in range(8):
        files = glob.glob(f"{vdir}/row_{r}_frames/*.png")
        if not files:
            raise SystemExit(f"no frames in {vdir}/row_{r}_frames")
        maps.append({frame_no(f): f for f in files})
    
    common = sorted(set.intersection(*[set(m) for m in maps]))
    warn = None
    if any(len(m) != len(common) for m in maps):
        warn = f"row folders differ in frame count: {[len(m) for m in maps]}; using {len(common)} common frames"
        print("WARNING:", warn)
    
    paths = [[maps[r][fn] for r in range(8)] for fn in common]
    sig = np.zeros((len(common), 8), dtype=object)
    std = np.zeros((len(common), 8), dtype=np.float32)
    
    for t, row in enumerate(paths):
        for r, p in enumerate(row):
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)[:, SIG_X[0]:SIG_X[1]]
            sig[t, r] = g
            std[t, r] = float(g.astype(np.float32).std())
    
    return common, paths, sig, std, warn

def auto_occ_threshold(std):
    """Otsu split between 'empty row' and 'row with an event' pixel spread."""
    v = np.clip(std.ravel(), 0, 100)
    hist, edges = np.histogram(v, bins=200, range=(0, 100))
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p); mu = np.cumsum(p * edges[:-1]); mt = mu[-1]
    sb = (mt * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    th = float(edges[int(np.argmax(sb)) + 1])
    return min(max(th, 6.0), 25.0)

def diff(a, b):
    return float(np.mean([mse(x, y) for x, y in zip(a, b)]))

def mse_tol(a, b, tol=2):
    """Like mse() but allows a small vertical misalignment (rows are cut at slightly
    different offsets: 63 px vs 62 px tall), so a correct match is not penalised for it."""
    h = min(a.shape[0], b.shape[0])
    af, bf = a[:h].astype(np.float32), b[:h].astype(np.float32)
    best = 1e18
    for dy in range(-tol, tol + 1):
        x, y = (af[dy:], bf[:h - dy]) if dy >= 0 else (af[:h + dy], bf[-dy:])
        best = min(best, float(((x - y) ** 2).mean()))
    return best

def shift_scores(old, new, new_occ):
    """Error for every possible shift s (lower is better). Rows j >= s must match
    old row j-s (empty rows must match empty rows too); rows j < s are the new
    events and must be occupied. Returns a sorted list of (error, s)."""
    out = []
    for s in range(9):
        tot = sum(mse_tol(new[j], old[j - s]) for j in range(s, 8))
        tot += sum(NEW_COST if new_occ[j] else 2500.0 for j in range(min(s, 8)))
        out.append((tot / 8.0, s))
    return sorted(out)

def best_shift(old, new, old_occ, new_occ):
    if not any(old_occ):
        return int(sum(new_occ)), 0.0
    
    c = shift_scores(old, new, new_occ)
    ratio = c[0][0] / max(c[1][0], 1e-6)
    return c[0][1], ratio

def process(vdir, vout):
    os.makedirs(vout + "/crops", exist_ok=True)
    
    ids, paths, sig, std, warn = load_video(vdir)
    n = len(ids)
    th = auto_occ_threshold(std)
    occ = std > th
    
    sigl = [list(sig[t]) for t in range(n)]
    stable = [t for t in range(n - 1) if diff(sigl[t], sigl[t + 1]) < STABLE_MSE] + [n - 1]
    kind = ["moving"] * n
    events, review = [], []
    
    def new_event(f):
        events.append({"id": len(events), "arrival_frame": f, "exit_frame": None, "obs": []})
        return len(events) - 1
    
    slots = [None] * 8
    
    def observe(t):
        for j, e in enumerate(slots):
            if e is not None:
                events[e]["obs"].append((t, j))
    
    def fill_untracked(t):
        """Safety net: a visible row with no event assigned is a missed event."""
        for j in range(7, -1, -1):
            if occ[t][j] and slots[j] is None:
                slots[j] = new_event(ids[t])
                events[slots[j]]["late"] = True
    
    last = stable[0]; kind[last] = "state"
    for j in range(7, -1, -1):
        if occ[last][j]:
            slots[j] = new_event(ids[last])
    observe(last)
    
    for t in stable[1:]:
        if diff(sigl[last], sigl[t]) < STABLE_MSE:
            kind[t] = "redundant"; fill_untracked(t); observe(t); last = t; continue
        
        kind[t] = "state"
        
        if not any(occ[last]):
            slots = [None] * 8
        
        s, ratio = best_shift(sigl[last], sigl[t], occ[last], occ[t])
        
        if ratio > AMBIG:
            review.append({"from_frame": ids[last], "to_frame": ids[t], "shift": s, "ratio": round(ratio, 2)})
        
        for j in range(7, 7 - s, -1):
            if j >= 0 and slots[j] is not None:
                events[slots[j]]["exit_frame"] = ids[last]
        
        moved = [None] * 8
        for j in range(s, 8):
            moved[j] = slots[j - s]
        
        for j in range(min(s, 8) - 1, -1, -1):
            if occ[t][j]:
                moved[j] = new_event(ids[last + 1])
        
        slots = moved
        fill_untracked(t)
        observe(t); last = t
    
    # ---- drop phantoms: "events" whose row was empty most of the time (pop-ups, overlays) ----
    def occ_frac(e):
        return sum(1 for t, j in e["obs"] if occ[t][j]) / max(len(e["obs"]), 1)
    
    rejected = [{"arrival_frame": e["arrival_frame"], "occupied_fraction": round(occ_frac(e), 2)}
                for e in events if not e["obs"] or occ_frac(e) < 0.5]
    
    kept = [e for e in events if e["obs"] and occ_frac(e) >= 0.5]
    
    for new_id, e in enumerate(kept):
        e["id"] = new_id
    
    # ---- sharp crops (full width rows) ----
    result = []
    for e in kept:
        ob = e["obs"]
        files = []
        for k, idx in enumerate(np.array_split(np.arange(len(ob)), CROPS)):
            if len(idx) == 0:
                continue
            
            best = None
            for i in idx:
                t, j = ob[i]
                im = cv2.imread(paths[t][j])
                sh = float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
                if best is None or sh > best[0]:
                    best = (sh, t, j, im)
            
            _, t, j, im = best
            name = f"crops/ev_{e['id']:03d}_{k}_f{ids[t]}_r{j}.png"
            cv2.imwrite(f"{vout}/{name}", im)
            files.append(name)
        
        rows = []
        for _, j in ob:
            if not rows or rows[-1] != j:
                rows.append(j)
        
        result.append({"id": e["id"], "arrival_frame": e["arrival_frame"], "exit_frame": e["exit_frame"],
                       "row_path": rows, "crops": files, "late_detected": bool(e.get("late"))})
    
    exited = sum(1 for r in result if r["exit_frame"] is not None)
    on_screen = len(result) - exited
    final_occ = int(sum(occ[stable[-1]]))
    
    checks = {
        "arrivals_equal_exited_plus_on_screen": len(result) == exited + on_screen,
        # independent check: the log never empties, so rows on screen at the end = min(events, 8)
        "final_rows_equal_min_events_8": final_occ == min(len(result), 8),
    }
    
    summary = {"frames": n, "first_frame": ids[0], "last_frame": ids[-1], "occupied_threshold_0_255": round(th, 1),
               "moving": kind.count("moving"), "redundant": kind.count("redundant"), "state": kind.count("state"),
               "events": len(result), "exited": exited, "on_screen_at_end": on_screen,
               "late_detected": sum(1 for r in result if r["late_detected"]),
               "rejected_phantoms": rejected,
               "occupied_rows_in_final_frame": final_occ, "checks": checks,
               "ambiguous": review, "warning": warn}
    
    json.dump({"summary": summary, "frame_kind": dict(zip(map(str, ids), kind)), "events": result},
              open(vout + "/step5.json", "w"), indent=1)
    
    # ---- overview image: one line per event, for checking against a hand count ----
    rows_img = []
    for r in result:
        c = cv2.imread(f"{vout}/{r['crops'][len(r['crops']) // 2]}")
        c = cv2.resize(c, (960, 62))
        lab = np.full((20, 960, 3), 245, np.uint8)
        st = "on screen at end" if r["exit_frame"] is None else f"pushed out at frame {r['exit_frame']}"
        cv2.putText(lab, f"Event {r['id']}  entered frame {r['arrival_frame']}  {st}  rows {r['row_path']}",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        rows_img += [lab, c]
    
    if rows_img:
        cv2.imwrite(vout + "/overview.png", np.vstack(rows_img))
    
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--expected", default="", help="hand counts, e.g. 2=21,3=30")
    a = ap.parse_args()
    
    exp = dict(x.split("=") for x in a.expected.split(",") if x)
    vids = a.videos or sorted(d for d in os.listdir(a.step4) if os.path.isdir(f"{a.step4}/{d}"))
    
    allsum, ok = {}, True
    for v in vids:
        s = process(f"{a.step4}/{v}", f"{a.out}/{v}")
        allsum[v] = s
        print(f"\nvideo {v}: {s['events']} events ({s['exited']} exited, {s['on_screen_at_end']} on screen), "
              f"frames {s['frames']}, moving/redundant/state {s['moving']}/{s['redundant']}/{s['state']}, "
              f"occupied threshold {s['occupied_threshold_0_255']}, ambiguous {len(s['ambiguous'])}, late-detected {s['late_detected']}, phantoms rejected {len(s['rejected_phantoms'])}")
        
        for k, val in s["checks"].items():
            if not val:
                ok = False; print(f"  CHECK FAILED: {k}")
        
        if v in exp:
            good = int(exp[v]) == s["events"]
            ok &= good
            print(f"  hand count {exp[v]} vs found {s['events']}: {'PASS' if good else 'FAIL'}")
    
    os.makedirs(a.out, exist_ok=True)
    json.dump(allsum, open(f"{a.out}/all_videos_summary.json", "w"), indent=1)
    
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED, review the lines above")

if __name__ == "__main__":
    main()
