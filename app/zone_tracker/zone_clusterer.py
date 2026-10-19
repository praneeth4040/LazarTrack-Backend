"""
Step 7 - Zone clusterer  (v3)
Input : zone_observations.json (step 6)
Output: zones.json

v3 key addition — CONTAINMENT ENFORCEMENT:
  Each zone N+1 must be geometrically INSIDE zone N.
    dist(centers) + r_inner <= r_outer * margin
  Zones whose center is outside all accepted zones are discarded as
  false positives (road arcs, ziplines, UI rings, etc.).

v2 features retained:
  * cluster by (cx,cy,r) with temporal-gap bridging
  * lit-boundary plateau detection and matching
  * plateau promotion for missing intermediate zones
  * monotonic radius ordering
"""
import argparse, json, math, os
import numpy as np


def cluster_rings(obs, tol, min_obs, min_n=150, min_cov=25, rlim=(15, 800)):
    """Cluster ring observations into distinct zone candidates."""
    rings = [
        (r["cx"], r["cy"], r["r"], o["t"], r["n"], r.get("cov", 0))
        for o in obs for r in o["rings"]
        if r["n"] >= min_n and r.get("cov", 0) >= min_cov and rlim[0] < r["r"] < rlim[1]
    ]
    cl = []
    for x in sorted(rings, key=lambda x: -x[4]):
        for c in cl:
            if (np.hypot(x[0] - c["m"][0], x[1] - c["m"][1]) < tol
                    and abs(x[2] - c["m"][2]) < tol):
                c["o"].append(x)
                c["m"] = np.median([o[:3] for o in c["o"]], 0)
                break
        else:
            cl.append({"o": [x], "m": np.array(x[:3])})

    cl = [c for c in cl if len(c["o"]) >= min_obs]

    # temporal gap bridging
    merged = True
    while merged:
        merged = False
        new_cl, used = [], set()
        for i, ci in enumerate(cl):
            if i in used:
                continue
            for j in range(i + 1, len(cl)):
                if j in used:
                    continue
                cj = cl[j]
                d_xy = np.hypot(ci["m"][0] - cj["m"][0], ci["m"][1] - cj["m"][1])
                d_r  = abs(ci["m"][2] - cj["m"][2])
                t_gap = min(
                    abs(min(o[3] for o in ci["o"]) - max(o[3] for o in cj["o"])),
                    abs(min(o[3] for o in cj["o"]) - max(o[3] for o in ci["o"])),
                )
                if d_xy < tol * 1.5 and d_r < tol * 1.5 and t_gap < 20.0:
                    ci["o"].extend(cj["o"])
                    ci["m"] = np.median([o[:3] for o in ci["o"]], 0)
                    used.add(j)
                    merged = True
            new_cl.append(ci)
        cl = new_cl

    cl.sort(key=lambda c: -c["m"][2])   # biggest first = chronological
    return cl


def find_plateaus(obs, win, std_tol=1.5, join_tol=3.0, min_cov=20):
    """Detect stable stretches in lit-boundary observations."""
    lit = [
        (o["t"], o["lit"]["cx"], o["lit"]["cy"], o["lit"]["r"])
        for o in obs if o["lit"] and o["lit"].get("cov", 0) >= min_cov
    ]
    out, k = [], 0
    while k < len(lit) - win + 1:
        w = np.array([l[1:] for l in lit[k:k + win]])
        if w.std(0).max() < std_tol:
            e = k + win
            while e < len(lit) and np.abs(np.array(lit[e][1:]) - w.mean(0)).max() < join_tol:
                e += 1
            seg = np.array([l[1:] for l in lit[k:e]])
            out.append((lit[k][0], lit[e - 1][0], np.median(seg, 0)))
            k = e
        else:
            k += 1
    return out


def _is_contained(icx, icy, ir, ocx, ocy, or_, tolerance_px=3.0):
    """True if inner circle (icx,icy,ir) is inside outer circle, allowing
    up to tolerance_px for tangent zones due to sub-pixel registration noise."""
    dist = np.hypot(icx - ocx, icy - ocy)
    return (dist + ir) <= (or_ + tolerance_px)


def _containment_filter(zones, tolerance_px=3.0, min_obs=15, max_spread=3.0, scale=1.0):
    """Enforce strict sequential containment with resolution scaling:
    Zone 1 is the primary large safe zone.
    Zone 2 must be strictly inside Zone 1.
    Zone 3 must be strictly inside Zone 2.
    Zone 4 must be strictly inside Zone 3, etc.
    Any candidate circle that is not strictly inside the immediately preceding zone
    or has high spatial jitter / insufficient support is rejected.
    """
    if not zones:
        return zones

    tol_px = tolerance_px * scale
    spread_limit = max_spread * scale

    # Filter out noisy, unstable detections (roads, terrain contours, track lines)
    cands = [
        z for z in zones
        if z.get("n_obs", 0) >= min_obs and max(z.get("spread_px", [0])) <= spread_limit
    ]
    if not cands:
        # Fallback to general zones if none passed strict filter
        cands = [z for z in zones if z.get("n_obs", 0) >= 10]
    if not cands:
        cands = list(zones)

    # Prioritize higher-confidence candidates when candidates have similar radius
    filtered_cands = []
    for c in sorted(cands, key=lambda x: -x.get("n_obs", 0)):
        if not any(abs(c["r"] - f["r"]) / max(c["r"], f["r"]) < 0.12 for f in filtered_cands):
            filtered_cands.append(c)
    filtered_cands.sort(key=lambda c: -c["r"])

    # Zone 1: pick largest candidate with strong support
    z1_pool = [c for c in filtered_cands if c["r"] > 250.0 * scale]
    z1 = max(z1_pool, key=lambda c: c.get("n_obs", 0)) if z1_pool else filtered_cands[0]

    accepted = [z1]

    for z in filtered_cands:
        if z in accepted:
            continue
        prev = accepted[-1]
        # Radius must be noticeably smaller (at most 88% of previous zone radius)
        if z["r"] >= prev["r"] * 0.88:
            continue
        # Must be strictly inside the immediately preceding zone
        if _is_contained(z["cx"], z["cy"], z["r"],
                         prev["cx"], prev["cy"], prev["r"], tol_px):
            accepted.append(z)

    return accepted


def _promote_plateaus(zones, extra_plateaus, tol, tolerance_px=3.0):
    """Promote unmatched plateaus that sit between two existing zone radii
    AND are geometrically inside the surrounding zone."""
    if not extra_plateaus or not zones:
        return

    zone_radii = sorted([z["r"] for z in zones], reverse=True)
    promoted = []

    for ep in list(extra_plateaus):
        if "Closing state" in ep.get("name", ""):
            continue
        r, cx, cy = ep["r"], ep["cx"], ep["cy"]
        t0, t1 = ep.get("plateau_s", [0, 0])
        candidate = False

        for i in range(len(zone_radii) - 1):
            r_big, r_small = zone_radii[i], zone_radii[i + 1]
            if r_small < r < r_big:
                outer = next((z for z in zones if abs(z["r"] - r_big) < tol), None)
                inner = next((z for z in zones if abs(z["r"] - r_small) < tol), None)
                if outer and _is_contained(cx, cy, r,
                                            outer["cx"], outer["cy"], outer["r"], tolerance_px):
                    # Must fit temporally between outer and inner
                    if inner and t0 > inner.get("first_seen_s", 9999) + 5.0:
                        continue
                    candidate = True
                    break

        if not candidate and r < zone_radii[-1] * 0.8 and r > 15:
            smallest = min(zones, key=lambda z: z["r"])
            if _is_contained(cx, cy, r,
                              smallest["cx"], smallest["cy"], smallest["r"], tolerance_px):
                if t0 >= smallest.get("first_seen_s", 0) - 5.0:
                    candidate = True

        if candidate:
            promoted.append(ep)

    for p in promoted:
        t0, t1 = p.get("plateau_s", [0, 0])
        zones.append({
            "name": "Zone (promoted)",
            "cx": p["cx"], "cy": p["cy"], "r": p["r"],
            "first_seen_s": t0, "last_seen_s": t1,
            "current_zone_s": [t0, t1],
            "n_obs": 0, "spread_px": [0.0, 0.0, 0.0],
            "confidence": "low",
        })
        extra_plateaus.remove(p)


def _enforce_monotonic_radius(zones):
    """Sort by descending radius and re-number sequentially."""
    zones.sort(key=lambda z: -z["r"])
    for i, z in enumerate(zones, 1):
        z["name"]  = f"Zone {i}"
        z["index"] = i


def run(obs_json, out_json, tol=12.0, min_seconds=2.0, plateau_seconds=2.0, map_dim=None):
    """Main clustering entry point with resolution scaling."""
    d   = json.load(open(obs_json))
    obs = d["observations"]
    dt  = d["step"] / d["fps"]
    t_end = obs[-1]["t"]

    if map_dim is None:
        coords = [r["cx"] for o in obs for r in o.get("rings", [])] + [r["cy"] for o in obs for r in o.get("rings", [])]
        max_c = max(coords) if coords else 1024.0
        map_dim = max(1024.0, float(max_c))

    scale = map_dim / 1024.0
    tol_scaled = tol * scale
    rlim_scaled = (15.0 * scale, 0.95 * map_dim)

    cl    = cluster_rings(
        obs,
        tol=tol_scaled,
        min_obs=max(3, math.ceil(min_seconds / dt)),
        min_n=80,
        min_cov=15,
        rlim=rlim_scaled,
    )
    plats = find_plateaus(obs, win=max(3, math.ceil(plateau_seconds / dt)))

    zones = []
    for n, c in enumerate(cl, 1):
        ts  = [o[3] for o in c["o"]]
        est = [c["m"]]
        plateau = None
        for t0, t1, m in plats:
            if np.hypot(*(m[:2] - c["m"][:2])) < tol_scaled and abs(m[2] - c["m"][2]) < tol_scaled:
                est.append(m)
                plateau = plateau or (t0, t1)
        v      = np.mean(est, 0)
        spread = np.std([o[:3] for o in c["o"]], 0)
        zones.append({
            "name":            f"Zone {n}",
            "cx":              float(v[0]),
            "cy":              float(v[1]),
            "r":               float(v[2]),
            "first_seen_s":    float(min(ts)),
            "last_seen_s":     float(max(ts)),
            "current_zone_s":  [float(plateau[0]), float(plateau[1])] if plateau else None,
            "n_obs":           len(c["o"]),
            "spread_px":       [float(x) for x in spread],
            "confidence":      "high" if (len(c["o"]) >= 3 * math.ceil(min_seconds / dt)
                                          and spread.max() < 5 * scale) else "medium",
        })

    # ── v3: CONTAINMENT FILTER ──────────────────────────────────────────────
    zones = _containment_filter(zones, tolerance_px=3.0, min_obs=15, max_spread=3.0, scale=scale)

    extra = []
    for t0, t1, m in plats:
        if all(np.hypot(*(m[:2] - [z["cx"], z["cy"]])) > tol_scaled
               or abs(m[2] - z["r"]) > tol_scaled for z in zones):
            extra.append({
                "name": ("Closing state (end of replay)"
                         if t1 >= t_end - 2 * dt - 1 else f"Plateau {t0:.0f}s"),
                "cx": float(m[0]), "cy": float(m[1]), "r": float(m[2]),
                "plateau_s": [float(t0), float(t1)],
            })

    _promote_plateaus(zones, extra, tol_scaled, tolerance_px=3.0 * scale)
    _enforce_monotonic_radius(zones)

    # Add t_start / t_end aliases for pipeline compatibility
    for z in zones:
        z["t_start"] = z.get("first_seen_s", 0)
        z["t_end"]   = z.get("last_seen_s", 0)

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump({"zones": zones, "extra": extra}, open(out_json, "w"), indent=2)

    for z in zones + extra:
        print(f'[zone_clusterer] {z["name"]}: '
              f'centre=({z["cx"]:.1f},{z["cy"]:.1f}) r={z["r"]:.1f}')
    if len(zones) < 2:
        print("[zone_clusterer] WARNING: fewer than 2 zones found - check registration / thresholds")
    return zones, extra


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs",         default="zone_observations.json")
    ap.add_argument("--out",         default="zones.json")
    ap.add_argument("--tol",         type=float, default=12.0)
    ap.add_argument("--min-seconds", type=float, default=2.0)
    a = ap.parse_args()
    run(a.obs, a.out, a.tol, a.min_seconds)
