"""
Step 6 - Zone detector (per-frame circle observations in BASE-MAP coordinates)

Inputs  : clean_map_video.mp4, transforms.npz (H(t): clean-frame px -> base-map px),
          ui_mask.png (static replay UI)
Output  : zone_observations.json

Two circle types are observed per sampled frame:
  ring : the thin semi-transparent white ring  = NEXT zone
  lit  : boundary between lit / darkened map   = CURRENT zone (darkening is applied outside it)
Both are fitted directly in base-map coordinates, so results from different
frames are directly comparable and can be clustered in step 7.

Lit detection needs an un-darkened reference of the map: frame `--baseline-frame`
(default 0, the full-island view before any zone is active).

v2 improvements
  • multi-scale top-hat (kernel 5, 9, 13)
  • lower ring threshold (20) and inlier floor (80)
  • up to 4 rings per frame
  • HSV pre-filter for ring-coloured pixels (low sat, high val)
  • adaptive lit-boundary ratio with Otsu fallback
  • Canny-based edge fallback for lit detection
"""
import argparse
import json
import os
import cv2
import numpy as np

try:
    from app.player_path.circle_fit import ransac_circle, coverage_deg
except ImportError:
    from circle_fit import ransac_circle, coverage_deg

# Multi-scale structuring elements for top-hat
RING_TOPHAT_KERNELS = [
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
]


# ------------------------------------------------------------------ io helpers
def load_transforms(path):
    z = np.load(path)
    key = next((k for k in ("H", "transforms", "H_t", "homographies", "Hs", "matrices") if k in z.files), z.files[0])
    H = np.asarray(z[key], dtype=np.float64)
    if H.ndim == 3 and H.shape[1:] == (2, 3):
        H = np.concatenate([H, np.tile([[[0, 0, 1.0]]], (len(H), 1, 1))], axis=1)
    assert H.ndim == 3 and H.shape[1:] == (3, 3), f"unexpected transform array {H.shape} (key '{key}')"
    return H


def load_valid_mask(path, shape, border=8, grow=7):
    """True where the map is usable. Handles either polarity of ui_mask.png
    (UI is assumed to be the minority of pixels)."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        m = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    else:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        if m.mean() > 0.5:
            m = ~m
    ui = cv2.dilate(m.astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0
    valid = ~ui
    valid[:border] = valid[-border:] = False
    valid[:, :border] = valid[:, -border:] = False
    return valid


def local_scale(H, w, h):
    """base-map px per frame px around the frame centre."""
    p = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2], [w / 2 + 1, h / 2], [w / 2, h / 2 + 1]]]), H)[0]
    d1 = p[1] - p[0]
    d2 = p[2] - p[0]
    return float(np.sqrt(abs(d1[0] * d2[1] - d1[1] * d2[0])))


def to_base(P, H):
    return cv2.perspectiveTransform(P.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2).astype(np.float64)


def overlay_mask(im, white_thr=212):
    """Pixels that are text / icons / path lines (saturated or near-white)."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    ov = ((hsv[..., 1] > 130) & (hsv[..., 2] > 120)) | (hsv[..., 2] > white_thr)
    return cv2.dilate(ov.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


# ------------------------------------------------------------------ detectors
def ring_points(im, valid, thr=20):
    """Multi-scale top-hat ring detection with HSV pre-filter.

    Returns array of (x, y) edge pixel coordinates (frame space).
    """
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g_blur = cv2.GaussianBlur(g, (0, 0), 0.8)

    # HSV pre-filter: ring pixels have low saturation, high value
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    ring_colour = (hsv[..., 1] < 80) & (hsv[..., 2] > 140)

    # Multi-scale top-hat — union the results
    combined = np.zeros(g.shape, dtype=np.uint8)
    for kernel in RING_TOPHAT_KERNELS:
        th = cv2.morphologyEx(g_blur, cv2.MORPH_TOPHAT, kernel)
        combined = np.maximum(combined, th)

    # Threshold
    c = (combined > thr)
    # Apply HSV ring-colour mask and valid mask
    c &= valid & ring_colour & ~overlay_mask(im)

    # Connected components — drop tiny noise
    n, lab, st, _ = cv2.connectedComponentsWithStats(c.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 15       # lowered from 30 to 15
    ys, xs = np.nonzero(keep[lab])
    return np.c_[xs, ys].astype(np.float64)


def find_rings(im, valid, H, s, rmin, rmax, maxn=4):
    """Fit up to *maxn* circles through detected ring edge points."""
    P = ring_points(im, valid)
    out = []
    if len(P) == 0:
        return out
    Pb = to_base(P, H)
    for _ in range(maxn):
        r = ransac_circle(Pb, rmin, rmax, tol=2.5 * s, iters=8000, min_inliers=80)
        if r is None:
            break
        c, inl = r
        cov = coverage_deg(Pb[inl], c)
        out.append({
            "cx": float(c[0]), "cy": float(c[1]), "r": float(c[2]),
            "n": int(inl.sum()), "cov": cov,
        })
        Pb = Pb[~inl]
        if len(Pb) < 60:           # lowered from 150
            break
    return out


def find_lit_boundary(im, valid, H, H0, base_im, base_valid, s, rmin, rmax):
    """Detect the lit/dark boundary circle (current zone)."""
    h, w = im.shape[:2]
    M = np.linalg.inv(H) @ H0                                          # baseline px -> this frame px
    b = cv2.warpPerspective(base_im, M, (w, h), flags=cv2.INTER_LINEAR)
    foot = cv2.warpPerspective(base_valid.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0
    b_bright = cv2.dilate((cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 2] > 200).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    mk = valid & foot & ~overlay_mask(im, 200) & ~b_bright
    mk = cv2.erode(mk.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mf = mk.astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(mf, (0, 0), 6), 1e-3)
    ratio = (cv2.GaussianBlur(g * mf, (0, 0), 6) / den) / np.maximum(cv2.GaussianBlur(gb * mf, (0, 0), 6) / den, 5)
    ok = den > 0.6

    # ---- adaptive lit threshold (Otsu on the ratio image within valid mask) ----
    ratio_u8 = np.clip(ratio * 128, 0, 255).astype(np.uint8)
    ratio_u8_valid = ratio_u8[ok]
    if len(ratio_u8_valid) > 100:
        otsu_val, _ = cv2.threshold(ratio_u8_valid, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        lit_thr = max(otsu_val / 128.0, 0.65)     # never go below 0.65
        lit_thr = min(lit_thr, 0.92)               # never go above 0.92
    else:
        lit_thr = 0.82                              # default

    lit = (ratio > lit_thr) & ok
    frac = float(lit[ok].mean()) if ok.any() else 0.0
    if frac < 0.03 or frac > 0.97:
        return None, frac

    # ---- primary: morphological gradient edge ----
    edge = cv2.morphologyEx(lit.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    edge &= ~(cv2.dilate((~ok).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    ys, xs = np.nonzero(edge)

    # ---- fallback: Canny edge on the ratio image ----
    if len(xs) < 40:
        ratio_blur = cv2.GaussianBlur((ratio * 255).astype(np.uint8), (5, 5), 1.0)
        canny = cv2.Canny(ratio_blur, 30, 90)
        canny &= ok.astype(np.uint8) * 255
        ys, xs = np.nonzero(canny > 0)

    if len(xs) < 40:
        return None, frac

    Pb = to_base(np.c_[xs, ys].astype(np.float64), H)
    r = ransac_circle(Pb, rmin, rmax, tol=4.0 * s, iters=6000, min_inliers=40)     # lowered from 60
    if r is None:
        return None, frac
    c, inl = r
    return {"cx": float(c[0]), "cy": float(c[1]), "r": float(c[2]), "n": int(inl.sum()), "cov": coverage_deg(Pb[inl], c)}, frac


import concurrent.futures

_WORKER_CTX = {}


def _init_worker(valid, base_im, base_valid, H0, rmin, rmax, w, h, fps):
    _WORKER_CTX["valid"] = valid
    _WORKER_CTX["base_im"] = base_im
    _WORKER_CTX["base_valid"] = base_valid
    _WORKER_CTX["H0"] = H0
    _WORKER_CTX["rmin"] = rmin
    _WORKER_CTX["rmax"] = rmax
    _WORKER_CTX["w"] = w
    _WORKER_CTX["h"] = h
    _WORKER_CTX["fps"] = fps


def _detect_frame(task):
    i, im, H = task
    valid = _WORKER_CTX["valid"]
    base_im = _WORKER_CTX["base_im"]
    base_valid = _WORKER_CTX["base_valid"]
    H0 = _WORKER_CTX["H0"]
    rmin = _WORKER_CTX["rmin"]
    rmax = _WORKER_CTX["rmax"]
    w = _WORKER_CTX["w"]
    h = _WORKER_CTX["h"]
    fps = _WORKER_CTX["fps"]

    s = local_scale(H, w, h)
    rec = {"frame": i, "t": round(i / fps, 3), "scale": round(s, 4)}
    rec["rings"] = find_rings(im, valid, H, s, rmin, rmax)
    rec["lit"], rec["lit_frac"] = find_lit_boundary(im, valid, H, H0, base_im, base_valid, s, rmin, rmax)
    return rec


# ------------------------------------------------------------------ main loop
def run(video, transforms, ui_mask, out_json, step=12, baseline_frame=0, rmin=20, rmax=None, log_every=20):
    H_all = load_transforms(transforms)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_vid != len(H_all):
        print(f"[warn] video has {n_vid} frames but transforms has {len(H_all)}; using the shorter one")
    n = min(n_vid, len(H_all))

    cap.set(cv2.CAP_PROP_POS_FRAMES, baseline_frame)
    ok, base_im = cap.read()
    assert ok, "cannot read baseline frame"
    h, w = base_im.shape[:2]
    valid = load_valid_mask(ui_mask, (h, w))
    base_valid = valid & ~overlay_mask(base_im, 200)
    H0 = H_all[baseline_frame]

    if rmax is None:
        pts_corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
        base_corners = cv2.perspectiveTransform(pts_corners, H0)[0]
        max_coord = max(float(np.abs(base_corners).max()), 1024.0)
        rmax = float(max(1200.0, 0.85 * max_coord))
        print(f"[zone_detector] Inferred dynamic rmax: {rmax:.1f} (from max base coordinate: {max_coord:.1f})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tasks, i = [], 0
    while i < n:
        if i % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, im = cap.read()
        if not ok:
            break
        tasks.append((i, im, H_all[i]))
        i += 1
    cap.release()

    workers = min(8, os.cpu_count() or 4)
    print(f"[zone_detector] Processing {len(tasks)} sampled frames using {workers} parallel workers...", flush=True)
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(valid, base_im, base_valid, H0, rmin, rmax, w, h, fps),
        ) as executor:
            obs = list(executor.map(_detect_frame, tasks, chunksize=4))
    except Exception as e:
        print(f"[zone_detector] Parallel execution fallback to sequential ({e})", flush=True)
        _init_worker(valid, base_im, base_valid, H0, rmin, rmax, w, h, fps)
        obs = [_detect_frame(t) for t in tasks]

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    json.dump({"fps": fps, "step": step, "n_frames": n, "observations": obs}, open(out_json, "w"), indent=2)
    print(f"[zone_detector] {len(obs)} sampled frames -> {out_json}")
    return obs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--ui-mask", required=True)
    ap.add_argument("--out", default="zone_observations.json")
    ap.add_argument("--step", type=int, default=12, help="analyse every N-th frame (12 @ 60fps = 5 samples/s)")
    ap.add_argument("--baseline-frame", type=int, default=0)
    ap.add_argument("--rmin", type=float, default=20)
    ap.add_argument("--rmax", type=float, default=1200)
    a = ap.parse_args()
    run(a.video, a.transforms, a.ui_mask, a.out, a.step, a.baseline_frame, a.rmin, a.rmax)
