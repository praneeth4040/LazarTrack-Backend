"""
Step 5 - Frame Registrar (camera motion tracker)          app/player_path/frame_registrar.py

Input   clean_map_video.mp4   cropped map stream (step 3), any FPS
        homography.json       frame 0 -> bermuda.jpeg (step 4)
Output  transforms.npz        H[N,3,3]  one frame->base-map homography for EVERY frame (full FPS)
        registration.json     summary + quality numbers (inliers, interpolation, drift)
        registration_check.png  frames warped over the base map, for a quick visual check

Why this step exists
    The replay camera pans and zooms all the time, so a pixel in frame 1000 is not the same place as the same
    pixel in frame 0. Every later step (finding players, building paths) needs "where on the island is this
    pixel?", i.e. one transform per frame.

How
    1. KEYFRAMES   register at most `max_register_fps` frames per second (SIFT is the slow part). At 60 FPS this
                   means every 6th frame; nothing is downsampled for later steps because of (4).
    2. CHAIN       match each keyframe to the previous one (SIFT + ratio test + RANSAC similarity). Chaining the
                   results gives A_k = keyframe k -> frame 0. If a match fails, fall back to older keyframes.
    3. ANCHOR      H_k = H0 @ A_k puts keyframe k on the base map, where H0 is the step-4 homography.
    4. INTERPOLATE the camera is smooth, so the frames between keyframes get their transform by interpolating
                   (log-scale, rotation, translation) of the two neighbouring keyframes. Result: a transform for all
                   N frames.
    5. VERIFY      (a) inlier counts, (b) drift check = match far-apart overlapping frames directly and compare with
                   what the chain predicts, (c) overlay image.

    A similarity (scale + rotation + translation) is used between frames because the replay camera does exactly
    that, and it is far more stable than a full homography on sparse, repetitive terrain.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np

SRC_KEY, SRC_FALLBACK, SRC_INTERP, SRC_HELD = 0, 1, 2, 3
SRC_NAMES = {SRC_KEY: "keyframe", SRC_FALLBACK: "keyframe_fallback", SRC_INTERP: "interpolated", SRC_HELD: "held"}


@dataclass
class RegistrarConfig:
    max_register_fps: float = 5.0           # keyframe rate cap (matches zone detector step=12 at 60fps)
    sift_features: int = 2000
    contrast_threshold: float = 0.02
    ratio: float = 0.75                     # Lowe ratio test
    ransac_px: float = 3.0
    min_inliers: int = 20
    min_inlier_ratio: float = 0.2
    fallback_back: tuple = (1, 2, 3, 5, 8)  # how many keyframes back to try if the previous one fails
    max_scale_step: float = 4.0             # reject a keyframe step that changes zoom by more than this factor
    cut_scale_step: float = 1.5             # a zoom jump bigger than this between neighbours = camera cut (no interpolation)
    cut_min_inliers: int = 40               # a cut must be backed by more inliers than a normal step
    ui_margin: int = 4                      # extra px ignored around the fixed UI
    drift_pairs: int = 30
    drift_min_gap_s: float = 5.0
    drift_min_overlap: float = 0.30
    seed: int = 0


# ----------------------------------------------------------------------------- homography input
def load_homography(path: str):
    """Read homography.json -> (H forward frame0->base, H inverse). Tolerates a few key spellings."""
    d = json.load(open(path))

    def find(keys):
        for k in keys:
            if k in d:
                v = d[k]
                if isinstance(v, dict):
                    v = v.get("matrix", v.get("H", v))
                return np.array(v, dtype=float).reshape(3, 3)
        return None

    H = find(["matrix_H_forward", "forward", "H_forward", "forward_homography", "homography", "H"])
    Hi = find(["matrix_H_inverse", "inverse", "H_inverse", "inverse_homography", "H_inv"])
    if H is None and Hi is None:
        raise KeyError(f"no 3x3 matrix found in {path}; keys are {list(d)}")
    if H is None:
        H = np.linalg.inv(Hi)
    if Hi is None:
        Hi = np.linalg.inv(H)
    scale = float(np.hypot(H[0, 0], H[1, 0]))
    if scale < 1.0 and float(np.hypot(Hi[0, 0], Hi[1, 0])) > 1.0:
        H, Hi = Hi, H
        print("note: homography.json 'forward' looked like base->frame; swapped")
    return H / H[2, 2], Hi / Hi[2, 2]


# ----------------------------------------------------------------------------- fixed UI mask
def compute_ui_mask(video: str, n: int = 60, grow: int = 6) -> np.ndarray:
    """255 = fixed interface (legend, counters, timeline) that must not be used for matching.
    Fixed UI stays sharp in the temporal MEDIAN of frames spread over the video; the moving map blurs away."""
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), min(n, max(total, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 8:
        raise ValueError("need at least 8 readable frames to build the UI mask")
    med = np.median(np.stack(frames), 0).astype(np.uint8)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(med.astype(np.float32), (0, 0), 1.0), cv2.CV_32F))
    sharp = (lap > max(6.0, 8.0 * float(np.median(lap)))).astype(np.uint8) * 255
    sharp = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    cnt, lab, st, _ = cv2.connectedComponentsWithStats(sharp)
    h, w = sharp.shape
    mask, bottom_top = np.zeros_like(sharp), None
    for j in range(1, cnt):
        x, y, bw, bh, area = st[j]
        if area < 250 or (bw > 0.8 * w and bh < 14):
            continue
        if y + bh / 2 > 0.75 * h:
            bottom_top = y if bottom_top is None else min(bottom_top, y)
            continue
        mask[y:y + bh, x:x + bw] = 255
    if bottom_top is not None:
        mask[max(bottom_top - grow, 0):, :] = 255
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))


# ----------------------------------------------------------------------------- matching
class Matcher:
    def __init__(self, cfg: RegistrarConfig, ui_mask):
        self.cfg = cfg
        self.sift = cv2.SIFT_create(nfeatures=cfg.sift_features, contrastThreshold=cfg.contrast_threshold)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
        self.clahe = cv2.createCLAHE(3.0, (8, 8))
        self.ui_mask = ui_mask
        self._fmask = None

    def _feature_mask(self, shape):
        if self._fmask is None or self._fmask.shape != shape[:2]:
            m = np.full(shape[:2], 255, np.uint8)
            if self.ui_mask is not None:
                u = self.ui_mask if self.ui_mask.shape == shape[:2] else cv2.resize(
                    self.ui_mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
                )
                m[u > 0] = 0
            self._fmask = m
        return self._fmask

    def features(self, frame):
        gray = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        kp, des = self.sift.detectAndCompute(gray, self._feature_mask(frame.shape))
        return (np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)), des

    def match(self, a, b):
        """similarity mapping frame A -> frame B as 3x3, plus (inliers, ratio); None if it fails"""
        (pa, da), (pb, db) = a, b
        if da is None or db is None or len(pa) < 8 or len(pb) < 8:
            return None
        pairs = self.flann.knnMatch(da, db, k=2)
        good = [m for p in pairs if len(p) == 2 for m, n in [p] if m.distance < self.cfg.ratio * n.distance]
        if len(good) < 8:
            return None
        P1 = pa[[g.queryIdx for g in good]]
        P2 = pb[[g.trainIdx for g in good]]
        M, inl = cv2.estimateAffinePartial2D(P1, P2, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.ransac_px)
        if M is None:
            return None
        n_inl = int(inl.sum())
        return np.vstack([M, [0, 0, 1]]), n_inl, n_inl / len(good)


def _scale(M):
    return float(np.hypot(M[0, 0], M[1, 0]))


# ----------------------------------------------------------------------------- stages 1-2: keyframe chain
def register_keyframes(video: str, cfg: RegistrarConfig, ui_mask, log=print):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / cfg.max_register_fps)))
    mt = Matcher(cfg, ui_mask)
    hist = deque(maxlen=max(cfg.fallback_back))
    kf_idx, kf_A, kf_src, kf_inl, kf_ratio = [], [], [], [], []
    raw = -1
    while True:
        if not cap.grab():
            break
        raw += 1
        if raw % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        feats = mt.features(frame)
        if not hist:
            A, src, inl, ratio = np.eye(3), SRC_KEY, -1, 1.0
        else:
            A = None
            for b in cfg.fallback_back:
                if b > len(hist):
                    break
                ref_feats, ref_A = hist[-b]
                r = mt.match(feats, ref_feats)
                if r is None:
                    continue
                M, inl, ratio = r
                cand = ref_A @ M
                step = _scale(cand) / _scale(ref_A)
                need = cfg.cut_min_inliers if not (1 / cfg.cut_scale_step <= step <= cfg.cut_scale_step) else cfg.min_inliers
                if inl >= need and ratio >= cfg.min_inlier_ratio and 1 / cfg.max_scale_step <= step <= cfg.max_scale_step:
                    A, src = cand, (SRC_KEY if b == 1 else SRC_FALLBACK)
                    break
            if A is None:
                A, src, inl, ratio = hist[-1][1], SRC_HELD, 0, 0.0
                kf_idx.append(raw); kf_A.append(A); kf_src.append(src); kf_inl.append(inl); kf_ratio.append(ratio)
                continue
        hist.append((feats, A))
        kf_idx.append(raw); kf_A.append(A); kf_src.append(src); kf_inl.append(inl); kf_ratio.append(ratio)
        if len(kf_idx) % 100 == 0:
            log(f"  keyframes {len(kf_idx)}  (frame {raw})")
    cap.release()
    return dict(fps=fps, stride=stride, n_frames=raw + 1, idx=np.array(kf_idx), A=np.array(kf_A),
                src=np.array(kf_src, np.uint8), inliers=np.array(kf_inl), ratio=np.array(kf_ratio))


# ----------------------------------------------------------------------------- stage 4: interpolation
def interpolate_transforms(kf: dict, cut_scale_step: float = 1.5):
    """A for every frame (N,3,3) by interpolating log-scale, rotation, translation between good keyframes."""
    n = kf["n_frames"]
    ok = kf["src"] != SRC_HELD
    idx, A = kf["idx"][ok], kf["A"][ok]
    a, b = A[:, 0, 0], A[:, 1, 0]
    ls, th = np.log(np.hypot(a, b)), np.unwrap(np.arctan2(b, a))
    f = np.arange(n)
    s_i = np.exp(np.interp(f, idx, ls))
    t_i = np.interp(f, idx, th)
    out = np.zeros((n, 3, 3))
    out[:, 0, 0] = s_i * np.cos(t_i); out[:, 0, 1] = -s_i * np.sin(t_i)
    out[:, 1, 0] = s_i * np.sin(t_i); out[:, 1, 1] = s_i * np.cos(t_i)
    out[:, 0, 2] = np.interp(f, idx, A[:, 0, 2])
    out[:, 1, 2] = np.interp(f, idx, A[:, 1, 2])
    out[:, 2, 2] = 1
    # Camera cuts: hold earlier keyframe transform across large zoom jumps
    jump = np.abs(np.diff(ls)) > np.log(cut_scale_step)
    for k in np.nonzero(jump)[0]:
        seg = (f >= idx[k]) & (f < idx[k + 1])
        out[seg] = A[k]
    src = np.full(n, SRC_INTERP, np.uint8)
    src[f > idx[-1]] = SRC_HELD
    src[kf["idx"]] = kf["src"]
    return out, src


# ----------------------------------------------------------------------------- stage 5: verification
def _read_frame(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    return f if ok else None


def drift_check(video: str, H: np.ndarray, kf: dict, cfg: RegistrarConfig, ui_mask, log=print) -> dict:
    """Match far-apart frames directly and compare with chain prediction to measure accumulated drift."""
    rng = np.random.default_rng(cfg.seed)
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    gx, gy = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 5), np.linspace(0.1 * h, 0.9 * h, 5))
    grid = np.stack([gx.ravel(), gy.ravel()], 1).reshape(-1, 1, 2).astype(np.float32)
    mt = Matcher(cfg, ui_mask)
    good_kf = kf["idx"][kf["src"] != SRC_HELD]
    min_gap = int(cfg.drift_min_gap_s * kf["fps"])
    pairs, tries = [], 0
    while len(pairs) < cfg.drift_pairs and tries < 4000 and len(good_kf) > 2:
        tries += 1
        i, j = sorted(rng.choice(good_kf, 2, replace=False))
        if j - i < min_gap:
            continue
        pi = cv2.perspectiveTransform(corners, H[i]).reshape(-1, 2).astype(np.float32)
        pj = cv2.perspectiveTransform(corners, H[j]).reshape(-1, 2).astype(np.float32)
        inter, _ = cv2.intersectConvexConvex(pi, pj)
        if inter / max(cv2.contourArea(pi), 1) >= cfg.drift_min_overlap:
            pairs.append((int(i), int(j)))
    rows = []
    for i, j in pairs:
        fi, fj = _read_frame(cap, i), _read_frame(cap, j)
        if fi is None or fj is None:
            continue
        r = mt.match(mt.features(fj), mt.features(fi))
        if r is None or r[1] < 40:
            continue
        M, inl, _ = r
        via_chain = cv2.perspectiveTransform(grid, H[j]).reshape(-1, 2)
        via_direct = cv2.perspectiveTransform(cv2.perspectiveTransform(grid, M), H[i]).reshape(-1, 2)
        err = float(np.median(np.hypot(*(via_chain - via_direct).T)))
        rows.append(dict(i=i, j=j, gap_s=round((j - i) / kf["fps"], 1), inliers=inl, err_base_px=round(err, 2)))
    cap.release()
    errs = np.array([r["err_base_px"] for r in rows]) if rows else np.array([np.nan])
    return dict(pairs_tested=len(pairs), pairs_used=len(rows),
                median_px=float(np.nanmedian(errs)), p90_px=float(np.nanpercentile(errs, 90)),
                max_px=float(np.nanmax(errs)), pairs=rows)


def overlay_check(video: str, base_path: str, H: np.ndarray, out_png: str, n: int = 6):
    """Generate a visual check: n frames warped onto the base map, side by side."""
    base = cv2.imread(base_path)
    cap = cv2.VideoCapture(video)
    tiles = []
    for i in np.linspace(0, len(H) - 1, n).astype(int):
        f = _read_frame(cap, i)
        if f is None:
            continue
        m = np.full(f.shape[:2], 255, np.uint8)
        w = cv2.warpPerspective(f, H[i], (base.shape[1], base.shape[0]))
        wm = cv2.warpPerspective(m, H[i], (base.shape[1], base.shape[0])) > 0
        o = base.copy()
        o[wm] = (0.5 * o[wm] + 0.5 * w[wm]).astype(np.uint8)
        cv2.putText(o, f"frame {i}", (30, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)
        tiles.append(cv2.resize(o, (700, 700)))
    cap.release()
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out_png, np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)]))


# ----------------------------------------------------------------------------- public entry points
def run(video: str, homography_json: str, out_dir: str, base_image=None,
        cfg: RegistrarConfig | None = None, log=print) -> dict:
    cfg = cfg or RegistrarConfig()
    os.makedirs(out_dir, exist_ok=True)
    H0, _ = load_homography(homography_json)
    log("building fixed-UI mask ...")
    ui = compute_ui_mask(video)
    cv2.imwrite(os.path.join(out_dir, "ui_mask.png"), ui)
    log("registering keyframes ...")
    kf = register_keyframes(video, cfg, ui, log)
    log(f"interpolating {kf['n_frames']} frame transforms ...")
    A, src = interpolate_transforms(kf, cfg.cut_scale_step)
    H = np.einsum("ij,njk->nik", H0, A)
    H /= H[:, 2:3, 2:3]
    np.savez_compressed(
        os.path.join(out_dir, "transforms.npz"),
        H=H, source=src,
        t=np.arange(kf["n_frames"]) / kf["fps"],
        fps=kf["fps"],
        keyframes=kf["idx"],
        keyframe_inliers=kf["inliers"],
    )
    log("drift check ...")
    drift = drift_check(video, H, kf, cfg, ui, log)
    scales = np.hypot(H[:, 0, 0], H[:, 1, 0])
    low = [
        (int(i), int(n))
        for i, n, s in zip(kf["idx"], kf["inliers"], kf["src"])
        if s != SRC_HELD and 0 <= n < 40
    ]
    report = dict(
        video=os.path.basename(video), fps=kf["fps"], frames=kf["n_frames"],
        keyframe_stride=kf["stride"], keyframes=len(kf["idx"]),
        registered_fps=round(kf["fps"] / kf["stride"], 2),
        frames_by_source={SRC_NAMES[k]: int((src == k).sum()) for k in SRC_NAMES},
        keyframes_failed=int((kf["src"] == SRC_HELD).sum()),
        keyframes_low_inliers=low[:20],
        scale_frame_to_base=dict(min=float(scales.min()), max=float(scales.max())),
        drift_check=drift,
        config=asdict(cfg),
    )
    json.dump(report, open(os.path.join(out_dir, "registration.json"), "w"), indent=2)
    if base_image:
        log("generating overlay check image ...")
        overlay_check(video, base_image, H, os.path.join(out_dir, "registration_check.png"))
    log(
        f"done: {kf['n_frames']} frames, {len(kf['idx'])} keyframes, "
        f"failed={report['keyframes_failed']}, "
        f"drift median={drift['median_px']:.1f}px / max={drift['max_px']:.1f}px"
    )
    return report


def load_transforms(path: str):
    """Load transforms.npz -> H (N,3,3) frame->base, source (N,), fps"""
    z = np.load(path)
    return z["H"], z["source"], float(z["fps"])


def frame_to_base(H: np.ndarray, pts) -> np.ndarray:
    """Map Nx2 frame pixels to base map coords using one frame's 3x3 transform."""
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser(description="Step 5: per-frame camera transforms (frame -> bermuda.jpeg)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--homography", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="bermuda.jpeg for the overlay check")
    ap.add_argument("--max-register-fps", type=float, default=10.0)
    a = ap.parse_args()
    run(a.video, a.homography, a.out, a.base, RegistrarConfig(max_register_fps=a.max_register_fps))


if __name__ == "__main__":
    main()
