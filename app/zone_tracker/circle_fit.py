"""Robust circle fitting (RANSAC + least squares) shared by the zone steps."""
import numpy as np


def circ3(p):
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
    uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
    return ux, uy, float(np.hypot(x1 - ux, y1 - uy))


def fit_ls(P):
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    s = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    r_sq = s[2] + s[0] ** 2 + s[1] ** 2
    return s[0], s[1], float(np.sqrt(max(0.0, r_sq)))


def residual(P, c):
    return np.abs(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) - c[2])


def coverage_deg(P, c, bin_deg=3):
    a = np.degrees(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    return len(np.unique((a // bin_deg).astype(int))) * bin_deg


def ransac_circle(P, rmin, rmax, tol, iters=6000, min_inliers=100, rng=None, search_cap=4000):
    """Returns (circle, inlier_mask) or None. Search runs on a subsample for speed,
    the final inlier count / refinement uses all points."""
    rng = rng or np.random.default_rng(0)
    if len(P) < min_inliers:
        return None
    S = P if len(P) <= search_cap else P[rng.choice(len(P), search_cap, replace=False)]
    best, bn = None, 0
    for _ in range(iters):
        c = circ3(S[rng.choice(len(S), 3, replace=False)])
        if c is None or not (rmin < c[2] < rmax):
            continue
        n = int((residual(S, c) < tol).sum())
        if n > bn:
            bn, best = n, c
    if best is None:
        return None
    for _ in range(4):
        inl = residual(P, best) < tol
        if inl.sum() < 20:
            break
        best = fit_ls(P[inl])
    inl = residual(P, best) < tol
    if inl.sum() < min_inliers or not (rmin < best[2] < rmax):
        return None
    return best, inl
