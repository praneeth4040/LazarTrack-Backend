"""
Grid Label Detector — Detects B-H / J-P grid coordinate labels
in Free Fire map images (both reference maps and in-game minimap frames).

The grid consists of:
  - 7 column labels across the top:  B, C, D, E, F, G, H
  - 7 row labels down the left side: J, K, L, M, N, O, P

Each label marks a grid line position, creating a 7×7 grid of 49 intersection
points.  These shared reference points enable precise alignment between any two
map images.
"""

import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("app.zone_tracker.grid_label_detector")

# Grid label identifiers (for debug annotation only)
TOP_LABELS = list("BCDEFGH")
LEFT_LABELS = list("JKLMNOP")

N_LABELS = 7           # always 7 labels in each strip


# ------------------------------------------------------------------ public API
def detect_grid_positions(
    image: np.ndarray,
    margin_frac: float = 0.055,
    min_blob_area: int = 6,
    max_blob_area: int = 4000,
    debug_dir: Optional[str] = None,
) -> Optional[Dict]:
    """Detect grid label positions in a Free Fire map image.

    Args
    ----
    image        : BGR map image (reference or video frame).
    margin_frac  : fraction of width/height to search for labels.
    min_blob_area: smallest connected component accepted as a letter.
    max_blob_area: largest  connected component accepted as a letter.
    debug_dir    : if set, write annotated debug images here.

    Returns
    -------
    dict with ``top_x`` (7 floats), ``left_y`` (7 floats),
    ``top_spacing``, ``left_spacing``, ``confidence``,
    ``n_detected_top``, ``n_detected_left``
    — or *None* if detection fails.
    """
    h, w = image.shape[:2]

    top_h  = max(int(h * margin_frac), 18)
    left_w = max(int(w * margin_frac), 18)

    top_strip  = image[:top_h, :]
    left_strip = image[:, :left_w]

    top_result = _detect_labels_in_strip(
        top_strip, axis="x",
        min_area=min_blob_area, max_area=max_blob_area,
        image_extent=w,
    )
    left_result = _detect_labels_in_strip(
        left_strip, axis="y",
        min_area=min_blob_area, max_area=max_blob_area,
        image_extent=h,
    )

    if top_result is None or left_result is None:
        log.warning("Grid label detection failed (top=%s, left=%s)",
                     top_result is not None, left_result is not None)
        return None

    top_x,  top_spacing,  n_det_top  = top_result
    left_y, left_spacing, n_det_left = left_result

    confidence = min(n_det_top, n_det_left) / float(N_LABELS)

    if debug_dir:
        _save_debug(image, top_x, left_y, debug_dir)

    return {
        "top_x":           top_x,
        "left_y":          left_y,
        "top_spacing":     top_spacing,
        "left_spacing":    left_spacing,
        "confidence":      confidence,
        "n_detected_top":  n_det_top,
        "n_detected_left": n_det_left,
    }


def get_grid_intersections(grid_info: Dict) -> np.ndarray:
    """Compute 49 grid intersection points from detected grid positions.

    Returns ndarray of shape (49, 2) — (x, y) for each intersection.
    """
    top_x  = grid_info["top_x"]
    left_y = grid_info["left_y"]

    pts = []
    for y in left_y:          # row-major order
        for x in top_x:
            pts.append([x, y])
    return np.array(pts, dtype=np.float64)


# ---------------------------------------------------------------- strip logic
def _detect_labels_in_strip(
    strip: np.ndarray,
    axis: str,
    min_area: int,
    max_area: int,
    image_extent: int,
    n_expected: int = N_LABELS,
) -> Optional[Tuple[List[float], float, int]]:
    """Find *n_expected* regularly-spaced letter blobs in *strip*.

    Parameters
    ----------
    strip        : BGR strip image (top margin or left margin).
    axis         : ``'x'`` → horizontal strip (return x centroids);
                   ``'y'`` → vertical strip  (return y centroids).
    min_area     : min connected-component area to consider.
    max_area     : max connected-component area to consider.
    image_extent : full image width (for x) or height (for y) — used for
                   sanity checks on the grid span.
    n_expected   : how many labels we expect (always 7).

    Returns ``(positions_list, spacing, n_actually_detected)`` or *None*.
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip.copy()

    # Collect candidate blobs across several brightness thresholds plus Otsu
    all_centroids: List[float] = []

    for thresh in _threshold_sweep(gray):
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        _collect_blobs(bw, axis, min_area, max_area, all_centroids)

    # Otsu
    otsu_val, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _collect_blobs(bw_otsu, axis, min_area, max_area, all_centroids)

    # Also try inverted Otsu (in case strip is light-on-dark vs dark-on-light)
    _collect_blobs(255 - bw_otsu, axis, min_area, max_area, all_centroids)

    # Merge nearby detections
    merged = _merge_nearby(sorted(all_centroids), tol=6.0)

    if len(merged) < 3:
        log.debug("Only %d blobs in %s strip; need ≥3", len(merged), axis)
        return None

    # Find the best regular set
    result = _find_regular_set(merged, n_expected, image_extent)
    if result is None:
        log.debug("No regular set found in %s strip from %d candidates", axis, len(merged))
        return None

    positions, spacing, n_detected = result
    log.info("Grid labels (%s): detected %d/%d  spacing=%.1fpx",
             axis, n_detected, n_expected, spacing)
    return positions, spacing, n_detected


def _threshold_sweep(gray: np.ndarray) -> List[int]:
    """Generate a range of threshold values to try."""
    med = int(np.median(gray))
    hi  = int(np.percentile(gray, 95))
    # We want thresholds above the dark background but below 255
    # Typically the background is <80 and letters are >150
    return sorted({max(t, 50) for t in [100, 120, 140, 160, med + 40, (med + hi) // 2]})


def _collect_blobs(
    binary: np.ndarray,
    axis: str,
    min_area: int,
    max_area: int,
    out: List[float],
):
    """Append centroid positions of letter-shaped blobs to *out*."""
    n, lab, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        if area < min_area or area > max_area:
            continue
        # Reject very elongated shapes (grid lines, UI borders)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 5:
            continue
        pos = centroids[i][0] if axis == "x" else centroids[i][1]
        out.append(pos)


# --------------------------------------------------------------- regularity
def _merge_nearby(sorted_positions: List[float], tol: float = 5.0) -> List[float]:
    """Merge positions within *tol* of each other (keep average)."""
    if not sorted_positions:
        return []
    groups: List[List[float]] = [[sorted_positions[0]]]
    for p in sorted_positions[1:]:
        if p - groups[-1][-1] <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [float(np.median(g)) for g in groups]


def _find_regular_set(
    positions: List[float],
    n: int,
    image_extent: int,
    tol_frac: float = 0.18,
) -> Optional[Tuple[List[float], float, int]]:
    """RANSAC-like search for *n* regularly-spaced values in *positions*.

    Tries every pair as anchor points with every possible grid-index
    assignment.  Keeps the model that matches the most candidate positions.

    Returns ``(final_positions, spacing, n_detected)`` or *None*.
    """
    positions = sorted(positions)
    min_spacing = image_extent * 0.05          # labels cannot be closer than 5% of image
    max_spacing = image_extent / (n - 2)       # labels should span <100% of image

    best_score = -1
    best_params = None                         # (origin, spacing, n_matched)

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            p0, p1 = positions[i], positions[j]

            for g0 in range(n):
                for g1 in range(g0 + 1, n):
                    spacing = (p1 - p0) / (g1 - g0)
                    if spacing < min_spacing or spacing > max_spacing:
                        continue

                    origin = p0 - g0 * spacing
                    tol = spacing * tol_frac

                    n_matched = 0
                    err_sum   = 0.0
                    for k in range(n):
                        expected = origin + k * spacing
                        dists = [abs(p - expected) for p in positions]
                        d = min(dists)
                        if d < tol:
                            n_matched += 1
                            err_sum += d

                    # Score: favour many matches, then low error
                    score = n_matched * 10000.0 - err_sum
                    if score > best_score:
                        best_score  = score
                        best_params = (origin, spacing, n_matched)

    if best_params is None or best_params[2] < 3:
        return None

    origin, spacing, n_detected = best_params

    # Build final positions, snapping to nearest actual detection when close
    tol = spacing * tol_frac
    final = []
    for k in range(n):
        expected = origin + k * spacing
        dists = [(abs(p - expected), p) for p in positions]
        d, closest = min(dists, key=lambda x: x[0])
        final.append(closest if d < tol else expected)

    return final, spacing, n_detected


# ---------------------------------------------------------------- debug I/O
def _save_debug(
    image: np.ndarray,
    top_x: List[float],
    left_y: List[float],
    debug_dir: str,
):
    """Draw detected grid lines and intersections on a copy of *image*."""
    os.makedirs(debug_dir, exist_ok=True)
    vis = image.copy()
    h, w = vis.shape[:2]

    for i, x in enumerate(top_x):
        xi = int(round(x))
        cv2.line(vis, (xi, 0), (xi, h), (0, 255, 0), 1)
        cv2.putText(vis, TOP_LABELS[i], (xi - 6, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for i, y in enumerate(left_y):
        yi = int(round(y))
        cv2.line(vis, (0, yi), (w, yi), (0, 255, 0), 1)
        cv2.putText(vis, LEFT_LABELS[i], (8, yi + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for x in top_x:
        for y in left_y:
            cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)

    out_path = os.path.join(debug_dir, "grid_detection.png")
    cv2.imwrite(out_path, vis)
    log.info("Grid detection debug saved → %s", out_path)
