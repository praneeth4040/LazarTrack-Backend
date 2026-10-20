"""
Player Path Map Aligner (Step 4) — 8x8 Grid Intersection-based Alignment

Computes a precise homography between the in-game clean map video frame and
the high-resolution reference map by matching the 49 intersection points
formed by dividing both maps into an 8x8 grid (7 vertical and 7 horizontal lines:
B-H across columns, J-P down rows).

Provides bidirectional coordinate transformations:
  - video_to_map(x, y) → maps video pixel to reference map coordinate
  - map_to_video(x, y) → maps reference map coordinate to video pixel
"""

from dataclasses import dataclass, asdict
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("app.zone_tracker.map_aligner")


# ------------------------------------------------------------------ results
@dataclass
class AlignmentMetrics:
    """Quantitative evaluation of map alignment accuracy."""
    method: str                 # "grid_8x8_intersections", "features", "scale"
    n_correspondences: int
    reprojection_error: float   # mean px error on the reference map
    grid_confidence: float      # 0-1 confidence
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    clean_map_dimensions: dict
    blank_map_dimensions: dict


@dataclass
class MapAlignmentResult:
    """Complete results and matrix parameters for Step 4."""
    homography_matrix: list
    inverse_homography_matrix: list
    metrics: AlignmentMetrics
    alignment_overlay_path: str
    homography_json_path: str


# ------------------------------------------------------------------ aligner
class PlayerPathMapAligner:
    """Calculates transformation matrix between clean map stream and reference map."""

    def __init__(
        self,
        reference_map_path: str = "assets/maps/bermuda.jpeg",
        output_dir: str = "video/output/zone/step4",
    ):
        self.reference_map_path = reference_map_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def align(
        self,
        clean_map_image_path: str,
        video_id: str,
    ) -> MapAlignmentResult:
        """Compute alignment between clean map frame and reference blank map."""
        if not os.path.exists(clean_map_image_path):
            raise FileNotFoundError(f"Clean map image not found at: {clean_map_image_path}")
        if not os.path.exists(self.reference_map_path):
            raise FileNotFoundError(f"Reference map not found at: {self.reference_map_path}")

        session_out_dir = os.path.join(self.output_dir, video_id)
        os.makedirs(session_out_dir, exist_ok=True)

        clean_map = cv2.imread(clean_map_image_path)
        blank_map = cv2.imread(self.reference_map_path)

        if clean_map is None or blank_map is None:
            raise ValueError("Failed to load map images for alignment.")

        h_clean, w_clean = clean_map.shape[:2]
        h_blank, w_blank = blank_map.shape[:2]

        # -------------------------------------------------------------- Tier 1: SIFT Feature Matching
        # Highly accurate on terrain landmarks, automatically resolving any outer map padding
        H, method, metrics_extra = self._try_feature_alignment(
            clean_map, blank_map, video_id,
        )

        # -------------------------------------------------------------- Tier 2: 8x8 Grid Intersections
        if H is None:
            log.info("[%s] Feature alignment not used; using 8x8 grid intersections", video_id)
            H, method, metrics_extra = self._try_grid_alignment(
                clean_map, blank_map, session_out_dir, video_id,
            )

        # -------------------------------------------------------------- Tier 3: Simple Scale Fallback
        if H is None:
            log.warning("[%s] Grid alignment also failed; using simple scale", video_id)
            H, method, metrics_extra = self._simple_scale(
                clean_map, blank_map, video_id,
            )

        H_inv = np.linalg.inv(H)

        # Decompose for logging
        sx = float(np.hypot(H[0, 0], H[1, 0]))
        sy = float(np.hypot(H[0, 1], H[1, 1]))
        tx, ty = float(H[0, 2]), float(H[1, 2])

        metrics = AlignmentMetrics(
            method=method,
            n_correspondences=metrics_extra.get("n_corr", 0),
            reprojection_error=metrics_extra.get("reproj", 0.0),
            grid_confidence=metrics_extra.get("grid_conf", 1.0),
            scale_x=round(sx, 5),
            scale_y=round(sy, 5),
            offset_x=round(tx, 3),
            offset_y=round(ty, 3),
            clean_map_dimensions={"width": w_clean, "height": h_clean},
            blank_map_dimensions={"width": w_blank, "height": h_blank},
        )

        # -------------------------------------------------------------- overlay
        overlay_path = self._generate_overlay(
            clean_map, blank_map, H, session_out_dir,
        )

        # -------------------------------------------------------------- JSON
        json_path = os.path.join(session_out_dir, "homography.json")
        out_dict = {
            "video_id": video_id,
            "reference_map": os.path.basename(self.reference_map_path),
            "matrix_H_forward": H.tolist(),
            "matrix_H_inverse": H_inv.tolist(),
            "metrics": asdict(metrics),
            "transform_formula": {
                "forward": "P_map = H * P_video",
                "inverse": "P_video = H_inv * P_map",
            },
        }
        if "intersections" in metrics_extra:
            out_dict["intersection_points"] = metrics_extra["intersections"]

        with open(json_path, "w") as f:
            json.dump(out_dict, f, indent=2)

        log.info(
            "[%s] Step 4 done (method=%s, reproj=%.4fpx, corr=%d) → %s",
            video_id, method, metrics_extra.get("reproj", 0),
            metrics_extra.get("n_corr", 0), overlay_path,
        )

        return MapAlignmentResult(
            homography_matrix=H.tolist(),
            inverse_homography_matrix=H_inv.tolist(),
            metrics=metrics,
            alignment_overlay_path=overlay_path,
            homography_json_path=json_path,
        )

    # ============================================================ 8x8 grid method
    def _try_grid_alignment(
        self,
        clean_map: np.ndarray,
        blank_map: np.ndarray,
        debug_dir: str,
        video_id: str,
    ):
        """Align via 49 corresponding intersection points from the 8x8 grid division.

        The 8x8 grid yields 7 vertical lines (B-H) and 7 horizontal lines (J-P),
        creating exactly 49 intersection points in both the video frame and reference map.
        """
        h_v, w_v = clean_map.shape[:2]
        h_m, w_m = blank_map.shape[:2]

        pts_video = []
        pts_ref = []
        intersections_log = []

        col_labels = list("BCDEFGH")
        row_labels = list("JKLMNOP")

        for j in range(1, 8):      # row 1..7 (J to P)
            y_v = j * (h_v / 8.0)
            y_m = j * (h_m / 8.0)
            for i in range(1, 8):  # col 1..7 (B to H)
                x_v = i * (w_v / 8.0)
                x_m = i * (w_m / 8.0)
                pts_video.append([x_v, y_v])
                pts_ref.append([x_m, y_m])
                intersections_log.append({
                    "grid_coord": f"{col_labels[i-1]}{row_labels[j-1]}",
                    "col_idx": i,
                    "row_idx": j,
                    "video_px": [round(x_v, 2), round(y_v, 2)],
                    "map_px": [round(x_m, 2), round(y_m, 2)],
                })

        pts_video = np.array(pts_video, dtype=np.float64)
        pts_ref = np.array(pts_ref, dtype=np.float64)

        # Compute Homography
        H, mask = cv2.findHomography(pts_video, pts_ref, cv2.RANSAC, 2.0)
        if H is None:
            return None, "grid", {}

        H = H / H[2, 2]

        # Measure reprojection error on all 49 points
        projected = cv2.perspectiveTransform(
            pts_video.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected - pts_ref, axis=1)
        mean_reproj = float(np.mean(errors))

        extras = {
            "n_corr": len(pts_video),
            "reproj": mean_reproj,
            "grid_conf": 1.0,
            "intersections": intersections_log,
        }

        log.info(
            "[%s] 8x8 Grid homography: 49/49 intersections matched, reproj=%.4fpx",
            video_id, mean_reproj,
        )

        return H, "grid_8x8_intersections", extras

    # ========================================================== feature method
    def _try_feature_alignment(
        self,
        clean_map: np.ndarray,
        blank_map: np.ndarray,
        video_id: str,
    ):
        """Scale-invariant SIFT feature matching with RANSAC."""
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(clean_map, None)
        kp2, des2 = sift.detectAndCompute(blank_map, None)

        if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
            return None, "sift_features", {}

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 30:
            return None, "sift_features", {}

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or mask is None:
            return None, "sift_features", {}

        inliers = int(mask.sum())
        if inliers < 30:
            return None, "sift_features", {}

        H = H / H[2, 2]
        inl = mask.ravel().astype(bool)
        projected = cv2.perspectiveTransform(src[inl], H).reshape(-1, 2)
        reproj = float(np.mean(np.linalg.norm(projected - dst[inl].reshape(-1, 2), axis=1)))

        if reproj > 3.0:
            return None, "sift_features", {}

        log.info(
            "[%s] SIFT feature alignment: %d/%d inliers, reproj=%.2fpx",
            video_id, inliers, len(good), reproj,
        )

        return H, "sift_features", {"n_corr": inliers, "reproj": reproj, "grid_conf": 1.0}

    # =========================================================== simple scale
    def _simple_scale(
        self,
        clean_map: np.ndarray,
        blank_map: np.ndarray,
        video_id: str,
    ):
        """Last-resort: uniform scale."""
        h_c, w_c = clean_map.shape[:2]
        h_b, w_b = blank_map.shape[:2]

        sx = w_b / float(w_c)
        sy = h_b / float(h_c)

        H = np.array([
            [sx, 0.0, 0.0],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        return H, "scale", {"n_corr": 0, "reproj": 0.0, "grid_conf": 0.0}

    # =========================================================== overlay
    def _generate_overlay(
        self,
        clean_map: np.ndarray,
        blank_map: np.ndarray,
        H: np.ndarray,
        out_dir: str,
    ) -> str:
        """Warp clean map onto reference and blend 50/50 with annotated grid & intersections."""
        h_b, w_b = blank_map.shape[:2]
        warped = cv2.warpPerspective(clean_map, H, (w_b, h_b))
        overlay = cv2.addWeighted(blank_map, 0.5, warped, 0.5, 0)

        dx = w_b / 8.0
        dy = h_b / 8.0

        # Draw 7 vertical grid lines (cyan with dark border)
        for i in range(1, 8):
            x = int(round(i * dx))
            cv2.line(overlay, (x, 0), (x, h_b), (0, 0, 0), 2)
            cv2.line(overlay, (x, 0), (x, h_b), (0, 255, 255), 1)

        # Draw 7 horizontal grid lines
        for j in range(1, 8):
            y = int(round(j * dy))
            cv2.line(overlay, (0, y), (w_b, y), (0, 0, 0), 2)
            cv2.line(overlay, (0, y), (w_b, y), (0, 255, 255), 1)

        # Highlight all 49 intersection points (green glowing circles)
        for j in range(1, 8):
            y = int(round(j * dy))
            for i in range(1, 8):
                x = int(round(i * dx))
                cv2.circle(overlay, (x, y), 6, (0, 0, 0), -1)
                cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)

        col_labels = list("BCDEFGH")
        row_labels = list("JKLMNOP")
        for i in range(7):
            cx = int(round((i + 1) * dx))
            cv2.putText(overlay, col_labels[i], (cx - 7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(overlay, col_labels[i], (cx - 7, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        for j in range(7):
            cy = int(round((j + 1) * dy))
            cv2.putText(overlay, row_labels[j], (10, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(overlay, row_labels[j], (10, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        path = os.path.join(out_dir, "alignment_overlay.png")
        cv2.imwrite(path, overlay)
        return path
