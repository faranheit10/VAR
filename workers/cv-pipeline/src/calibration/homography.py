"""4-Point & Multi-Keypoint Homography Engine (ViewTransformer).

Converts 2D camera pixels into physical pitch turf meters and vice-versa
using robust RANSAC-estimated homography matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from schemas import PitchCalibrationPayload, PitchCorrespondence


DEFAULT_PITCH_LENGTH_M: float = 50.0
DEFAULT_PITCH_WIDTH_M: float = 32.0


def compute_reprojection_rmse(
    ground_truth_meters: np.ndarray,
    reprojected_meters: np.ndarray,
) -> float:
    """Compute Root Mean Square Error (RMSE) between ground-truth and reprojected meter coordinates.

    Args:
        ground_truth_meters: Array of shape (N, 2) in meters.
        reprojected_meters: Array of shape (N, 2) in meters.

    Returns:
        Reprojection RMSE in meters.
    """
    gt = ground_truth_meters.reshape(-1, 2).astype(np.float64)
    rep = reprojected_meters.reshape(-1, 2).astype(np.float64)
    l2_errors = np.linalg.norm(gt - rep, axis=1)
    rmse = float(np.sqrt(np.mean(l2_errors ** 2)))
    return rmse


class ViewTransformer:
    """Perspective transformation engine between 2D camera pixels and 2D turf meters."""

    def __init__(
        self,
        source_pixels: np.ndarray,
        target_meters: np.ndarray,
        pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
        pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
        ransac_reproj_threshold: float = 3.0,
    ) -> None:
        """Fit forward and inverse homography matrices between pixel coordinates and turf meters.

        Args:
            source_pixels: Array of shape (N, 2) containing camera pixel coordinates (u, v).
            target_meters: Array of shape (N, 2) containing physical turf coordinates (X, Y).
            pitch_length_m: Pitch length in meters (default 50.0m).
            pitch_width_m: Pitch width in meters (default 32.0m).
            ransac_reproj_threshold: Maximum allowed reprojection error in pixels to treat as inlier.
        """
        src = source_pixels.reshape(-1, 1, 2).astype(np.float64)
        dst = target_meters.reshape(-1, 1, 2).astype(np.float64)

        if len(src) < 4:
            raise ValueError(f"At least 4 correspondences required, got {len(src)}")

        # Estimate homography from pixels to meters using RANSAC
        homography_matrix, inliers_mask = cv2.findHomography(
            srcPoints=src,
            dstPoints=dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_reproj_threshold,
        )

        if homography_matrix is None:
            raise RuntimeError("Failed to compute valid homography matrix from correspondences")

        self.H_pixel_to_meter: np.ndarray = homography_matrix.astype(np.float64)

        # Verify invertibility and condition number
        det = float(np.linalg.det(self.H_pixel_to_meter))
        if abs(det) < 1e-12:
            raise RuntimeError(f"Singular homography matrix encountered (det={det})")

        cond = float(np.linalg.cond(self.H_pixel_to_meter))
        if cond > 1e12:
            raise RuntimeError(f"Ill-conditioned homography matrix (condition number={cond})")

        self.H_meter_to_pixel: np.ndarray = np.linalg.inv(self.H_pixel_to_meter)
        self.inliers_mask: Optional[np.ndarray] = inliers_mask
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)

    @classmethod
    def from_calibration_payload(
        cls,
        payload: Union[PitchCalibrationPayload, Dict[str, Any]],
        ransac_reproj_threshold: float = 3.0,
    ) -> ViewTransformer:
        """Construct a ViewTransformer instance from a validated PitchCalibrationPayload."""
        if isinstance(payload, PitchCalibrationPayload):
            p_data = payload.model_dump()
        else:
            # Validate dict through Pydantic schema
            validated = PitchCalibrationPayload.model_validate(payload)
            p_data = validated.model_dump()

        correspondences: List[Dict[str, Any]] = p_data["correspondences"]
        source_pixels = np.array(
            [[c["pixel_u"], c["pixel_v"]] for c in correspondences],
            dtype=np.float64,
        )
        target_meters = np.array(
            [[c["meter_x"], c["meter_y"]] for c in correspondences],
            dtype=np.float64,
        )

        return cls(
            source_pixels=source_pixels,
            target_meters=target_meters,
            pitch_length_m=float(p_data.get("pitch_length_m", DEFAULT_PITCH_LENGTH_M)),
            pitch_width_m=float(p_data.get("pitch_width_m", DEFAULT_PITCH_WIDTH_M)),
            ransac_reproj_threshold=ransac_reproj_threshold,
        )

    def pixel_to_meter(self, points_uv: np.ndarray) -> np.ndarray:
        """Transform camera pixel coordinates (u, v) into physical turf meters (X, Y).

        Formula:
            p' = H_pixel_to_meter * [u, v, 1]^T
            X_meter = p'_x / p'_z, Y_meter = p'_y / p'_z

        Args:
            points_uv: Array of shape (2,), (N, 2), or (N, 1, 2) in pixel space.

        Returns:
            Array of shape (N, 2) (or (2,) if 1D input) in meters.
        """
        arr = np.asarray(points_uv, dtype=np.float64)
        is_1d = arr.ndim == 1 and arr.shape[0] == 2
        pts = arr.reshape(-1, 1, 2)

        transformed: np.ndarray = cv2.perspectiveTransform(pts, self.H_pixel_to_meter)
        res: np.ndarray = transformed.reshape(-1, 2)
        if is_1d:
            out_1d: np.ndarray = np.asarray(res[0], dtype=np.float64)
            return out_1d
        out_2d: np.ndarray = np.asarray(res, dtype=np.float64)
        return out_2d

    def meter_to_pixel(self, points_xy: np.ndarray) -> np.ndarray:
        """Transform physical turf meters (X, Y) into camera pixel coordinates (u, v).

        Formula:
            p = H_meter_to_pixel * [X, Y, 1]^T
            u = p_u / p_w, v = p_v / p_w

        Args:
            points_xy: Array of shape (2,), (N, 2), or (N, 1, 2) in meter space.

        Returns:
            Array of shape (N, 2) (or (2,) if 1D input) in pixels.
        """
        arr = np.asarray(points_xy, dtype=np.float64)
        is_1d = arr.ndim == 1 and arr.shape[0] == 2
        pts = arr.reshape(-1, 1, 2)

        transformed: np.ndarray = cv2.perspectiveTransform(pts, self.H_meter_to_pixel)
        res: np.ndarray = transformed.reshape(-1, 2)
        if is_1d:
            out_1d: np.ndarray = np.asarray(res[0], dtype=np.float64)
            return out_1d
        out_2d: np.ndarray = np.asarray(res, dtype=np.float64)
        return out_2d

    def compute_roundtrip_rmse(self, points_meter: np.ndarray) -> float:
        """Compute round-trip reprojection RMSE ||p - H^-1(H(p))||_2 in meters."""
        pts_m = points_meter.reshape(-1, 2).astype(np.float64)
        pixels = self.meter_to_pixel(pts_m)
        recovered_meters = self.pixel_to_meter(pixels)
        return compute_reprojection_rmse(pts_m, recovered_meters)

    def is_inside_pitch(self, points_meter: np.ndarray, margin_m: float = 0.5) -> np.ndarray:
        """Filter out spectators and substitutes outside turf boundaries.

        Enforces EDGE_CASES_AND_GUARDRAILS.md Rule 1:
            -0.5m <= X <= 50.5m and -0.5m <= Y <= 32.5m
        """
        pts = points_meter.reshape(-1, 2).astype(np.float64)
        x = pts[:, 0]
        y = pts[:, 1]
        valid_x = (x >= -margin_m) & (x <= self.pitch_length_m + margin_m)
        valid_y = (y >= -margin_m) & (y <= self.pitch_width_m + margin_m)
        return valid_x & valid_y


def render_top_down_pitch(
    pitch_length_m: float = DEFAULT_PITCH_LENGTH_M,
    pitch_width_m: float = DEFAULT_PITCH_WIDTH_M,
    px_per_meter: int = 20,
    grid_step_m: float = 5.0,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Render a top-down 2D green pitch visualization with accurate markings and 5m x 5m grid.

    Args:
        pitch_length_m: Pitch length in meters (default 50.0m).
        pitch_width_m: Pitch width in meters (default 32.0m).
        px_per_meter: Scale factor in pixels per meter.
        grid_step_m: Grid line interval in meters.
        output_path: Optional file path to save JPEG artifact.

    Returns:
        Rendered BGR image array.
    """
    margin_px = 60
    pitch_w_px = int(round(pitch_length_m * px_per_meter))
    pitch_h_px = int(round(pitch_width_m * px_per_meter))

    total_w = pitch_w_px + 2 * margin_px
    total_h = pitch_h_px + 2 * margin_px

    # Outer border / turf surround
    img = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    img[:] = (24, 38, 26)  # Dark surround

    # Base pitch area with alternating lawn mowed stripes
    stripe_w_px = int(round(5.0 * px_per_meter))
    num_stripes = int(np.ceil(pitch_w_px / stripe_w_px))
    for s in range(num_stripes):
        x1 = margin_px + s * stripe_w_px
        x2 = min(margin_px + (s + 1) * stripe_w_px, margin_px + pitch_w_px)
        color = (34, 112, 48) if s % 2 == 0 else (38, 128, 54)
        cv2.rectangle(img, (x1, margin_px), (x2, margin_px + pitch_h_px), color, -1)

    # 5m x 5m subtle tactical grid
    grid_color = (60, 150, 75)
    for x_m in np.arange(0.0, pitch_length_m + 0.1, grid_step_m):
        gx = margin_px + int(round(x_m * px_per_meter))
        cv2.line(img, (gx, margin_px), (gx, margin_px + pitch_h_px), grid_color, 1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{int(x_m)}m",
            (gx - 10, margin_px - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 220, 180),
            1,
            cv2.LINE_AA,
        )

    for y_m in np.arange(0.0, pitch_width_m + 0.1, grid_step_m):
        gy = margin_px + int(round(y_m * px_per_meter))
        cv2.line(img, (margin_px, gy), (margin_px + pitch_w_px, gy), grid_color, 1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{int(y_m)}m",
            (margin_px - 45, gy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 220, 180),
            1,
            cv2.LINE_AA,
        )

    # White pitch markings
    white = (255, 255, 255)
    line_thickness = 2

    # Pitch boundary (touchlines and goal lines)
    cv2.rectangle(
        img,
        (margin_px, margin_px),
        (margin_px + pitch_w_px, margin_px + pitch_h_px),
        white,
        line_thickness,
        cv2.LINE_AA,
    )

    # Halfway line
    half_x = margin_px + int(round((pitch_length_m / 2.0) * px_per_meter))
    cv2.line(
        img,
        (half_x, margin_px),
        (half_x, margin_px + pitch_h_px),
        white,
        line_thickness,
        cv2.LINE_AA,
    )

    # Center circle (radius 5.0m) and center mark
    center_y = margin_px + int(round((pitch_width_m / 2.0) * px_per_meter))
    center_r = int(round(5.0 * px_per_meter))
    cv2.circle(img, (half_x, center_y), center_r, white, line_thickness, cv2.LINE_AA)
    cv2.circle(img, (half_x, center_y), 4, white, -1, cv2.LINE_AA)

    # Penalty areas (9m depth, 18m width, centered at Y = 16.0m)
    box_depth_px = int(round(9.0 * px_per_meter))
    box_w_m = 18.0
    box_y1 = margin_px + int(round(((pitch_width_m - box_w_m) / 2.0) * px_per_meter))
    box_y2 = margin_px + int(round(((pitch_width_m + box_w_m) / 2.0) * px_per_meter))

    # Left penalty area
    cv2.rectangle(
        img,
        (margin_px, box_y1),
        (margin_px + box_depth_px, box_y2),
        white,
        line_thickness,
        cv2.LINE_AA,
    )
    # Right penalty area
    cv2.rectangle(
        img,
        (margin_px + pitch_w_px - box_depth_px, box_y1),
        (margin_px + pitch_w_px, box_y2),
        white,
        line_thickness,
        cv2.LINE_AA,
    )

    # Penalty spots (at 7.0m from each goal line)
    spot_left_x = margin_px + int(round(7.0 * px_per_meter))
    spot_right_x = margin_px + int(round((pitch_length_m - 7.0) * px_per_meter))
    cv2.circle(img, (spot_left_x, center_y), 4, white, -1, cv2.LINE_AA)
    cv2.circle(img, (spot_right_x, center_y), 4, white, -1, cv2.LINE_AA)

    # Goals (5.0m width, Y in [13.5, 18.5], extending 1.5m deep behind goal line)
    goal_w_m = 5.0
    goal_depth_px = int(round(1.5 * px_per_meter))
    goal_y1 = margin_px + int(round(((pitch_width_m - goal_w_m) / 2.0) * px_per_meter))
    goal_y2 = margin_px + int(round(((pitch_width_m + goal_w_m) / 2.0) * px_per_meter))

    # Left goal
    cv2.rectangle(img, (margin_px - goal_depth_px, goal_y1), (margin_px, goal_y2), (200, 200, 200), 2)
    # Right goal
    cv2.rectangle(
        img,
        (margin_px + pitch_w_px, goal_y1),
        (margin_px + pitch_w_px + goal_depth_px, goal_y2),
        (200, 200, 200),
        2,
    )

    # Title Banner
    cv2.putText(
        img,
        f"8v8 TURFVISION TACTICAL RADAR PITCH ({pitch_length_m:.0f}m x {pitch_width_m:.0f}m)",
        (margin_px, total_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 240, 220),
        1,
        cv2.LINE_AA,
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), img)

    return img


def overlay_pitch_grid_on_camera(
    frame: np.ndarray,
    view_transformer: ViewTransformer,
    grid_step_m: float = 5.0,
    rmse_meters: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Project 5m x 5m meter grid and pitch boundaries onto camera frame via homography.

    Args:
        frame: Video image frame (H, W, 3).
        view_transformer: Configured ViewTransformer instance.
        grid_step_m: Grid spacing in physical meters.
        rmse_meters: Reprojection RMSE in meters for header text.
        output_path: Optional path to save visual artifact.

    Returns:
        Annotated camera frame with projected homography grid.
    """
    overlay = frame.copy()
    L = view_transformer.pitch_length_m
    W = view_transformer.pitch_width_m

    # Sample density along lines
    num_samples = 40

    # Draw transverse meter grid lines (Y constant, varying X)
    grid_color = (0, 220, 220)  # Bright Cyan
    for y_m in np.arange(0.0, W + 0.01, grid_step_m):
        x_pts = np.linspace(0.0, L, num_samples)
        y_pts = np.full_like(x_pts, y_m)
        pts_m = np.stack([x_pts, y_pts], axis=1)
        pts_px = view_transformer.meter_to_pixel(pts_m).astype(np.int32)
        cv2.polylines(overlay, [pts_px], isClosed=False, color=grid_color, thickness=1, lineType=cv2.LINE_AA)

    # Draw longitudinal meter grid lines (X constant, varying Y)
    for x_m in np.arange(0.0, L + 0.01, grid_step_m):
        y_pts = np.linspace(0.0, W, num_samples)
        x_pts = np.full_like(y_pts, x_m)
        pts_m = np.stack([x_pts, y_pts], axis=1)
        pts_px = view_transformer.meter_to_pixel(pts_m).astype(np.int32)
        cv2.polylines(overlay, [pts_px], isClosed=False, color=grid_color, thickness=1, lineType=cv2.LINE_AA)

    # Bold outer pitch perimeter
    boundary_color = (0, 255, 128)  # Neon Green
    corners_m = np.array(
        [[0.0, 0.0], [L, 0.0], [L, W], [0.0, W]],
        dtype=np.float64,
    )
    corners_px = view_transformer.meter_to_pixel(corners_m).astype(np.int32)
    cv2.polylines(overlay, [corners_px], isClosed=True, color=boundary_color, thickness=3, lineType=cv2.LINE_AA)

    # Project Center Circle
    theta = np.linspace(0, 2 * np.pi, 72)
    circle_x = (L / 2.0) + 5.0 * np.cos(theta)
    circle_y = (W / 2.0) + 5.0 * np.sin(theta)
    circle_m = np.stack([circle_x, circle_y], axis=1)
    circle_px = view_transformer.meter_to_pixel(circle_m).astype(np.int32)
    cv2.polylines(overlay, [circle_px], isClosed=True, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # Halfway line
    half_m = np.array([[L / 2.0, 0.0], [L / 2.0, W]], dtype=np.float64)
    half_px = view_transformer.meter_to_pixel(half_m).astype(np.int32)
    cv2.line(overlay, tuple(half_px[0]), tuple(half_px[1]), (255, 255, 0), 2, cv2.LINE_AA)

    # Project Penalty spots
    spots_m = np.array([[7.0, W / 2.0], [L - 7.0, W / 2.0]], dtype=np.float64)
    spots_px = view_transformer.meter_to_pixel(spots_m).astype(np.int32)
    for spot in spots_px:
        cv2.circle(overlay, tuple(spot), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(spot), 8, (255, 255, 255), 2, cv2.LINE_AA)

    # Header badge overlay
    banner_h = 60
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], banner_h), (20, 20, 20), -1)

    title = f"TURFVISION HOMOGRAPHY 5m x 5m METER GRID PROJECTION ({L:.0f}m x {W:.0f}m)"
    if rmse_meters is not None:
        title += f" | Reprojection RMSE: {rmse_meters:.4f}m (Gold Standard: < 0.15m)"

    cv2.putText(
        overlay,
        title,
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), overlay)

    return overlay
