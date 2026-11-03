"""Fisheye & Ultra-Wide Lens Undistortion Module for 8v8 Turf Vision.

Provides high-throughput lens undistortion with cached rectification maps
to transform curved ultra-wide smartphone frames into accurate pinhole perspective
projections prior to 2D homography and player tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from schemas import CameraIntrinsics

__all__ = [
    "CameraIntrinsics",
    "LensUndistorter",
    "build_camera_matrix_and_dist_coeffs",
    "compute_collinearity_rmse",
    "create_comparison_montage",
    "distort_points",
    "get_cached_undistorter",
    "undistort_frame",
    "undistort_points",
]


# Module-level registry of cached LensUndistorter instances keyed by intrinsics signature
_UNDISTORTER_CACHE: Dict[Tuple[float, ...], LensUndistorter] = {}


def _intrinsics_to_tuple(intrinsics: Union[CameraIntrinsics, Dict[str, Any]]) -> Tuple[float, ...]:
    """Convert CameraIntrinsics model or dict into a hashable float tuple."""
    if isinstance(intrinsics, CameraIntrinsics):
        return (
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.cx),
            float(intrinsics.cy),
            float(intrinsics.k1),
            float(intrinsics.k2),
            float(intrinsics.p1),
            float(intrinsics.p2),
        )
    return (
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
        float(intrinsics["k1"]),
        float(intrinsics["k2"]),
        float(intrinsics.get("p1", 0.0)),
        float(intrinsics.get("p2", 0.0)),
    )


def build_camera_matrix_and_dist_coeffs(
    intrinsics: Union[CameraIntrinsics, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct OpenCV 3x3 camera matrix K and 5x1 distortion coefficient vector.

    Args:
        intrinsics: CameraIntrinsics schema instance or dict with fx, fy, cx, cy, k1, k2, p1, p2.

    Returns:
        Tuple of (K, dist_coeffs) as float64 numpy arrays.
    """
    fx, fy, cx, cy, k1, k2, p1, p2 = _intrinsics_to_tuple(intrinsics)

    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
    return camera_matrix, dist_coeffs


class LensUndistorter:
    """High-performance lens undistortion worker with cached rectification maps."""

    def __init__(
        self,
        intrinsics: Union[CameraIntrinsics, Dict[str, Any]],
        map_type: int = cv2.CV_32FC1,
    ) -> None:
        """Initialize lens undistorter with camera intrinsics and distortion parameters.

        Args:
            intrinsics: CameraIntrinsics object or dict of camera parameters.
            map_type: OpenCV map type, defaults to cv2.CV_32FC1 for high-speed remap.
        """
        self.intrinsics_tuple = _intrinsics_to_tuple(intrinsics)
        self.K, self.dist_coeffs = build_camera_matrix_and_dist_coeffs(intrinsics)
        self.map_type = map_type
        self._map_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def get_rectification_maps(self, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve or construct cached undistort rectification maps for target dimensions.

        Args:
            width: Image pixel width.
            height: Image pixel height.

        Returns:
            Tuple of (map1, map2) rectification coordinate arrays.
        """
        shape_key = (width, height)
        if shape_key not in self._map_cache:
            map1, map2 = cv2.initUndistortRectifyMap(
                cameraMatrix=self.K,
                distCoeffs=self.dist_coeffs,
                R=None,
                newCameraMatrix=self.K,
                size=(width, height),
                m1type=self.map_type,
            )
            self._map_cache[shape_key] = (map1, map2)
        return self._map_cache[shape_key]

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Undistort an input video frame using cached rectification maps.

        Args:
            frame: Input image array of shape (H, W, C) or (H, W).

        Returns:
            Undistorted image array with identical shape and dtype.
        """
        height, width = frame.shape[:2]
        map1, map2 = self.get_rectification_maps(width=width, height=height)
        undistorted: np.ndarray = cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return undistorted

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Transform distorted 2D pixel coordinates into undistorted pinhole pixel coordinates.

        Args:
            points: Array of shape (N, 2) or (N, 1, 2) representing distorted (u, v) points.

        Returns:
            Array of shape (N, 2) containing undistorted (u, v) points in pixel space.
        """
        pts_2d = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted: np.ndarray = cv2.undistortPoints(
            pts_2d,
            cameraMatrix=self.K,
            distCoeffs=self.dist_coeffs,
            P=self.K,
        )
        return undistorted.reshape(-1, 2)

    def distort_points(self, points: np.ndarray) -> np.ndarray:
        """Apply forward lens distortion to ideal pinhole pixel coordinates.

        Implements Brown-Conrady radial and tangential lens distortion:
            x = (u - cx) / fx, y = (v - cy) / fy
            r^2 = x^2 + y^2
            x_d = x * (1 + k1*r^2 + k2*r^4) + 2*p1*x*y + p2*(r^2 + 2*x^2)
            y_d = y * (1 + k1*r^2 + k2*r^4) + p1*(r^2 + 2*y^2) + 2*p2*x*y
            u_d = fx * x_d + cx, v_d = fy * y_d + cy

        Args:
            points: Array of shape (N, 2) representing ideal pinhole (u, v) coordinates.

        Returns:
            Array of shape (N, 2) representing distorted pixel coordinates.
        """
        pts = points.reshape(-1, 2).astype(np.float64)
        u = pts[:, 0]
        v = pts[:, 1]

        fx = float(self.K[0, 0])
        fy = float(self.K[1, 1])
        cx = float(self.K[0, 2])
        cy = float(self.K[1, 2])
        k1 = float(self.dist_coeffs[0])
        k2 = float(self.dist_coeffs[1])
        p1 = float(self.dist_coeffs[2])
        p2 = float(self.dist_coeffs[3])

        # Normalized coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy

        r2 = x * x + y * y
        r4 = r2 * r2
        radial = 1.0 + k1 * r2 + k2 * r4

        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        u_dist = fx * x_d + cx
        v_dist = fy * y_d + cy

        return np.stack([u_dist, v_dist], axis=1)


def get_cached_undistorter(
    intrinsics: Union[CameraIntrinsics, Dict[str, Any]],
) -> LensUndistorter:
    """Get or instantiate a LensUndistorter singleton for given intrinsics."""
    key = _intrinsics_to_tuple(intrinsics)
    if key not in _UNDISTORTER_CACHE:
        _UNDISTORTER_CACHE[key] = LensUndistorter(intrinsics=intrinsics)
    return _UNDISTORTER_CACHE[key]


def undistort_frame(
    frame: np.ndarray,
    intrinsics: Union[CameraIntrinsics, Dict[str, Any]],
) -> np.ndarray:
    """Undistort an image frame using cached lens distortion maps.

    Args:
        frame: Input image array of shape (H, W, C) or (H, W).
        intrinsics: CameraIntrinsics instance or dict with camera parameters.

    Returns:
        Undistorted image array.
    """
    undistorter = get_cached_undistorter(intrinsics)
    return undistorter.undistort(frame)


def undistort_points(
    points: np.ndarray,
    intrinsics: Union[CameraIntrinsics, Dict[str, Any]],
) -> np.ndarray:
    """Undistort 2D pixel coordinates using camera intrinsics."""
    undistorter = get_cached_undistorter(intrinsics)
    return undistorter.undistort_points(points)


def distort_points(
    points: np.ndarray,
    intrinsics: Union[CameraIntrinsics, Dict[str, Any]],
) -> np.ndarray:
    """Apply forward lens distortion to 2D pixel coordinates."""
    undistorter = get_cached_undistorter(intrinsics)
    return undistorter.distort_points(points)


def compute_collinearity_rmse(points: np.ndarray) -> float:
    """Compute Root Mean Square Error (RMSE) of points relative to straight line fit y = m*x + b.

    Args:
        points: Array of shape (N, 2) containing (u, v) coordinates along a suspected line.

    Returns:
        Collinearity RMSE in pixels.
    """
    pts = points.reshape(-1, 2).astype(np.float64)
    if len(pts) < 2:
        return 0.0

    u = pts[:, 0]
    v = pts[:, 1]

    # Check orientation: if line is predominantly vertical, fit x = m*y + b
    u_spread = float(np.max(u) - np.min(u))
    v_spread = float(np.max(v) - np.min(v))

    if u_spread >= v_spread:
        # Standard touchline: y = m*x + b
        poly = np.polyfit(u, v, deg=1)
        v_pred = np.polyval(poly, u)
        rmse = float(np.sqrt(np.mean((v - v_pred) ** 2)))
    else:
        # Predominantly vertical line: x = m*y + b
        poly = np.polyfit(v, u, deg=1)
        u_pred = np.polyval(poly, v)
        rmse = float(np.sqrt(np.mean((u - u_pred) ** 2)))

    return rmse


def create_comparison_montage(
    original_frame: np.ndarray,
    undistorted_frame: np.ndarray,
    points_distorted: Optional[np.ndarray] = None,
    points_undistorted: Optional[np.ndarray] = None,
    rmse_distorted: Optional[float] = None,
    rmse_undistorted: Optional[float] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Generate side-by-side comparison visualization and save artifact.

    Args:
        original_frame: Raw distorted frame (H, W, 3).
        undistorted_frame: Undistorted frame (H, W, 3).
        points_distorted: Optional (N, 2) sampled points along distorted touchline.
        points_undistorted: Optional (N, 2) sampled points along undistorted touchline.
        rmse_distorted: Measured RMSE before undistortion.
        rmse_undistorted: Measured RMSE after undistortion.
        output_path: Destination path to save JPEG artifact.

    Returns:
        Annotated comparison montage image array.
    """
    h, w = original_frame.shape[:2]

    # Create annotated copies
    vis_orig = original_frame.copy()
    vis_undist = undistorted_frame.copy()

    # Draw touchline points on distorted frame (red curved trace)
    if points_distorted is not None:
        pts_d = points_distorted.reshape(-1, 2).astype(np.int32)
        for i in range(len(pts_d) - 1):
            cv2.line(vis_orig, tuple(pts_d[i]), tuple(pts_d[i + 1]), (0, 0, 255), 3, cv2.LINE_AA)
        for pt in pts_d:
            cv2.circle(vis_orig, tuple(pt), 5, (0, 165, 255), -1, cv2.LINE_AA)

    # Draw touchline points on undistorted frame (green straight trace)
    if points_undistorted is not None:
        pts_u = points_undistorted.reshape(-1, 2).astype(np.int32)
        for i in range(len(pts_u) - 1):
            cv2.line(vis_undist, tuple(pts_u[i]), tuple(pts_u[i + 1]), (0, 255, 0), 3, cv2.LINE_AA)
        for pt in pts_u:
            cv2.circle(vis_undist, tuple(pt), 5, (0, 255, 255), -1, cv2.LINE_AA)

    # Header labels
    header_h = 70
    montage_w = w * 2
    montage_h = h + header_h
    montage = np.zeros((montage_h, montage_w, 3), dtype=np.uint8)

    # Place frames
    montage[header_h:, :w] = vis_orig
    montage[header_h:, w:] = vis_undist

    # Banner backgrounds
    cv2.rectangle(montage, (0, 0), (w, header_h), (25, 25, 35), -1)
    cv2.rectangle(montage, (w, 0), (montage_w, header_h), (20, 40, 20), -1)

    # Add text badges
    dist_label = "RAW DISTORTED (Ultra-Wide Barrel Lens)"
    if rmse_distorted is not None:
        dist_label += f" | Near Touchline Bow RMSE: {rmse_distorted:.2f}px"

    undist_label = "UNDISTORTED (Rectified Pinhole Model)"
    if rmse_undistorted is not None:
        undist_label += f" | Straight Line Fit RMSE: {rmse_undistorted:.4f}px"

    cv2.putText(
        montage,
        dist_label,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (100, 100, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        montage,
        undist_label,
        (w + 30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (120, 255, 120),
        2,
        cv2.LINE_AA,
    )

    # Save to disk if requested
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), montage)

    return montage
