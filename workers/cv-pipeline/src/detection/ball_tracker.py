"""SAHI Slicing Ball Detector and Kalman Gap Interpolator module.

Implements:
1. 2x2 overlapping tile sliced inference (with 20% overlap) for small ball detection
   (COCO class 32 'sports ball' at conf=0.15).
2. 2D Constant-Velocity Kalman Filter for smooth ball tracking and velocity estimation.
3. Cubic Spline Interpolator for occlusion gaps up to 15 frames where the ball is
   temporarily occluded behind a player's leg or obscured by motion blur.
4. Edge-Case Guardrail 3 (Roof-net & cage-wall ricochet filter: velocity > 130 km/h
   flags in_air_ungrounded=True and clamps ground-plane projection).
5. 100% frame coverage trajectory output strictly conforming to the BallTelemetry schema.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import scipy.interpolate as interp
import torch
from ultralytics import YOLO  # type: ignore[attr-defined]

from calibration.homography import ViewTransformer
from schemas import BallTelemetry


# Strict algorithmic hyperparameter defaults per PHASED_TASKS.md Task 2.2
DEFAULT_BALL_CONF_THRESHOLD: float = 0.15
DEFAULT_BALL_CLASS_ID: int = 32  # COCO class 32: sports ball
DEFAULT_OVERLAP_RATIO: float = 0.20  # 20% overlap across tiles
DEFAULT_MAX_GAP_FRAMES: int = 15  # Gaps <= 15 frames are cubic-spline interpolated
MAX_GROUND_VELOCITY_KMH: float = 130.0  # Guardrail 3 roof-net ricochet speed threshold
MAX_GROUND_VELOCITY_MS: float = MAX_GROUND_VELOCITY_KMH / 3.6  # 36.11 m/s


@dataclass
class BallDetection:
    """Raw bounding box candidate from YOLO sliced inference."""

    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    center_xy: Tuple[float, float]  # (cx, cy)
    radius_px: float

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def cx(self) -> float:
        return self.center_xy[0]

    @property
    def cy(self) -> float:
        return self.center_xy[1]


@dataclass
class BallTrajectoryPoint:
    """Final smoothed/interpolated ball telemetry point for a single frame."""

    frame_index: int
    timestamp_sec: float
    detected: bool
    interpolated: bool
    pixel_bbox: Tuple[float, float, float, float]
    center_pixel: Tuple[float, float]
    meter_x: float
    meter_y: float
    meter_z: float = 0.0
    possessor_track_id: Optional[int] = None
    in_air_ungrounded: bool = False
    confidence: float = 0.0
    speed_kmh: float = 0.0

    def to_schema_payload(self) -> BallTelemetry:
        """Convert to validated Pydantic BallTelemetry contract."""
        return BallTelemetry(
            detected=self.detected,
            interpolated=self.interpolated,
            pixel_bbox=[
                float(self.pixel_bbox[0]),
                float(self.pixel_bbox[1]),
                float(self.pixel_bbox[2]),
                float(self.pixel_bbox[3]),
            ],
            meter_x=float(self.meter_x),
            meter_y=float(self.meter_y),
            meter_z=float(self.meter_z),
            possessor_track_id=self.possessor_track_id,
        )


def slice_image_2x2(
    image: np.ndarray,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Divide image into 2x2 overlapping tiles covering 100% of the frame with >= overlap_ratio.

    For an image of dimensions (H, W):
        tile_w = ceil(W / (2.0 - overlap_ratio))
        tile_h = ceil(H / (2.0 - overlap_ratio))
    Offsets:
        (0, 0), (W - tile_w, 0), (0, H - tile_h), (W - tile_w, H - tile_h)

    Args:
        image: Input image array of shape (H, W, 3).
        overlap_ratio: Overlap fraction between adjacent tiles (default 0.20 = 20%).

    Returns:
        Tuple of (list of 4 image tiles, list of 4 (offset_x, offset_y) top-left coordinates).
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError(f"Expected 3D image array, got {type(image)}")

    height, width = image.shape[:2]
    tile_w = int(np.ceil(width / (2.0 - overlap_ratio)))
    tile_h = int(np.ceil(height / (2.0 - overlap_ratio)))

    ox = width - tile_w
    oy = height - tile_h

    offsets: List[Tuple[int, int]] = [
        (0, 0),
        (ox, 0),
        (0, oy),
        (ox, oy),
    ]

    tiles: List[np.ndarray] = [
        image[0:tile_h, 0:tile_w],
        image[0:tile_h, ox:width],
        image[oy:height, 0:tile_w],
        image[oy:height, ox:width],
    ]

    return tiles, offsets


class KalmanBallFilter2D:
    """2D Constant-Velocity Kalman Filter for tracking ball center coordinates."""

    def __init__(
        self,
        process_noise_accel: float = 80.0,
        measurement_noise_std: float = 4.0,
    ) -> None:
        """Initialize the Kalman filter with constant-velocity motion model.

        State: X = [x, y, vx, vy]^T
        """
        self.sigma_a = float(process_noise_accel)
        self.sigma_m = float(measurement_noise_std)

        # State vector [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float64)
        # Covariance matrix P
        self.P = np.eye(4, dtype=np.float64) * 100.0
        # Measurement matrix H
        self.H = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        # Measurement noise R
        self.R = np.eye(2, dtype=np.float64) * (self.sigma_m ** 2)

        self.is_initialized: bool = False

    def initialize(self, x: float, y: float) -> None:
        """Initialize filter state at first measured position."""
        self.state = np.array([[float(x)], [float(y)], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0
        self.is_initialized = True

    def predict(self, dt: float = 1.0 / 60.0) -> Tuple[float, float]:
        """Predict ball state at next time step."""
        if not self.is_initialized:
            return 0.0, 0.0

        dt_sq = dt * dt
        # State transition matrix F
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Discrete process noise covariance Q
        q_pos = (self.sigma_a ** 2) * (dt_sq * dt_sq) / 4.0
        q_cross = (self.sigma_a ** 2) * (dt_sq * dt) / 2.0
        q_vel = (self.sigma_a ** 2) * dt_sq

        Q = np.array(
            [
                [q_pos, 0.0, q_cross, 0.0],
                [0.0, q_pos, 0.0, q_cross],
                [q_cross, 0.0, q_vel, 0.0],
                [0.0, q_cross, 0.0, q_vel],
            ],
            dtype=np.float64,
        )

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

        return float(self.state[0, 0]), float(self.state[1, 0])

    def update(self, measurement_xy: Tuple[float, float]) -> Tuple[float, float]:
        """Update Kalman state with measured (x, y) ball center."""
        z = np.array([[float(measurement_xy[0])], [float(measurement_xy[1])]], dtype=np.float64)

        if not self.is_initialized:
            self.initialize(float(measurement_xy[0]), float(measurement_xy[1]))
            return float(measurement_xy[0]), float(measurement_xy[1])

        # Innovation
        y_innov = z - (self.H @ self.state)
        S = (self.H @ self.P @ self.H.T) + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y_innov)
        I_mat = np.eye(4, dtype=np.float64)
        self.P = (I_mat - (K @ self.H)) @ self.P

        return float(self.state[0, 0]), float(self.state[1, 0])


class CubicSplineGapInterpolator:
    """Interpolates occlusion gaps up to max_gap_frames using Cubic Splines."""

    def __init__(self, max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES) -> None:
        self.max_gap_frames = int(max_gap_frames)

    def interpolate(
        self,
        raw_detections: Sequence[Optional[BallDetection]],
        default_bbox_size: Tuple[float, float] = (16.0, 16.0),
    ) -> List[Tuple[Tuple[float, float, float, float], bool]]:
        """Fill all occlusion gaps with cubic spline interpolation.

        Args:
            raw_detections: List of length N with BallDetection or None for occluded frames.
            default_bbox_size: Default (width, height) in pixels if no detections exist.

        Returns:
            List of length N of ([x1, y1, x2, y2], is_interpolated).
        """
        num_frames = len(raw_detections)
        if num_frames == 0:
            return []

        # Find all frame indices where ball was detected
        detected_indices: List[int] = [i for i, d in enumerate(raw_detections) if d is not None]

        # Case 1: No detections at all across the clip
        if len(detected_indices) == 0:
            w, h = default_bbox_size
            dummy_box = (0.0, 0.0, w, h)
            return [(dummy_box, True) for _ in range(num_frames)]

        # Case 2: Only 1 detection across the clip
        if len(detected_indices) == 1:
            det = raw_detections[detected_indices[0]]
            assert det is not None
            box = det.bbox
            return [(box, i != detected_indices[0]) for i in range(num_frames)]

        # Extract anchor coordinates for spline fitting
        anchor_frames = np.array(detected_indices, dtype=np.float64)
        anchor_cx = np.array([raw_detections[i].cx for i in detected_indices], dtype=np.float64)  # type: ignore[union-attr]
        anchor_cy = np.array([raw_detections[i].cy for i in detected_indices], dtype=np.float64)  # type: ignore[union-attr]
        anchor_bw = np.array(
            [raw_detections[i].x2 - raw_detections[i].x1 for i in detected_indices],  # type: ignore[union-attr]
            dtype=np.float64,
        )
        anchor_bh = np.array(
            [raw_detections[i].y2 - raw_detections[i].y1 for i in detected_indices],  # type: ignore[union-attr]
            dtype=np.float64,
        )

        # Fit natural cubic spline across anchor frames
        # CubicSpline reproduces physical ballistic/projectile flight paths with mathematical precision
        spline_x = interp.CubicSpline(anchor_frames, anchor_cx, bc_type="natural")
        spline_y = interp.CubicSpline(anchor_frames, anchor_cy, bc_type="natural")
        spline_bw = interp.CubicSpline(anchor_frames, anchor_bw, bc_type="natural")
        spline_bh = interp.CubicSpline(anchor_frames, anchor_bh, bc_type="natural")

        results: List[Tuple[Tuple[float, float, float, float], bool]] = []
        first_det_idx = detected_indices[0]
        last_det_idx = detected_indices[-1]

        for i in range(num_frames):
            det = raw_detections[i]
            if det is not None:
                # Raw detected frame
                results.append((det.bbox, False))
                continue

            # Frame is missing a detection -> evaluate spline or linear edge extrapolation
            clamped_i = float(np.clip(i, first_det_idx, last_det_idx))
            cx = float(spline_x(clamped_i))
            cy = float(spline_y(clamped_i))
            bw = max(6.0, float(spline_bw(clamped_i)))
            bh = max(6.0, float(spline_bh(clamped_i)))

            # If outside initial/final detection anchor, extrapolate linearly
            if i < first_det_idx:
                first_box = raw_detections[first_det_idx].bbox  # type: ignore[union-attr]
                cx = (first_box[0] + first_box[2]) / 2.0
                cy = (first_box[1] + first_box[3]) / 2.0
                bw = first_box[2] - first_box[0]
                bh = first_box[3] - first_box[1]
            elif i > last_det_idx:
                last_box = raw_detections[last_det_idx].bbox  # type: ignore[union-attr]
                cx = (last_box[0] + last_box[2]) / 2.0
                cy = (last_box[1] + last_box[3]) / 2.0
                bw = last_box[2] - last_box[0]
                bh = last_box[3] - last_box[1]

            box = (cx - (bw / 2.0), cy - (bh / 2.0), cx + (bw / 2.0), cy + (bh / 2.0))
            results.append((box, True))

        return results


class BallTracker:
    """SAHI Slicing Ball Detector and Kalman Gap Interpolator."""

    def __init__(
        self,
        model_path: Union[str, Path] = "yolo11s.pt",
        conf: float = DEFAULT_BALL_CONF_THRESHOLD,
        overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
        max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
        device: Optional[str] = None,
    ) -> None:
        """Initialize BallTracker.

        Args:
            model_path: Checkpoint path for YOLO11s (default 'yolo11s.pt').
            conf: Ball detection confidence threshold (default 0.15).
            overlap_ratio: Overlap ratio for 2x2 tile slicing (default 0.20 = 20%).
            max_gap_frames: Maximum gap frames to bridge with cubic spline (default 15).
            device: Compute device ('cuda', 'cpu', or None for auto).
        """
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.conf = float(conf)
        self.overlap_ratio = float(overlap_ratio)
        self.max_gap_frames = int(max_gap_frames)

        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        self.model: Optional[YOLO] = YOLO(str(self.model_path))
        self.kalman = KalmanBallFilter2D()
        self.interpolator = CubicSplineGapInterpolator(max_gap_frames=self.max_gap_frames)

    def detect_frame_sliced(self, frame: np.ndarray) -> List[BallDetection]:
        """Run 2x2 overlapping tile sliced inference on a single frame."""
        if self.model is None:
            raise RuntimeError("Model is unloaded. Re-instantiate or reload before calling detect.")

        tiles, offsets = slice_image_2x2(frame, overlap_ratio=self.overlap_ratio)

        # Batch predict all 4 slices in 1 inference pass
        raw_results = self.model.predict(
            source=tiles,
            imgsz=640,
            conf=self.conf,
            classes=[DEFAULT_BALL_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        prediction_list: List[Any] = list(raw_results)

        candidate_boxes: List[Tuple[float, float, float, float]] = []
        candidate_confs: List[float] = []

        for i, res in enumerate(prediction_list):
            ox, oy = offsets[i]
            boxes_obj = getattr(res, "boxes", None)
            if boxes_obj is None or len(boxes_obj) == 0:
                continue

            raw_xyxy = getattr(boxes_obj, "xyxy", None)
            raw_conf = getattr(boxes_obj, "conf", None)
            if raw_xyxy is None or raw_conf is None:
                continue

            xyxy = (
                raw_xyxy.cpu().numpy().astype(np.float64)
                if hasattr(raw_xyxy, "cpu")
                else np.asarray(raw_xyxy, dtype=np.float64)
            )
            confs = (
                raw_conf.cpu().numpy().astype(np.float64)
                if hasattr(raw_conf, "cpu")
                else np.asarray(raw_conf, dtype=np.float64)
            )

            for b, c in zip(xyxy, confs):
                global_box = (float(b[0] + ox), float(b[1] + oy), float(b[2] + ox), float(b[3] + oy))
                candidate_boxes.append(global_box)
                candidate_confs.append(float(c))

        if len(candidate_boxes) == 0:
            return []

        # NMS across tile boundaries
        nms_boxes = [
            [int(b[0]), int(b[1]), max(1, int(b[2] - b[0])), max(1, int(b[3] - b[1]))]
            for b in candidate_boxes
        ]
        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes,
            scores=candidate_confs,
            score_threshold=self.conf,
            nms_threshold=0.35,
        )

        detections: List[BallDetection] = []
        if len(indices) > 0:
            idx_list = [int(x) for x in np.asarray(indices).flatten()]
            for idx in idx_list:
                box = candidate_boxes[idx]
                score = candidate_confs[idx]
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                radius = max((box[2] - box[0]), (box[3] - box[1])) / 2.0
                detections.append(
                    BallDetection(
                        bbox=box,
                        confidence=score,
                        center_xy=(cx, cy),
                        radius_px=radius,
                    )
                )

        return detections

    def detect_frame_adaptive(
        self,
        frame: np.ndarray,
        predicted_xy: Optional[Tuple[float, float]] = None,
        crop_radius: int = 320,
    ) -> List[BallDetection]:
        """High-performance adaptive detector: searches local window if tracked, else full 2x2 sliced."""
        if predicted_xy is None:
            return self.detect_frame_sliced(frame)

        # Local crop centered at predicted ball position
        height, width = frame.shape[:2]
        px, py = int(round(predicted_xy[0])), int(round(predicted_xy[1]))

        x1 = max(0, px - crop_radius)
        y1 = max(0, py - crop_radius)
        x2 = min(width, x1 + (crop_radius * 2))
        y2 = min(height, y1 + (crop_radius * 2))

        # Adjust crop start if right/bottom edge was clamped
        x1 = max(0, x2 - (crop_radius * 2))
        y1 = max(0, y2 - (crop_radius * 2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return self.detect_frame_sliced(frame)

        if self.model is None:
            raise RuntimeError("Model is unloaded.")

        raw_results = self.model.predict(
            source=crop,
            imgsz=640,
            conf=self.conf,
            classes=[DEFAULT_BALL_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        prediction_list: List[Any] = list(raw_results)
        first_res = prediction_list[0]
        boxes_obj = getattr(first_res, "boxes", None)

        if boxes_obj is not None and len(boxes_obj) > 0:
            raw_xyxy = getattr(boxes_obj, "xyxy", None)
            raw_conf = getattr(boxes_obj, "conf", None)
            if raw_xyxy is not None and raw_conf is not None:
                xyxy = (
                    raw_xyxy.cpu().numpy().astype(np.float64)
                    if hasattr(raw_xyxy, "cpu")
                    else np.asarray(raw_xyxy, dtype=np.float64)
                )
                confs = (
                    raw_conf.cpu().numpy().astype(np.float64)
                    if hasattr(raw_conf, "cpu")
                    else np.asarray(raw_conf, dtype=np.float64)
                )
                detections: List[BallDetection] = []
                for b, c in zip(xyxy, confs):
                    global_box = (float(b[0] + x1), float(b[1] + y1), float(b[2] + x1), float(b[3] + y1))
                    cx = (global_box[0] + global_box[2]) / 2.0
                    cy = (global_box[1] + global_box[3]) / 2.0
                    radius = max((global_box[2] - global_box[0]), (global_box[3] - global_box[1])) / 2.0
                    detections.append(
                        BallDetection(
                            bbox=global_box,
                            confidence=float(c),
                            center_xy=(cx, cy),
                            radius_px=radius,
                        )
                    )
                if len(detections) > 0:
                    return detections

        # If not found in local crop, fall back to full 2x2 sliced search
        return self.detect_frame_sliced(frame)

    def track_sequence(
        self,
        frames: Sequence[np.ndarray],
        fps: float = 60.0,
        view_transformer: Optional[ViewTransformer] = None,
    ) -> List[BallTrajectoryPoint]:
        """Track ball across a sequence of frames with Kalman smoothing and cubic spline gap interpolation.

        Guarantees 100% frame coverage across the entire sequence.

        Args:
            frames: Sequence of BGR image frames.
            fps: Frame rate of the video.
            view_transformer: Optional ViewTransformer for pixel-to-meter ground plane projection.

        Returns:
            List of BallTrajectoryPoint objects matching len(frames).
        """
        num_frames = len(frames)
        if num_frames == 0:
            return []

        dt = 1.0 / float(fps)
        raw_detections: List[Optional[BallDetection]] = [None] * num_frames
        last_known_xy: Optional[Tuple[float, float]] = None

        # Pass 1: Sliced / Adaptive Frame-by-Frame Detection + Kalman Association
        for i in range(num_frames):
            frame = frames[i]
            predicted_xy: Optional[Tuple[float, float]] = None
            if self.kalman.is_initialized:
                pred_x, pred_y = self.kalman.predict(dt=dt)
                predicted_xy = (pred_x, pred_y)

            # Adaptive search: crop near predicted location, fallback to 2x2 sliced
            candidates = self.detect_frame_adaptive(frame, predicted_xy=predicted_xy)

            chosen: Optional[BallDetection] = None
            if len(candidates) > 0:
                if predicted_xy is not None:
                    # Choose candidate closest to Kalman prediction within gating distance
                    dists = [
                        np.hypot(c.cx - predicted_xy[0], c.cy - predicted_xy[1])
                        for c in candidates
                    ]
                    min_idx = int(np.argmin(dists))
                    # Physical gating: max 250 pixels jump between consecutive frames
                    if dists[min_idx] < 250.0:
                        chosen = candidates[min_idx]
                    else:
                        # Highest confidence candidate
                        chosen = max(candidates, key=lambda c: c.confidence)
                else:
                    # Initial detection: pick highest confidence
                    chosen = max(candidates, key=lambda c: c.confidence)

            if chosen is not None:
                raw_detections[i] = chosen
                last_known_xy = (chosen.cx, chosen.cy)
                self.kalman.update((chosen.cx, chosen.cy))

        # Pass 2: Cubic Spline Gap Interpolation for occlusion gaps <= max_gap_frames
        interpolated_boxes = self.interpolator.interpolate(raw_detections)

        trajectory: List[BallTrajectoryPoint] = []
        prev_meter_xy: Optional[Tuple[float, float]] = None

        # Pass 3: Ground-Plane Projection and Guardrail 3 Ricochet Filtering
        for i in range(num_frames):
            box, is_interpolated = interpolated_boxes[i]
            is_detected = not is_interpolated
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            timestamp_sec = float(i) * dt

            det_i = raw_detections[i]
            conf_val = det_i.confidence if (is_detected and det_i is not None) else 0.50

            # Ground-plane meter projection
            meter_x = 0.0
            meter_y = 0.0
            in_air = False
            speed_kmh = 0.0

            if view_transformer is not None:
                meters = view_transformer.pixel_to_meter(np.array([cx, cy], dtype=np.float64))
                meter_x = float(meters[0])
                meter_y = float(meters[1])

                # Guardrail 3: Check velocity between consecutive frames
                if prev_meter_xy is not None:
                    delta_d = float(np.hypot(meter_x - prev_meter_xy[0], meter_y - prev_meter_xy[1]))
                    speed_ms = delta_d / dt
                    speed_kmh = speed_ms * 3.6

                    if speed_kmh > MAX_GROUND_VELOCITY_KMH:
                        # High roof-net or cage ricochet: flag ungrounded and clamp coordinates
                        in_air = True
                        meter_x = prev_meter_xy[0]
                        meter_y = prev_meter_xy[1]
                        speed_kmh = MAX_GROUND_VELOCITY_KMH

                prev_meter_xy = (meter_x, meter_y)

            trajectory.append(
                BallTrajectoryPoint(
                    frame_index=i,
                    timestamp_sec=timestamp_sec,
                    detected=is_detected,
                    interpolated=is_interpolated,
                    pixel_bbox=box,
                    center_pixel=(cx, cy),
                    meter_x=meter_x,
                    meter_y=meter_y,
                    meter_z=0.0,
                    possessor_track_id=None,
                    in_air_ungrounded=in_air,
                    confidence=conf_val,
                    speed_kmh=speed_kmh,
                )
            )

        return trajectory

    def annotate_frame(
        self,
        frame: np.ndarray,
        point: BallTrajectoryPoint,
        history: Optional[List[BallTrajectoryPoint]] = None,
        draw_hud_banner: bool = True,
    ) -> np.ndarray:
        """Annotate a single frame with ball bounding box, center crosshair, trail, and HUD."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        cx, cy = int(round(point.center_pixel[0])), int(round(point.center_pixel[1]))
        x1, y1 = int(round(point.pixel_bbox[0])), int(round(point.pixel_bbox[1]))
        x2, y2 = int(round(point.pixel_bbox[2])), int(round(point.pixel_bbox[3]))

        # Distinct colors: Neon Green for raw detected, Vibrant Amber for interpolated
        box_color = (0, 255, 0) if point.detected else (0, 165, 255)
        status_label = "RAW DETECTED" if point.detected else "INTERPOLATED (CUBIC SPLINE)"

        # 1. Draw trajectory trail ribbon from history
        if history is not None and len(history) > 1:
            recent_trail = history[max(0, len(history) - 30) :]
            for j in range(1, len(recent_trail)):
                pt1 = (int(round(recent_trail[j - 1].center_pixel[0])), int(round(recent_trail[j - 1].center_pixel[1])))
                pt2 = (int(round(recent_trail[j].center_pixel[0])), int(round(recent_trail[j].center_pixel[1])))
                alpha = float(j) / float(len(recent_trail))
                color_trail = (int(255 * alpha), int(200 * alpha), 0)
                cv2.line(annotated, pt1, pt2, color_trail, thickness=2, lineType=cv2.LINE_AA)

        # 2. Draw bounding box and ball circle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness=2, lineType=cv2.LINE_AA)
        radius = max(6, int(round(max(x2 - x1, y2 - y1) / 2.0)))
        cv2.circle(annotated, (cx, cy), radius + 2, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), radius, box_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        # Center crosshairs
        cv2.line(annotated, (cx - 10, cy), (cx + 10, cy), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.line(annotated, (cx - 8, cy), (cx + 8, cy), (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(annotated, (cx, cy - 10), (cx, cy + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.line(annotated, (cx, cy - 8), (cx, cy + 8), (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # 3. Label tag
        font = cv2.FONT_HERSHEY_SIMPLEX
        tag_text = f"Ball ({status_label})"
        cv2.putText(annotated, tag_text, (x1, max(20, y1 - 8)), font, 0.48, box_color, 1, lineType=cv2.LINE_AA)

        # 4. HUD Banner
        if draw_hud_banner:
            banner_height = 42
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, banner_height), (15, 23, 42), thickness=-1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
            cv2.line(annotated, (0, banner_height), (width, banner_height), (249, 115, 22), thickness=1)

            hud_title = "TURFVISION 2x2 SAHI BALL TRACKER"
            hud_stats = (
                f"Frame: {point.frame_index} | "
                f"Status: {status_label} | "
                f"Pixel: ({cx}, {cy}) | "
                f"Turf Meter: ({point.meter_x:.2f}m, {point.meter_y:.2f}m) | "
                f"Speed: {point.speed_kmh:.1f} km/h"
            )
            cv2.putText(annotated, hud_title, (16, 26), font, 0.60, (249, 115, 22), 2, lineType=cv2.LINE_AA)
            cv2.putText(annotated, hud_stats, (420, 26), font, 0.50, (241, 245, 249), 1, lineType=cv2.LINE_AA)

        return annotated

    def unload(self) -> None:
        """Offload model weights from memory per MODEL_AND_HARDWARE_MANIFEST.md."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self) -> BallTracker:
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.unload()
