"""Multi-Object Player Tracker with Ground-Plane Meter Projection and Kalman Smoothing.

Wraps ByteTrack (track_activation_threshold=0.25, lost_track_buffer=60, minimum_matching_threshold=0.8)
with ground-plane projection and 2D spatial consistency to track players across
an 8v8 football pitch with minimal ID fragmentation and zero out-of-bounds leakage.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics.trackers.byte_tracker import BYTETracker

from calibration.homography import ViewTransformer
from detection.player_detector import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_IOU_THRESHOLD,
    PlayerDetection,
    PlayerDetectionResult,
    PlayerDetector,
    extract_foot_coordinates,
)
from schemas import PlayerTelemetry, RoleEnum


# Strict algorithmic hyperparameter defaults per PHASED_TASKS.md Task 2.3
DEFAULT_TRACK_ACTIVATION_THRESHOLD: float = 0.25
DEFAULT_LOST_TRACK_BUFFER: int = 60  # Retain lost tracks for 60 frames (1 second at 60fps)
DEFAULT_MINIMUM_MATCHING_THRESHOLD: float = 0.80
DEFAULT_FRAME_RATE: int = 60

# Turf bounds per PHASED_TASKS.md Task 2.3: pitch [-0.5, 50.5] x [-0.5, 32.5] with 2m safety margin
PITCH_X_MIN_ALLOWED: float = -2.0
PITCH_X_MAX_ALLOWED: float = 52.0
PITCH_Y_MIN_ALLOWED: float = -2.0
PITCH_Y_MAX_ALLOWED: float = 34.0


@dataclass
class TrackedPlayer:
    """Represents a persistent player track in camera pixels and physical turf meters."""

    track_id: int
    pixel_bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    foot_pixel: Tuple[float, float]  # [u_c, v_max]
    meter_x: float
    meter_y: float
    speed_kmh: float = 0.0
    is_sprinting: bool = False
    role: RoleEnum = RoleEnum.PLAYER
    team_id: int = 0

    @property
    def x1(self) -> float:
        return self.pixel_bbox[0]

    @property
    def y1(self) -> float:
        return self.pixel_bbox[1]

    @property
    def x2(self) -> float:
        return self.pixel_bbox[2]

    @property
    def y2(self) -> float:
        return self.pixel_bbox[3]

    def to_player_telemetry(self) -> PlayerTelemetry:
        """Convert to validated Pydantic PlayerTelemetry contract."""
        return PlayerTelemetry(
            track_id=self.track_id,
            team_id=self.team_id,
            role=self.role,
            confidence=float(np.clip(self.confidence, 0.0, 1.0)),
            pixel_bbox=[
                float(self.pixel_bbox[0]),
                float(self.pixel_bbox[1]),
                float(self.pixel_bbox[2]),
                float(self.pixel_bbox[3]),
            ],
            meter_x=float(self.meter_x),
            meter_y=float(self.meter_y),
            speed_kmh=float(np.clip(self.speed_kmh, 0.0, 36.0)),
            is_sprinting=self.is_sprinting,
        )


class GroundPlaneKalmanSmoother:
    """Ground-plane 2D Kalman filter per track to eliminate bounding box flicker."""

    def __init__(self, dt: float = 1.0 / 60.0) -> None:
        self.dt = float(dt)
        # track_id -> (state [x, y, vx, vy]^T, covariance P (4x4))
        self.filters: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.last_timestamps: Dict[int, float] = {}

    def update(self, track_id: int, meter_x: float, meter_y: float) -> Tuple[float, float, float]:
        """Update track filter with new ground-plane meter coordinates.

        Returns:
            Tuple of (smoothed_meter_x, smoothed_meter_y, speed_kmh).
        """
        z = np.array([[float(meter_x)], [float(meter_y)]], dtype=np.float64)

        if track_id not in self.filters:
            state = np.array([[float(meter_x)], [float(meter_y)], [0.0], [0.0]], dtype=np.float64)
            P = np.eye(4, dtype=np.float64) * 5.0
            self.filters[track_id] = (state, P)
            return float(meter_x), float(meter_y), 0.0

        state, P = self.filters[track_id]

        # Transition matrix
        dt = self.dt
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        Q = np.eye(4, dtype=np.float64) * 0.1
        R = np.eye(2, dtype=np.float64) * 0.2

        # Predict
        state_pred = F @ state
        P_pred = F @ P @ F.T + Q

        # Update
        y_innov = z - (H @ state_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        state_up = state_pred + (K @ y_innov)
        P_up = (np.eye(4, dtype=np.float64) - (K @ H)) @ P_pred

        self.filters[track_id] = (state_up, P_up)

        vx = float(state_up[2, 0])
        vy = float(state_up[3, 0])
        speed_ms = float(np.hypot(vx, vy))
        speed_kmh = speed_ms * 3.6

        return float(state_up[0, 0]), float(state_up[1, 0]), speed_kmh


class GroundPlaneReidentifier:
    """Maintains track continuity across occlusions using ground-plane physical proximity."""

    def __init__(self, max_reid_distance_m: float = 2.5, max_lost_frames: int = 60) -> None:
        self.max_dist = float(max_reid_distance_m)
        self.max_frames = int(max_lost_frames)
        # raw_id -> canonical_id
        self.id_remap: Dict[int, int] = {}
        # canonical_id -> (last_meter_x, last_meter_y, last_frame_idx)
        self.recent_tracks: Dict[int, Tuple[float, float, int]] = {}

    def get_canonical_id(self, raw_id: int, meter_x: float, meter_y: float, frame_idx: int) -> int:
        """Map raw ByteTrack ID to a persistent canonical ID based on spatial proximity."""
        if raw_id in self.id_remap:
            can_id = self.id_remap[raw_id]
            self.recent_tracks[can_id] = (meter_x, meter_y, frame_idx)
            return can_id

        # Search for a recently lost track within spatial proximity
        best_match_id: Optional[int] = None
        min_dist = float("inf")

        for lost_id, (lx, ly, lframe) in self.recent_tracks.items():
            if frame_idx - lframe <= self.max_frames:
                d = float(np.hypot(meter_x - lx, meter_y - ly))
                if d < self.max_dist and d < min_dist:
                    min_dist = d
                    best_match_id = lost_id

        if best_match_id is not None:
            self.id_remap[raw_id] = best_match_id
            self.recent_tracks[best_match_id] = (meter_x, meter_y, frame_idx)
            return best_match_id

        self.id_remap[raw_id] = raw_id
        self.recent_tracks[raw_id] = (meter_x, meter_y, frame_idx)
        return raw_id


class BoxesWrapper:
    """Lightweight wrapper exposing xywh, conf, and cls with boolean indexing for BYTETracker."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.xywh)

    def __getitem__(self, mask: Any) -> BoxesWrapper:
        return BoxesWrapper(self.xywh[mask], self.conf[mask], self.cls[mask])


class ByteTrack:
    """Supervision-compatible ByteTrack wrapper with strict hyperparameter defaults."""

    def __init__(
        self,
        track_activation_threshold: float = DEFAULT_TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer: int = DEFAULT_LOST_TRACK_BUFFER,
        minimum_matching_threshold: float = DEFAULT_MINIMUM_MATCHING_THRESHOLD,
        frame_rate: int = DEFAULT_FRAME_RATE,
    ) -> None:
        self.track_activation_threshold = float(track_activation_threshold)
        self.lost_track_buffer = int(lost_track_buffer)
        self.minimum_matching_threshold = float(minimum_matching_threshold)
        self.frame_rate = int(frame_rate)

        args = SimpleNamespace(
            track_high_thresh=self.track_activation_threshold,
            track_low_thresh=0.10,
            new_track_thresh=self.track_activation_threshold,
            track_buffer=self.lost_track_buffer,
            match_thresh=self.minimum_matching_threshold,
            fuse_score=True,
        )
        self.tracker = BYTETracker(args)  # type: ignore[no-untyped-call]

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with supervision Detections and return tracked detections."""
        if len(detections) == 0:
            return sv.Detections.empty()

        xyxy = detections.xyxy
        conf = (
            np.asarray(detections.confidence, dtype=np.float32)
            if detections.confidence is not None
            else np.ones(len(xyxy), dtype=np.float32)
        )
        cls = (
            np.asarray(detections.class_id, dtype=np.float32)
            if detections.class_id is not None
            else np.zeros(len(xyxy), dtype=np.float32)
        )

        # Convert xyxy [x1, y1, x2, y2] to center xywh [cx, cy, w, h]
        x1 = xyxy[:, 0]
        y1 = xyxy[:, 1]
        x2 = xyxy[:, 2]
        y2 = xyxy[:, 3]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = np.maximum(1.0, x2 - x1)
        h = np.maximum(1.0, y2 - y1)
        xywh = np.column_stack([cx, cy, w, h]).astype(np.float32)

        boxes = BoxesWrapper(xywh=xywh, conf=conf, cls=cls)
        tracked_rows = self.tracker.update(boxes)

        if len(tracked_rows) == 0:
            return sv.Detections.empty()

        tracked_xyxy = tracked_rows[:, :4].astype(np.float32)
        tracked_ids = tracked_rows[:, 4].astype(int)
        tracked_conf = tracked_rows[:, 5].astype(np.float32)
        tracked_cls = tracked_rows[:, 6].astype(int)
        tracked_orig_indices = tracked_rows[:, 7].astype(int)

        # Map custom data fields (such as foot coordinates)
        data: Dict[str, Any] = {}
        for key, val in detections.data.items():
            if isinstance(val, (list, np.ndarray)):
                data[key] = np.asarray(val)[tracked_orig_indices]

        return sv.Detections(
            xyxy=tracked_xyxy,
            confidence=tracked_conf,
            class_id=tracked_cls,
            tracker_id=tracked_ids,
            data=data,
        )


class PlayerTracker:
    """Multi-Object Tracker combining YOLO11, ByteTrack, and Ground-Plane Projection."""

    def __init__(
        self,
        view_transformer: ViewTransformer,
        model_path: Union[str, Path] = "yolo11s.pt",
        track_activation_threshold: float = DEFAULT_TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer: int = DEFAULT_LOST_TRACK_BUFFER,
        minimum_matching_threshold: float = DEFAULT_MINIMUM_MATCHING_THRESHOLD,
        frame_rate: int = DEFAULT_FRAME_RATE,
        pitch_margin_m: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        """Initialize PlayerTracker.

        Args:
            view_transformer: ViewTransformer instance for ground-plane projection.
            model_path: Path to YOLO11 model checkpoint.
            track_activation_threshold: Min confidence to activate new track (default 0.25).
            lost_track_buffer: Frames to retain lost tracks (default 60).
            minimum_matching_threshold: Cost threshold for Hungarian matching (default 0.8).
            frame_rate: Frame rate of the video.
            pitch_margin_m: Boundary margin to filter touchline spectators (default 0.5m).
            device: Compute device.
        """
        self.view_transformer = view_transformer
        self.pitch_margin_m = float(pitch_margin_m)
        self.frame_rate = int(frame_rate)
        self.frame_idx: int = 0

        self.detector = PlayerDetector(
            model_path=model_path,
            imgsz=DEFAULT_IMGSZ,
            conf=track_activation_threshold,
            iou=DEFAULT_IOU_THRESHOLD,
            classes=[0],
            device=device,
        )

        self.bytetrack = ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

        self.smoother = GroundPlaneKalmanSmoother(dt=1.0 / float(self.frame_rate))
        self.reid = GroundPlaneReidentifier(max_reid_distance_m=2.5, max_lost_frames=lost_track_buffer)

    def update(
        self,
        frame: np.ndarray,
        detection_result: Optional[PlayerDetectionResult] = None,
    ) -> List[TrackedPlayer]:
        """Update tracker with a new video frame.

        Args:
            frame: Input BGR image frame (H, W, 3).
            detection_result: Optional precomputed PlayerDetectionResult.

        Returns:
            List of TrackedPlayer instances with validated physical coordinates.
        """
        if detection_result is None:
            detection_result = self.detector.detect(
                frame,
                view_transformer=self.view_transformer,
                pitch_margin_m=self.pitch_margin_m,
            )

        # 1. Filter out spectators and subs on touchline per EDGE_CASES Rule 1
        # Only detections inside pitch + 0.5m margin enter ByteTrack
        if detection_result.inside_pitch_mask is not None and len(detection_result.detections) > 0:
            keep_indices = np.flatnonzero(detection_result.inside_pitch_mask)
        else:
            keep_indices = np.arange(len(detection_result.detections))

        if len(keep_indices) == 0:
            self.frame_idx += 1
            return []

        filtered_sv = sv.Detections(
            xyxy=detection_result.sv_detections.xyxy[keep_indices],
            confidence=detection_result.sv_detections.confidence[keep_indices]
            if detection_result.sv_detections.confidence is not None
            else None,
            class_id=detection_result.sv_detections.class_id[keep_indices]
            if detection_result.sv_detections.class_id is not None
            else None,
            data={"foot_coordinates": detection_result.foot_coordinates[keep_indices]},
        )

        # 2. Update ByteTrack
        tracked_sv = self.bytetrack.update_with_detections(filtered_sv)

        if len(tracked_sv) == 0 or tracked_sv.tracker_id is None:
            self.frame_idx += 1
            return []

        # 3. Project tracked foot anchors to physical pitch meters and smooth
        tracked_players: List[TrackedPlayer] = []
        for i in range(len(tracked_sv)):
            raw_id = int(tracked_sv.tracker_id[i])
            box = tuple(float(c) for c in tracked_sv.xyxy[i])
            conf_val = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 0.85
            cls_val = int(tracked_sv.class_id[i]) if tracked_sv.class_id is not None else 0

            # Foot anchor
            u_c = (box[0] + box[2]) / 2.0
            v_max = box[3]
            foot_px = (u_c, v_max)

            # Ground-plane projection
            meter_coords = self.view_transformer.pixel_to_meter(np.array([u_c, v_max], dtype=np.float64))
            raw_mx = float(meter_coords[0])
            raw_my = float(meter_coords[1])

            # Spatial consistency re-identification to eliminate fragmentation
            canon_id = self.reid.get_canonical_id(raw_id, raw_mx, raw_my, self.frame_idx)

            # Ground-plane Kalman smoothing to eliminate pixel jitter
            smooth_mx, smooth_my, speed_kmh = self.smoother.update(canon_id, raw_mx, raw_my)

            # Strict bounds enforcement: all players lie within [-2.0m, 52.0m] x [-2.0m, 34.0m]
            bounded_mx = float(np.clip(smooth_mx, PITCH_X_MIN_ALLOWED, PITCH_X_MAX_ALLOWED))
            bounded_my = float(np.clip(smooth_my, PITCH_Y_MIN_ALLOWED, PITCH_Y_MAX_ALLOWED))

            is_sprint = speed_kmh > 20.0

            tracked_players.append(
                TrackedPlayer(
                    track_id=canon_id,
                    pixel_bbox=(box[0], box[1], box[2], box[3]),
                    confidence=conf_val,
                    class_id=cls_val,
                    foot_pixel=foot_px,
                    meter_x=bounded_mx,
                    meter_y=bounded_my,
                    speed_kmh=speed_kmh,
                    is_sprinting=is_sprint,
                    role=RoleEnum.PLAYER,
                    team_id=0,
                )
            )

        self.frame_idx += 1
        return tracked_players

    def track_clip(
        self,
        frames: Sequence[np.ndarray],
        fps: float = 60.0,
    ) -> List[List[TrackedPlayer]]:
        """Track players across all frames of a video clip."""
        results: List[List[TrackedPlayer]] = []
        for frame in frames:
            step_tracks = self.update(frame)
            results.append(step_tracks)
        return results

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[TrackedPlayer],
        draw_hud_banner: bool = True,
        draw_foot_anchors: bool = True,
    ) -> np.ndarray:
        """Annotate frame with tracked player bounding boxes, IDs, and ground meters."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        for player in tracks:
            x1, y1, x2, y2 = (
                int(round(player.x1)),
                int(round(player.y1)),
                int(round(player.x2)),
                int(round(player.y2)),
            )
            u_c, v_max = int(round(player.foot_pixel[0])), int(round(player.foot_pixel[1]))

            # Unique color per track ID
            np.random.seed(player.track_id)
            color = tuple(int(c) for c in np.random.randint(60, 255, size=3))

            # 1. Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2, lineType=cv2.LINE_AA)

            # 2. Foot anchor point
            if draw_foot_anchors:
                cv2.circle(annotated, (u_c, v_max), 6, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(annotated, (u_c, v_max), 4, color, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(annotated, (u_c, v_max), 1, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

            # 3. Label tag: ID and physical turf position
            label = f"ID:{player.track_id} ({player.meter_x:.1f}m, {player.meter_y:.1f}m)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            lbl_y1 = max(0, y1 - th - baseline - 4)
            cv2.rectangle(annotated, (x1, lbl_y1), (x1 + tw + 6, y1), (20, 20, 20), thickness=-1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - baseline - 1),
                font,
                font_scale,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        # 4. HUD Banner
        if draw_hud_banner:
            banner_height = 42
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, banner_height), (15, 23, 42), thickness=-1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
            cv2.line(annotated, (0, banner_height), (width, banner_height), (16, 185, 129), thickness=1)

            hud_title = "TURFVISION BYTETRACK GROUND-PLANE TRACKER"
            hud_stats = (
                f"Frame: {self.frame_idx} | "
                f"Active Tracks: {len(tracks)} | "
                f"Lost Buffer: {self.bytetrack.lost_track_buffer}f | "
                f"Match Thresh: {self.bytetrack.minimum_matching_threshold}"
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated, hud_title, (16, 26), font, 0.60, (16, 185, 129), 2, lineType=cv2.LINE_AA)
            cv2.putText(annotated, hud_stats, (460, 26), font, 0.50, (241, 245, 249), 1, lineType=cv2.LINE_AA)

        return annotated

    def unload(self) -> None:
        """Offload resources per MODEL_AND_HARDWARE_MANIFEST.md."""
        self.detector.unload()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> PlayerTracker:
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.unload()
