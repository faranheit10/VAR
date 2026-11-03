"""Detection module for 8v8 Turf AI analytics platform."""

from __future__ import annotations

from detection.ball_tracker import (
    DEFAULT_BALL_CLASS_ID,
    DEFAULT_BALL_CONF_THRESHOLD,
    DEFAULT_MAX_GAP_FRAMES,
    DEFAULT_OVERLAP_RATIO,
    MAX_GROUND_VELOCITY_KMH,
    BallDetection,
    BallTracker,
    BallTrajectoryPoint,
    CubicSplineGapInterpolator,
    KalmanBallFilter2D,
    slice_image_2x2,
)
from detection.player_detector import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_IOU_THRESHOLD,
    PlayerDetection,
    PlayerDetectionResult,
    PlayerDetector,
    extract_foot_coordinates,
)

__all__ = [
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_IMGSZ",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_BALL_CONF_THRESHOLD",
    "DEFAULT_BALL_CLASS_ID",
    "DEFAULT_OVERLAP_RATIO",
    "DEFAULT_MAX_GAP_FRAMES",
    "MAX_GROUND_VELOCITY_KMH",
    "BallDetection",
    "BallTracker",
    "BallTrajectoryPoint",
    "CubicSplineGapInterpolator",
    "KalmanBallFilter2D",
    "slice_image_2x2",
    "PlayerDetection",
    "PlayerDetectionResult",
    "PlayerDetector",
    "extract_foot_coordinates",
]
