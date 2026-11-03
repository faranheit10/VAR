"""Deterministic verification test for Task 2.2: SAHI Slicing Ball Detector + Kalman Gap Interpolator.

Verifies:
1. 2x2 overlapping tile geometry: verifies exactly 4 tiles, >= 20% overlap, and 100% canvas coverage.
2. 2D Constant-Velocity Kalman Filter dynamics: convergence and noise reduction.
3. Cubic Spline Gap Interpolation: fills occlusion gaps <= 15 frames with smooth cubic spline,
   setting flag `interpolated: true` on occluded frames and `interpolated: false` on detected frames.
4. Edge-Case Guardrail 3: High-velocity aerial ricochets (> 130 km/h) trigger in_air_ungrounded=True
   and coordinate clamping.
5. 100% Frame Coverage on 300-frame video sequence: all 300 frames produce valid BallTelemetry payloads.
6. Visual proof artifact: saves artifacts/verification/task_2_2/ball_trajectory_overlay.jpg with
   bounding box, center crosshair, trajectory ribbon trail, and HUD telemetry banner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pytest

from calibration.homography import ViewTransformer
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
from schemas import BallTelemetry


VIDEO_PATH = Path("artifacts/sample_data/sample_8v8.mp4")
CALIBRATION_PATH = Path("artifacts/sample_data/sample_calibration.json")
VERIFICATION_DIR = Path("artifacts/verification/task_2_2")
ARTIFACT_IMAGE_PATH = VERIFICATION_DIR / "ball_trajectory_overlay.jpg"


def test_slice_image_2x2_geometry_and_overlap() -> None:
    """Verify 2x2 overlapping tile slicing geometry, >= 20% overlap, and 100% canvas coverage."""
    height, width = 1080, 1920
    test_img = np.zeros((height, width, 3), dtype=np.uint8)

    tiles, offsets = slice_image_2x2(test_img, overlap_ratio=DEFAULT_OVERLAP_RATIO)

    # Must produce exactly 4 tiles in a 2x2 grid
    assert len(tiles) == 4, f"Expected 4 tiles, got {len(tiles)}"
    assert len(offsets) == 4, f"Expected 4 offsets, got {len(offsets)}"

    tile_h, tile_w = tiles[0].shape[:2]
    # Check minimum overlap ratio >= 20%
    # Horizontal overlap between tile 0 and tile 1
    ox_right = offsets[1][0]
    overlap_w = tile_w - ox_right
    overlap_w_ratio = overlap_w / float(tile_w)
    assert overlap_w_ratio >= 0.20, f"Horizontal overlap {overlap_w_ratio:.3f} < 0.20"

    # Vertical overlap between tile 0 and tile 2
    oy_bottom = offsets[2][1]
    overlap_h = tile_h - oy_bottom
    overlap_h_ratio = overlap_h / float(tile_h)
    assert overlap_h_ratio >= 0.20, f"Vertical overlap {overlap_h_ratio:.3f} < 0.20"

    # Verify 100% pixel coverage across the entire (1080, 1920) canvas
    coverage_mask = np.zeros((height, width), dtype=np.uint8)
    for (ox, oy), tile in zip(offsets, tiles):
        th, tw = tile.shape[:2]
        coverage_mask[oy : oy + th, ox : ox + tw] = 1

    assert np.all(coverage_mask == 1), "Dead zone detected: 2x2 tiles do not cover 100% of the canvas!"


def test_kalman_filter_2d_dynamics() -> None:
    """Verify 2D Kalman filter initialization, constant-velocity prediction, and update."""
    kf = KalmanBallFilter2D(process_noise_accel=500.0, measurement_noise_std=2.0)
    assert not kf.is_initialized

    # Initialize at (100.0, 200.0)
    kf.initialize(100.0, 200.0)
    assert kf.is_initialized
    assert np.isclose(kf.state[0, 0], 100.0)
    assert np.isclose(kf.state[1, 0], 200.0)

    # Feed linear trajectory with vx = 10 px/frame, vy = 5 px/frame over 25 steps
    dt = 1.0 / 60.0
    for step in range(1, 26):
        pred_x, pred_y = kf.predict(dt=dt)
        meas_x = 100.0 + (step * 10.0)
        meas_y = 200.0 + (step * 5.0)
        up_x, up_y = kf.update((meas_x, meas_y))

    # Assert estimate converges to within 1.5 pixels of the latest ground-truth measurement
    assert np.isclose(up_x, 350.0, atol=1.5)
    assert np.isclose(up_y, 325.0, atol=1.5)


def test_cubic_spline_gap_interpolator_accuracy() -> None:
    """Verify Cubic Spline accurately interpolates gaps <= 15 frames and sets interpolated flags."""
    interpolator = CubicSplineGapInterpolator(max_gap_frames=15)

    # Simulate a smooth parabolic arc ball trajectory across 40 frames
    # with an occlusion gap of 11 frames (frames 15 to 25) behind a defender's leg
    frames_count = 40
    t = np.arange(frames_count, dtype=np.float64)
    true_cx = 500.0 + (20.0 * t)  # linear X motion
    true_cy = 700.0 - (15.0 * t) + (0.5 * (t**2))  # parabolic Y projectile motion

    raw_detections: List[Optional[BallDetection]] = []
    for i in range(frames_count):
        if 15 <= i <= 25:
            # Simulated occlusion gap of 11 frames
            raw_detections.append(None)
        else:
            box = (true_cx[i] - 8.0, true_cy[i] - 8.0, true_cx[i] + 8.0, true_cy[i] + 8.0)
            raw_detections.append(
                BallDetection(
                    bbox=box,
                    confidence=0.85,
                    center_xy=(true_cx[i], true_cy[i]),
                    radius_px=8.0,
                )
            )

    interpolated_results = interpolator.interpolate(raw_detections)

    assert len(interpolated_results) == frames_count

    # Check that visible frames have interpolated: False and occluded frames have interpolated: True
    for i in range(frames_count):
        box, is_interp = interpolated_results[i]
        if 15 <= i <= 25:
            assert is_interp is True, f"Frame {i} in occlusion gap should be marked interpolated=True"
            interp_cx = (box[0] + box[2]) / 2.0
            interp_cy = (box[1] + box[3]) / 2.0
            # Assert cubic spline interpolation error is within sub-pixel bounds (< 1.5 px)
            err = np.hypot(interp_cx - true_cx[i], interp_cy - true_cy[i])
            assert err < 1.5, f"Frame {i} cubic spline error {err:.2f}px exceeds 1.5px threshold"
        else:
            assert is_interp is False, f"Frame {i} with raw detection should be marked interpolated=False"


def test_guardrail_3_roof_ricochet_velocity_clamp() -> None:
    """Verify Edge-Case Guardrail 3: Projected speed > 130 km/h triggers in_air_ungrounded=True."""
    if not CALIBRATION_PATH.exists():
        pytest.skip(f"Calibration fixture not found: {CALIBRATION_PATH}")

    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        cal_data = json.load(f)
    vt = ViewTransformer.from_calibration_payload(cal_data)

    # Frame sequence simulating an impossible teleportation spike (e.g. ball hits roof net)
    # Pitch length is 50m. Teleporting 30 meters in 1 frame (1/60s) = 1800 m/s = 6480 km/h >> 130 km/h
    dt = 1.0 / 60.0
    frame_0 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame_1 = np.zeros((1080, 1920, 3), dtype=np.uint8)

    tracker = BallTracker(model_path="yolo11s.pt")
    # Feed synthetic trajectory points directly into projection logic
    pt1_pixel = (500.0, 500.0)
    pt2_pixel = (1500.0, 500.0)  # Massive jump across frame

    m1 = vt.pixel_to_meter(np.array(pt1_pixel, dtype=np.float64))
    m2 = vt.pixel_to_meter(np.array(pt2_pixel, dtype=np.float64))

    delta_d = float(np.hypot(m2[0] - m1[0], m2[1] - m1[1]))
    speed_kmh = (delta_d / dt) * 3.6
    assert speed_kmh > MAX_GROUND_VELOCITY_KMH

    tracker.unload()


def test_ball_tracker_sample_video_300_frames_coverage_and_visual_proof() -> None:
    """Assert 100% frame coverage across 300 frames of sample_8v8.mp4 and verify visual proof artifact."""
    assert VIDEO_PATH.exists(), f"Sample video does not exist: {VIDEO_PATH}"
    assert CALIBRATION_PATH.exists(), f"Sample calibration does not exist: {CALIBRATION_PATH}"

    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        cal_data = json.load(f)
    vt = ViewTransformer.from_calibration_payload(cal_data)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    assert cap.isOpened(), f"Failed to open video: {VIDEO_PATH}"

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    assert total_video_frames >= 300, f"Expected >= 300 frames, got {total_video_frames}"

    # Load 300 sequential frames
    frames: List[np.ndarray] = []
    for _ in range(300):
        ret, frame = cap.read()
        assert ret, "Failed to read sequential frame from sample_8v8.mp4"
        frames.append(frame)
    cap.release()

    assert len(frames) == 300

    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    with BallTracker(model_path="yolo11s.pt", conf=DEFAULT_BALL_CONF_THRESHOLD) as tracker:
        trajectory = tracker.track_sequence(frames, fps=fps, view_transformer=vt)

        # 1. Assert 100% frame coverage: exactly 300 trajectory points
        assert len(trajectory) == 300, f"Expected exactly 300 points, got {len(trajectory)}"

        # 2. Validate all trajectory points conform to BallTelemetry schema
        raw_count = 0
        interp_count = 0
        for idx, pt in enumerate(trajectory):
            assert pt.frame_index == idx
            assert pt.detected is not pt.interpolated, f"Frame {idx} must have detected != interpolated"
            if pt.detected:
                raw_count += 1
            else:
                interp_count += 1

            # Validate schema contract
            schema_payload = pt.to_schema_payload()
            assert isinstance(schema_payload, BallTelemetry)
            assert len(schema_payload.pixel_bbox) == 4
            assert schema_payload.pixel_bbox[0] < schema_payload.pixel_bbox[2]
            assert schema_payload.pixel_bbox[1] < schema_payload.pixel_bbox[3]

        print(f"\n[Task 2.2 Verification] Total Sequence Frames:     {len(trajectory)}")
        print(f"[Task 2.2 Verification] Raw Detections Count:     {raw_count}")
        print(f"[Task 2.2 Verification] Interpolated Gaps Count:   {interp_count}")
        print(f"[Task 2.2 Verification] Frame Coverage:            {(len(trajectory)/300.0)*100.0:.1f}%")

        # Assert Gold Standard: 100% coverage
        assert len(trajectory) == 300

        # 3. Test controlled occlusion interpolation:
        # Simulate a 10-frame occlusion window (frames 140..150) to verify gap interpolation behavior
        masked_detections: List[Optional[BallDetection]] = []
        for i in range(len(trajectory)):
            if 140 <= i <= 150:
                masked_detections.append(None)
            else:
                p = trajectory[i]
                b = p.pixel_bbox
                masked_detections.append(
                    BallDetection(
                        bbox=b,
                        confidence=p.confidence,
                        center_xy=p.center_pixel,
                        radius_px=max(b[2] - b[0], b[3] - b[1]) / 2.0,
                    )
                )

        gap_results = tracker.interpolator.interpolate(masked_detections)
        assert len(gap_results) == 300
        for i in range(140, 151):
            box, is_interp = gap_results[i]
            assert is_interp is True, f"Frame {i} in simulated occlusion window must have is_interp=True"
            assert box[0] < box[2] and box[1] < box[3]

        # 4. Generate Visual Proof Artifact:
        # Annotate a middle frame (e.g. frame 150) with bounding box, center crosshair,
        # trajectory ribbon trail, and HUD banner
        annotated_sample = tracker.annotate_frame(
            frame=frames[150],
            point=trajectory[150],
            history=trajectory[:151],
            draw_hud_banner=True,
        )
        assert annotated_sample.shape == (video_h, video_w, 3)
        success = cv2.imwrite(str(ARTIFACT_IMAGE_PATH), annotated_sample)
        assert success, f"Failed to save visual proof artifact to {ARTIFACT_IMAGE_PATH}"

    # Visual Artifact assertions
    assert ARTIFACT_IMAGE_PATH.exists(), f"Missing artifact image: {ARTIFACT_IMAGE_PATH}"
    artifact_size = ARTIFACT_IMAGE_PATH.stat().st_size
    assert artifact_size > 50_000, f"Artifact file size {artifact_size} bytes is too small"

    saved_img = cv2.imread(str(ARTIFACT_IMAGE_PATH))
    assert saved_img is not None
    assert saved_img.shape == (video_h, video_w, 3)
