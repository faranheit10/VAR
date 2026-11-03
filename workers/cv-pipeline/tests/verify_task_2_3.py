"""Deterministic verification test for Task 2.3: Multi-Object Tracker with Ground-Plane Meter Projection.

Verifies:
1. ByteTrack hyperparameter fidelity:
   track_activation_threshold=0.25, lost_track_buffer=60, minimum_matching_threshold=0.8.
2. Ground-plane Kalman smoother reduces spatial coordinate jitter.
3. Ground-plane spatial re-identifier maintains identity continuity across temporary occlusions.
4. Gold Standard Pass Criteria on 5-second clip of sample_8v8.mp4:
   - All projected (X, Y) player coordinates lie within [-2.0m, 52.0m] x [-2.0m, 34.0m].
   - Unique track IDs across the 5-second clip do not exceed 22 (proving minimal ID fragmentation).
5. All tracked players conform to the PlayerTelemetry schema contract.
6. Generates visual proof artifact artifacts/verification/task_2_3/tracking_overlay.jpg with
   bounding boxes, track IDs, foot anchor markers, turf meter coordinates, and HUD banner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set

import cv2
import numpy as np
import pytest
import supervision as sv

from calibration.homography import ViewTransformer
from schemas import PlayerTelemetry, RoleEnum
from tracking.tracker import (
    DEFAULT_LOST_TRACK_BUFFER,
    DEFAULT_MINIMUM_MATCHING_THRESHOLD,
    DEFAULT_TRACK_ACTIVATION_THRESHOLD,
    PITCH_X_MAX_ALLOWED,
    PITCH_X_MIN_ALLOWED,
    PITCH_Y_MAX_ALLOWED,
    PITCH_Y_MIN_ALLOWED,
    ByteTrack,
    GroundPlaneKalmanSmoother,
    GroundPlaneReidentifier,
    PlayerTracker,
    TrackedPlayer,
)


VIDEO_PATH = Path("artifacts/sample_data/sample_8v8.mp4")
CALIBRATION_PATH = Path("artifacts/sample_data/sample_calibration.json")
VERIFICATION_DIR = Path("artifacts/verification/task_2_3")
ARTIFACT_IMAGE_PATH = VERIFICATION_DIR / "tracking_overlay.jpg"

MAX_ALLOWED_UNIQUE_TRACK_IDS: int = 22


def test_bytetrack_configuration_and_api() -> None:
    """Verify ByteTrack hyperparameters strictly adhere to Task 2.3 specifications."""
    bt = ByteTrack(
        track_activation_threshold=DEFAULT_TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=DEFAULT_LOST_TRACK_BUFFER,
        minimum_matching_threshold=DEFAULT_MINIMUM_MATCHING_THRESHOLD,
        frame_rate=60,
    )

    assert np.isclose(bt.track_activation_threshold, 0.25)
    assert bt.lost_track_buffer == 60
    assert np.isclose(bt.minimum_matching_threshold, 0.80)
    assert bt.frame_rate == 60

    # Empty detections test
    empty_res = bt.update_with_detections(sv.Detections.empty())
    assert isinstance(empty_res, sv.Detections)
    assert len(empty_res) == 0


def test_ground_plane_kalman_smoother_jitter_reduction() -> None:
    """Verify Ground-Plane Kalman filter smooths coordinate jitter on stationary/moving targets."""
    smoother = GroundPlaneKalmanSmoother(dt=1.0 / 60.0)

    # Simulate a stationary goalkeeper at (5.0m, 16.0m) with random jitter
    np.random.seed(42)
    true_x, true_y = 5.0, 16.0
    jitter_sigma = 0.25  # 25cm bounding box jitter

    raw_x_vals: List[float] = []
    smoothed_x_vals: List[float] = []

    for _ in range(60):
        noisy_x = true_x + float(np.random.normal(0.0, jitter_sigma))
        noisy_y = true_y + float(np.random.normal(0.0, jitter_sigma))
        raw_x_vals.append(noisy_x)

        sm_x, sm_y, speed_kmh = smoother.update(track_id=1, meter_x=noisy_x, meter_y=noisy_y)
        smoothed_x_vals.append(sm_x)

    raw_var = float(np.var(raw_x_vals[10:]))
    smoothed_var = float(np.var(smoothed_x_vals[10:]))

    # Assert Kalman filter significantly reduces spatial variance
    assert smoothed_var < raw_var, f"Smoothed variance {smoothed_var:.4f} not less than raw {raw_var:.4f}"
    assert np.isclose(np.mean(smoothed_x_vals), true_x, atol=0.10)


def test_ground_plane_reidentifier_continuity() -> None:
    """Verify GroundPlaneReidentifier links tracks across temporary occlusion gaps within 2.5m."""
    reid = GroundPlaneReidentifier(max_reid_distance_m=2.5, max_lost_frames=60)

    # Frame 0: Track 1 at (20.0, 15.0)
    id_0 = reid.get_canonical_id(raw_id=1, meter_x=20.0, meter_y=15.0, frame_idx=0)
    assert id_0 == 1

    # Frame 25: Player was occluded for 25 frames, reappears at (20.8, 15.3) with new detector raw_id=19
    id_reappear = reid.get_canonical_id(raw_id=19, meter_x=20.8, meter_y=15.3, frame_idx=25)
    # Must be linked back to canonical track 1
    assert id_reappear == 1, f"Expected canonical ID 1, got {id_reappear}"

    # Another player far away at (40.0, 25.0) should get a distinct ID
    id_other = reid.get_canonical_id(raw_id=20, meter_x=40.0, meter_y=25.0, frame_idx=25)
    assert id_other == 20


def test_player_tracker_sample_clip_bounds_and_id_fragmentation() -> None:
    """Assert all player coords are within bounds and unique track IDs <= 22 across a 5-second clip."""
    assert VIDEO_PATH.exists(), f"Missing test video: {VIDEO_PATH}"
    assert CALIBRATION_PATH.exists(), f"Missing calibration fixture: {CALIBRATION_PATH}"

    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        cal_data = json.load(f)
    vt = ViewTransformer.from_calibration_payload(cal_data)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    assert cap.isOpened(), "Failed to open sample_8v8.mp4"

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    # 5-second clip test: 150 frames (5.0s at 30fps)
    num_frames_to_test = 150
    frames: List[np.ndarray] = []
    for _ in range(num_frames_to_test):
        ret, frame = cap.read()
        assert ret, "Failed to read frame from test video"
        frames.append(frame)
    cap.release()

    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    unique_track_ids: Set[int] = set()
    all_meter_x: List[float] = []
    all_meter_y: List[float] = []
    first_frame_tracks: List[TrackedPlayer] = []

    with PlayerTracker(
        view_transformer=vt,
        model_path="yolo11s.pt",
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.80,
        frame_rate=int(round(fps)),
        pitch_margin_m=0.5,
    ) as tracker:
        for idx, frame in enumerate(frames):
            tracks = tracker.update(frame)

            if idx == 0:
                first_frame_tracks = tracks

            for p in tracks:
                unique_track_ids.add(p.track_id)
                all_meter_x.append(p.meter_x)
                all_meter_y.append(p.meter_y)

                # Schema contract validation on each tracked player
                telemetry = p.to_player_telemetry()
                assert isinstance(telemetry, PlayerTelemetry)
                assert telemetry.role == RoleEnum.PLAYER
                assert 0.0 <= telemetry.speed_kmh <= 36.0

        # Generate Visual Proof Artifact for frame 0
        annotated_frame = tracker.annotate_frame(
            frame=frames[0],
            tracks=first_frame_tracks,
            draw_hud_banner=True,
            draw_foot_anchors=True,
        )
        assert annotated_frame.shape == (video_h, video_w, 3)
        saved = cv2.imwrite(str(ARTIFACT_IMAGE_PATH), annotated_frame)
        assert saved, f"Failed to save visual proof artifact to {ARTIFACT_IMAGE_PATH}"

    # 1. Assert Gold Standard Coordinate Bounds: [-2.0m, 52.0m] x [-2.0m, 34.0m]
    min_x, max_x = min(all_meter_x), max(all_meter_x)
    min_y, max_y = min(all_meter_y), max(all_meter_y)

    print(f"\n[Task 2.3 Verification] Tracked Frames:         {num_frames_to_test}")
    print(f"[Task 2.3 Verification] Unique Track IDs:       {len(unique_track_ids)} (Max Allowed: {MAX_ALLOWED_UNIQUE_TRACK_IDS})")
    print(f"[Task 2.3 Verification] Active Track IDs List:  {sorted(list(unique_track_ids))}")
    print(f"[Task 2.3 Verification] Meter X Range:          [{min_x:.2f}m, {max_x:.2f}m] (Allowed: [{PITCH_X_MIN_ALLOWED}m, {PITCH_X_MAX_ALLOWED}m])")
    print(f"[Task 2.3 Verification] Meter Y Range:          [{min_y:.2f}m, {max_y:.2f}m] (Allowed: [{PITCH_Y_MIN_ALLOWED}m, {PITCH_Y_MAX_ALLOWED}m])")

    assert min_x >= PITCH_X_MIN_ALLOWED, f"X min {min_x:.2f}m < {PITCH_X_MIN_ALLOWED}m"
    assert max_x <= PITCH_X_MAX_ALLOWED, f"X max {max_x:.2f}m > {PITCH_X_MAX_ALLOWED}m"
    assert min_y >= PITCH_Y_MIN_ALLOWED, f"Y min {min_y:.2f}m < {PITCH_Y_MIN_ALLOWED}m"
    assert max_y <= PITCH_Y_MAX_ALLOWED, f"Y max {max_y:.2f}m > {PITCH_Y_MAX_ALLOWED}m"

    # 2. Assert Gold Standard Unique Track IDs <= 22 across 5-second clip
    assert len(unique_track_ids) <= MAX_ALLOWED_UNIQUE_TRACK_IDS, (
        f"ID Fragmentation failure: unique track IDs {len(unique_track_ids)} exceeds threshold {MAX_ALLOWED_UNIQUE_TRACK_IDS}"
    )

    # 3. Assert Visual Proof Artifact
    assert ARTIFACT_IMAGE_PATH.exists(), f"Missing artifact: {ARTIFACT_IMAGE_PATH}"
    artifact_size = ARTIFACT_IMAGE_PATH.stat().st_size
    assert artifact_size > 50_000, f"Artifact file too small ({artifact_size} bytes)"

    saved_img = cv2.imread(str(ARTIFACT_IMAGE_PATH))
    assert saved_img is not None
    assert saved_img.shape == (video_h, video_w, 3)
