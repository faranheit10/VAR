"""Deterministic verification script for Task 1.2: 4-Point & Multi-Keypoint Homography Transformer (ViewTransformer).

Gold Standard Acceptance Criteria:
1. Round-trip reprojection RMSE ||p - H^-1(H(p))||_2 < 0.15 meters across all 4 corners and penalty spots.
2. Direct correspondence reprojection RMSE < 0.15 meters.
3. Deliverable 1 created: artifacts/verification/task_1_2/pitch_grid.jpg (top-down 50m x 32m green pitch).
4. Deliverable 2 created: artifacts/verification/task_1_2/camera_grid_overlay.jpg (5m x 5m grid projected on camera frame).
5. Touchline margin filtering verifies EDGE_CASES_AND_GUARDRAILS.md Rule 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from calibration.homography import (
    DEFAULT_PITCH_LENGTH_M,
    DEFAULT_PITCH_WIDTH_M,
    ViewTransformer,
    compute_reprojection_rmse,
    overlay_pitch_grid_on_camera,
    render_top_down_pitch,
)
from ingest.undistort import undistort_frame
from schemas import PitchCalibrationPayload


SAMPLE_DATA_DIR = Path("artifacts/sample_data")
SAMPLE_VIDEO_PATH = SAMPLE_DATA_DIR / "sample_8v8.mp4"
SAMPLE_CALIBRATION_PATH = SAMPLE_DATA_DIR / "sample_calibration.json"
VERIFICATION_DIR = Path("artifacts/verification/task_1_2")
PITCH_GRID_ARTIFACT_PATH = VERIFICATION_DIR / "pitch_grid.jpg"
CAMERA_OVERLAY_ARTIFACT_PATH = VERIFICATION_DIR / "camera_grid_overlay.jpg"


@pytest.fixture(scope="module")
def sample_calibration() -> PitchCalibrationPayload:
    """Load and validate sample calibration payload fixture."""
    assert SAMPLE_CALIBRATION_PATH.exists(), f"Missing calibration fixture at {SAMPLE_CALIBRATION_PATH}"
    with open(SAMPLE_CALIBRATION_PATH, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return PitchCalibrationPayload.model_validate(data)


def test_homography_reprojection_rmse_and_deliverables(
    sample_calibration: PitchCalibrationPayload,
) -> None:
    """Verify round-trip reprojection RMSE < 0.15m and generate visual artifacts."""
    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    transformer = ViewTransformer.from_calibration_payload(sample_calibration)

    # 1. Direct correspondence reprojection test
    corresp = sample_calibration.correspondences
    src_pixels = np.array([[c.pixel_u, c.pixel_v] for c in corresp], dtype=np.float64)
    expected_meters = np.array([[c.meter_x, c.meter_y] for c in corresp], dtype=np.float64)

    projected_meters = transformer.pixel_to_meter(src_pixels)
    direct_rmse = compute_reprojection_rmse(expected_meters, projected_meters)

    print(f"\n[Gold Standard Check: Homography Direct Reprojection]")
    print(f"  - Direct Correspondences Reprojection RMSE: {direct_rmse:.6f} meters (Threshold: < 0.15m)")
    assert direct_rmse < 0.15, f"Direct reprojection RMSE {direct_rmse:.4f}m exceeds 0.15m threshold"

    # 2. Keypoints Round-Trip Reprojection Test: 4 corners + penalty spots + center mark + goal posts
    L = sample_calibration.pitch_length_m
    W = sample_calibration.pitch_width_m
    keypoints_m = np.array(
        [
            [0.0, 0.0],          # Top-left corner
            [L, 0.0],            # Top-right corner
            [L, W],              # Bottom-right corner
            [0.0, W],            # Bottom-left corner
            [7.0, W / 2.0],      # Left penalty spot
            [L - 7.0, W / 2.0],  # Right penalty spot
            [L / 2.0, W / 2.0],  # Center mark
            [0.0, 13.5],         # Left goal post 1
            [0.0, 18.5],         # Left goal post 2
            [L, 13.5],           # Right goal post 1
            [L, 18.5],           # Right goal post 2
        ],
        dtype=np.float64,
    )

    # Dense 5m x 5m grid test points
    grid_x, grid_y = np.meshgrid(
        np.arange(0.0, L + 0.1, 5.0),
        np.arange(0.0, W + 0.1, 5.0),
    )
    dense_grid_m = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    all_eval_points_m = np.vstack([keypoints_m, dense_grid_m])

    # Round trip: meters -> pixel -> meter
    pixels = transformer.meter_to_pixel(all_eval_points_m)
    recovered_meters = transformer.pixel_to_meter(pixels)

    round_trip_rmse = compute_reprojection_rmse(all_eval_points_m, recovered_meters)
    l2_errors = np.linalg.norm(all_eval_points_m - recovered_meters, axis=1)
    max_round_trip_error = float(np.max(l2_errors))

    print(f"[Gold Standard Check: Round-Trip Reprojection]")
    print(f"  - Keypoints & 5m Grid Evaluated: {len(all_eval_points_m)} points")
    print(f"  - Round-Trip Reprojection RMSE:  {round_trip_rmse:.6f} meters (Threshold: < 0.15m)")
    print(f"  - Max Round-Trip Error:          {max_round_trip_error:.6f} meters")

    # Assert exact Gold Standard Threshold from GOLD_STANDARDS.md
    assert (
        round_trip_rmse < 0.15
    ), f"Round-trip reprojection RMSE {round_trip_rmse:.6f}m exceeds 0.15m threshold"
    assert (
        max_round_trip_error < 0.15
    ), f"Max reprojection error {max_round_trip_error:.6f}m exceeds 0.15m threshold"

    # 3. Deliverable 1: Render top-down 50m x 32m green pitch
    pitch_img = render_top_down_pitch(
        pitch_length_m=L,
        pitch_width_m=W,
        px_per_meter=20,
        grid_step_m=5.0,
        output_path=PITCH_GRID_ARTIFACT_PATH,
    )
    assert PITCH_GRID_ARTIFACT_PATH.exists(), f"Pitch grid artifact missing at {PITCH_GRID_ARTIFACT_PATH}"
    assert PITCH_GRID_ARTIFACT_PATH.stat().st_size > 1000, "Pitch grid file unexpectedly small"
    reloaded_pitch = cv2.imread(str(PITCH_GRID_ARTIFACT_PATH))
    assert reloaded_pitch is not None, "Failed to load generated pitch image"
    assert reloaded_pitch.shape == pitch_img.shape
    print(f"[Deliverable 1 Check] Saved: {PITCH_GRID_ARTIFACT_PATH} ({pitch_img.shape[1]}x{pitch_img.shape[0]})")

    # 4. Deliverable 2: Project 5m x 5m meter grid onto camera frame
    assert SAMPLE_VIDEO_PATH.exists(), f"Video file missing at {SAMPLE_VIDEO_PATH}"
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO_PATH))
    assert cap.isOpened(), "Could not open video file"
    ret, raw_frame = cap.read()
    cap.release()
    assert ret, "Failed to read first frame from sample video"

    # Undistort frame first
    undistorted_frame = undistort_frame(raw_frame, sample_calibration.camera_intrinsics)

    # Project grid and save overlay artifact
    overlay_img = overlay_pitch_grid_on_camera(
        frame=undistorted_frame,
        view_transformer=transformer,
        grid_step_m=5.0,
        rmse_meters=round_trip_rmse,
        output_path=CAMERA_OVERLAY_ARTIFACT_PATH,
    )
    assert CAMERA_OVERLAY_ARTIFACT_PATH.exists(), f"Overlay artifact missing at {CAMERA_OVERLAY_ARTIFACT_PATH}"
    assert CAMERA_OVERLAY_ARTIFACT_PATH.stat().st_size > 1000, "Overlay file unexpectedly small"
    reloaded_overlay = cv2.imread(str(CAMERA_OVERLAY_ARTIFACT_PATH))
    assert reloaded_overlay is not None, "Failed to load generated overlay image"
    assert reloaded_overlay.shape == overlay_img.shape
    print(f"[Deliverable 2 Check] Saved: {CAMERA_OVERLAY_ARTIFACT_PATH} ({overlay_img.shape[1]}x{overlay_img.shape[0]})")


def test_4k_data_contract_reprojection() -> None:
    """Verify round-trip RMSE < 0.15m on 4K schema payload from DATA_CONTRACTS.md."""
    payload_4k: Dict[str, Any] = {
        "match_id": "4k-reference-match",
        "pitch_length_m": 50.0,
        "pitch_width_m": 32.0,
        "camera_intrinsics": {
            "k1": -0.28,
            "k2": 0.07,
            "p1": 0.0,
            "p2": 0.0,
            "fx": 1850.0,
            "fy": 1850.0,
            "cx": 1920.0,
            "cy": 1080.0,
        },
        "correspondences": [
            {"label": "top_left_corner", "pixel_u": 412.5, "pixel_v": 380.2, "meter_x": 0.0, "meter_y": 0.0},
            {"label": "top_right_corner", "pixel_u": 3410.0, "pixel_v": 375.8, "meter_x": 50.0, "meter_y": 0.0},
            {"label": "bottom_right_corner", "pixel_u": 3810.0, "pixel_v": 1980.0, "meter_x": 50.0, "meter_y": 32.0},
            {"label": "bottom_left_corner", "pixel_u": 95.0, "pixel_v": 1995.0, "meter_x": 0.0, "meter_y": 32.0},
        ],
    }

    transformer_4k = ViewTransformer.from_calibration_payload(payload_4k)

    # Test keypoints
    pts_m = np.array(
        [
            [0.0, 0.0],
            [50.0, 0.0],
            [50.0, 32.0],
            [0.0, 32.0],
            [7.0, 16.0],
            [43.0, 16.0],
            [25.0, 16.0],
        ],
        dtype=np.float64,
    )

    rt_rmse = transformer_4k.compute_roundtrip_rmse(pts_m)
    print(f"\n[4K Reference Contract Check]")
    print(f"  - 4K Round-Trip Reprojection RMSE: {rt_rmse:.6f} meters (Threshold: < 0.15m)")
    assert rt_rmse < 0.15, f"4K round-trip RMSE {rt_rmse:.6f}m exceeds 0.15m"


def test_touchline_spectator_filtering(sample_calibration: PitchCalibrationPayload) -> None:
    """Verify touchline spectator filtering conforms to EDGE_CASES_AND_GUARDRAILS.md Rule 1."""
    transformer = ViewTransformer.from_calibration_payload(sample_calibration)

    test_points_m = np.array(
        [
            [25.0, 16.0],  # Dead center -> True
            [0.0, 0.0],    # Corner boundary -> True
            [50.0, 32.0],  # Corner boundary -> True
            [-0.4, 16.0],  # 0.4m outside left goal line (within 0.5m margin) -> True
            [50.4, 16.0],  # 0.4m outside right goal line (within 0.5m margin) -> True
            [-0.6, 16.0],  # 0.6m outside (spectator) -> False
            [50.6, 16.0],  # 0.6m outside (spectator) -> False
            [25.0, 32.6],  # 0.6m outside touchline (substitute bench) -> False
            [25.0, -0.6],  # 0.6m outside far touchline -> False
        ],
        dtype=np.float64,
    )

    mask = transformer.is_inside_pitch(test_points_m, margin_m=0.5)
    expected_mask = np.array([True, True, True, True, True, False, False, False, False], dtype=bool)

    np.testing.assert_array_equal(mask, expected_mask)
    print("\n[Guardrail Check] Touchline boundary & spectator filter passed")


def test_invalid_correspondences_handling() -> None:
    """Verify appropriate exceptions on invalid or fewer than 4 correspondences."""
    with pytest.raises(ValueError, match="At least 4 correspondences"):
        ViewTransformer(
            source_pixels=np.array([[10, 10], [20, 20], [30, 30]]),
            target_meters=np.array([[0, 0], [1, 1], [2, 2]]),
        )
