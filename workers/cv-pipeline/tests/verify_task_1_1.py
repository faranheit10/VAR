"""Deterministic verification script for Task 1.1: Fisheye / Ultra-Wide Lens Undistortion Module.

Gold Standard Acceptance Criteria:
1. Collinearity RMSE of points sampled along the near touchline is < 2.5 pixels from a straight line fit (y = mx + b).
2. If undistortion is skipped or omitted, the touchline bows significantly (>= 5.0 px on 1080p, >= 30.0 px on 4K).
3. Per-frame execution time after warmup is strictly < 15.0 ms.
4. Visual proof artifact saved to artifacts/verification/task_1_1/undistorted_check.jpg.
5. Invertibility / round-trip mapping error is < 0.5 pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from ingest.undistort import (
    CameraIntrinsics,
    LensUndistorter,
    compute_collinearity_rmse,
    create_comparison_montage,
    distort_points,
    undistort_frame,
    undistort_points,
)
from schemas import PitchCalibrationPayload


SAMPLE_DATA_DIR = Path("artifacts/sample_data")
SAMPLE_VIDEO_PATH = SAMPLE_DATA_DIR / "sample_8v8.mp4"
SAMPLE_CALIBRATION_PATH = SAMPLE_DATA_DIR / "sample_calibration.json"
VERIFICATION_DIR = Path("artifacts/verification/task_1_1")
ARTIFACT_IMAGE_PATH = VERIFICATION_DIR / "undistorted_check.jpg"


@pytest.fixture(scope="module")
def sample_calibration() -> PitchCalibrationPayload:
    """Load and validate sample calibration payload fixture."""
    assert SAMPLE_CALIBRATION_PATH.exists(), f"Missing calibration fixture at {SAMPLE_CALIBRATION_PATH}"
    with open(SAMPLE_CALIBRATION_PATH, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return PitchCalibrationPayload.model_validate(data)


def test_collinearity_rmse_and_visual_artifact(sample_calibration: PitchCalibrationPayload) -> None:
    """Verify collinearity RMSE < 2.5 px on undistorted touchline and output debug visual artifact."""
    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    intrinsics = sample_calibration.camera_intrinsics

    # 1. Test collinearity on near touchline (1080p sample calibration)
    # The near touchline connects bottom-left (100, 980) to bottom-right (1820, 980)
    u_ideal = np.linspace(100.0, 1820.0, 60)
    v_ideal = np.full_like(u_ideal, 980.0)
    pts_ideal = np.stack([u_ideal, v_ideal], axis=1)

    # Distort the ideal straight touchline using the camera's barrel distortion coefficients
    pts_distorted = distort_points(pts_ideal, intrinsics)
    rmse_distorted = compute_collinearity_rmse(pts_distorted)

    # Undistort the curved touchline
    pts_undistorted = undistort_points(pts_distorted, intrinsics)
    rmse_undistorted = compute_collinearity_rmse(pts_undistorted)

    print(f"\n[Gold Standard Check 1080p]")
    print(f"  - Distorted Touchline Bow RMSE:    {rmse_distorted:.2f} px (demonstrates lens curvature)")
    print(f"  - Undistorted Touchline Line RMSE: {rmse_undistorted:.4f} px (Gold Standard Threshold: < 2.5 px)")

    # Assert Gold Standard criteria
    assert rmse_distorted > 5.0, f"Expected noticeable barrel distortion, got RMSE {rmse_distorted}"
    assert rmse_undistorted < 2.5, f"Collinearity RMSE {rmse_undistorted} exceeds 2.5 px threshold"

    # 2. Test 4K Reference Intrinsics from DATA_CONTRACTS.md (where bowing exceeds 30+ px)
    intrinsics_4k = CameraIntrinsics(
        fx=1850.0,
        fy=1850.0,
        cx=1920.0,
        cy=1080.0,
        k1=-0.28,
        k2=0.07,
        p1=0.0,
        p2=0.0,
    )
    u_4k_ideal = np.linspace(95.0, 3810.0, 100)
    v_4k_ideal = np.linspace(1995.0, 1980.0, 100)
    pts_4k_ideal = np.stack([u_4k_ideal, v_4k_ideal], axis=1)

    pts_4k_distorted = distort_points(pts_4k_ideal, intrinsics_4k)
    rmse_4k_dist = compute_collinearity_rmse(pts_4k_distorted)
    pts_4k_undist = undistort_points(pts_4k_distorted, intrinsics_4k)
    rmse_4k_undist = compute_collinearity_rmse(pts_4k_undist)

    print(f"[Gold Standard Check 4K Reference]")
    print(f"  - 4K Distorted Touchline Bow RMSE:    {rmse_4k_dist:.2f} px (Gold Standard Law: bows 30+ px)")
    print(f"  - 4K Undistorted Touchline Line RMSE: {rmse_4k_undist:.4f} px (Gold Standard Threshold: < 2.5 px)")

    # Assert 4K bowing exceeds 30 px (per GOLD_STANDARDS.md table: "touchline bows by 30+ pixels")
    assert rmse_4k_dist >= 30.0, f"Expected 4K distorted touchline bow >= 30.0 px, got {rmse_4k_dist:.2f}"
    assert rmse_4k_undist < 2.5, f"4K Undistorted RMSE {rmse_4k_undist:.4f} exceeds 2.5 px threshold"

    # 3. Read video frame, undistort, and save visual proof artifact
    assert SAMPLE_VIDEO_PATH.exists(), f"Video file missing at {SAMPLE_VIDEO_PATH}"
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO_PATH))
    assert cap.isOpened(), "Could not open video file"
    ret, raw_frame = cap.read()
    cap.release()
    assert ret, "Failed to read first frame from sample video"

    # Undistort the real frame
    undist_frame = undistort_frame(raw_frame, intrinsics)
    assert undist_frame.shape == raw_frame.shape
    assert undist_frame.dtype == raw_frame.dtype

    # Create and save comparison montage
    montage = create_comparison_montage(
        original_frame=raw_frame,
        undistorted_frame=undist_frame,
        points_distorted=pts_distorted,
        points_undistorted=pts_undistorted,
        rmse_distorted=rmse_distorted,
        rmse_undistorted=rmse_undistorted,
        output_path=ARTIFACT_IMAGE_PATH,
    )

    assert ARTIFACT_IMAGE_PATH.exists(), f"Artifact not created at {ARTIFACT_IMAGE_PATH}"
    assert ARTIFACT_IMAGE_PATH.stat().st_size > 1000, "Artifact image file is unexpectedly small"
    saved_img = cv2.imread(str(ARTIFACT_IMAGE_PATH))
    assert saved_img is not None, "Failed to reload saved verification image"
    assert saved_img.shape == montage.shape
    print(f"  - Visual Proof Artifact Saved:     {ARTIFACT_IMAGE_PATH} ({saved_img.shape[1]}x{saved_img.shape[0]})")


def test_per_frame_execution_time_after_warmup(sample_calibration: PitchCalibrationPayload) -> None:
    """Verify per-frame undistortion execution time after warmup is strictly < 15.0 ms."""
    intrinsics = sample_calibration.camera_intrinsics
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Warmup phase (to trigger rectification map calculation and memory caching)
    for _ in range(5):
        _ = undistort_frame(frame, intrinsics)

    # Benchmark loop (30 iterations)
    num_iterations = 30
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = undistort_frame(frame, intrinsics)
    elapsed_total_sec = time.perf_counter() - start_time

    avg_ms_per_frame = (elapsed_total_sec / num_iterations) * 1000.0
    print(f"\n[Gold Standard Latency Check]")
    print(f"  - Per-frame execution time: {avg_ms_per_frame:.2f} ms (Gold Standard Threshold: < 15.0 ms)")

    assert (
        avg_ms_per_frame < 15.0
    ), f"Per-frame execution time {avg_ms_per_frame:.2f} ms exceeds threshold of 15.0 ms"


def test_lens_undistorter_map_caching(sample_calibration: PitchCalibrationPayload) -> None:
    """Verify rectification maps are cached and not reallocated on identical dimensions."""
    intrinsics = sample_calibration.camera_intrinsics
    undistorter = LensUndistorter(intrinsics)

    map1_a, map2_a = undistorter.get_rectification_maps(1920, 1080)
    map1_b, map2_b = undistorter.get_rectification_maps(1920, 1080)

    # Pointer identity confirms caching
    assert map1_a is map1_b, "Rectification map1 was not cached in memory"
    assert map2_a is map2_b, "Rectification map2 was not cached in memory"


def test_roundtrip_coordinate_invertibility(sample_calibration: PitchCalibrationPayload) -> None:
    """Verify forward distortion and inverse undistortion round-trip has sub-pixel error < 0.5 px."""
    intrinsics = sample_calibration.camera_intrinsics

    u_orig = np.linspace(200.0, 1720.0, 25)
    v_orig = np.linspace(200.0, 950.0, 25)
    pts_orig = np.stack([u_orig, v_orig], axis=1)

    pts_dist = distort_points(pts_orig, intrinsics)
    pts_recovered = undistort_points(pts_dist, intrinsics)

    l2_errors = np.linalg.norm(pts_orig - pts_recovered, axis=1)
    mean_error = float(np.mean(l2_errors))
    max_error = float(np.max(l2_errors))

    print(f"\n[Invertibility Check]")
    print(f"  - Mean round-trip error: {mean_error:.4f} px (threshold: < 0.5 px)")
    print(f"  - Max round-trip error:  {max_error:.4f} px (threshold: < 0.5 px)")

    assert mean_error < 0.5, f"Mean round-trip error {mean_error:.4f} px exceeds 0.5 px"
    assert max_error < 0.5, f"Max round-trip error {max_error:.4f} px exceeds 0.5 px"
