"""Deterministic verification script for Task 0.3: Deterministic Test Video & Ground-Truth Fixture Generator.

Gold Standard Acceptance Criteria:
- sample_8v8.mp4 exists in artifacts/sample_data/
- Video FPS >= 25.0
- Video width >= 1920
- Total frame count >= 300
- sample_calibration.json exists and strictly validates against PitchCalibrationPayload
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import pytest

from schemas import PitchCalibrationPayload


SAMPLE_DATA_DIR = Path("artifacts/sample_data")
SAMPLE_VIDEO_PATH = SAMPLE_DATA_DIR / "sample_8v8.mp4"
CALIBRATION_JSON_PATH = SAMPLE_DATA_DIR / "sample_calibration.json"


def test_sample_video_exists_and_meets_gold_standards() -> None:
    """Assert sample_8v8.mp4 exists with required resolution, FPS, and frame count."""
    assert SAMPLE_VIDEO_PATH.exists(), f"Video not found at {SAMPLE_VIDEO_PATH}"
    assert SAMPLE_VIDEO_PATH.stat().st_size > 0, "Video file is empty"

    cap = cv2.VideoCapture(str(SAMPLE_VIDEO_PATH))
    assert cap.isOpened(), f"Failed to open video file at {SAMPLE_VIDEO_PATH}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\n[Gold Standard Check] Video FPS: {fps} (required: >= 25.0)")
    print(f"[Gold Standard Check] Video Width: {width} (required: >= 1920)")
    print(f"[Gold Standard Check] Video Height: {height}")
    print(f"[Gold Standard Check] Frame Count: {frame_count} (required: >= 300)")

    # Assert exact Gold Standard Thresholds from PHASED_TASKS.md Task 0.3
    assert fps >= 25.0, f"Expected FPS >= 25.0, got {fps}"
    assert width >= 1920, f"Expected width >= 1920, got {width}"
    assert frame_count >= 300, f"Expected frame count >= 300, got {frame_count}"


def test_calibration_fixture_validates_against_schema() -> None:
    """Assert sample_calibration.json exists, has 4 corners, and conforms to PitchCalibrationPayload."""
    assert CALIBRATION_JSON_PATH.exists(), f"Calibration JSON not found at {CALIBRATION_JSON_PATH}"

    with open(CALIBRATION_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate against strict Pydantic contract
    payload = PitchCalibrationPayload.model_validate(data)
    assert len(payload.correspondences) >= 4, "Must have at least 4 pitch correspondences"
    assert payload.pitch_length_m == 50.0
    assert payload.pitch_width_m == 32.0

    print("\n[Schema Check] sample_calibration.json validated successfully against PitchCalibrationPayload")
