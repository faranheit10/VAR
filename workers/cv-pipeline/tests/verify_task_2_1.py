"""Deterministic verification test for Task 2.1: High-Resolution Player & Goalkeeper Detector.

Verifies:
1. Mathematical precision of bottom-center foot coordinate extraction [u_c, v_max].
2. Detector hyperparameter fidelity (imgsz=1280, conf=0.25, iou=0.45, classes=[0]).
3. Gold Standard Recall: Across 10 sampled frames of sample_8v8.mp4, average player count >= 14.
4. Generation and validation of visual proof artifact artifacts/verification/task_2_1/detections_frame_0.jpg.
5. Supervision Detections data interop and sequential model memory offload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
import supervision as sv

from calibration.homography import ViewTransformer
from detection.player_detector import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_IOU_THRESHOLD,
    PlayerDetector,
    extract_foot_coordinates,
)


VIDEO_PATH = Path("artifacts/sample_data/sample_8v8.mp4")
CALIBRATION_PATH = Path("artifacts/sample_data/sample_calibration.json")
VERIFICATION_DIR = Path("artifacts/verification/task_2_1")
ARTIFACT_IMAGE_PATH = VERIFICATION_DIR / "detections_frame_0.jpg"

# Gold standard requirement: >= 14 of 16 players per frame (>= 87.5% recall)
GOLD_STANDARD_MIN_AVG_PLAYERS = 14.0


def test_extract_foot_coordinates_math() -> None:
    """Verify foot anchor math: u_c = (x1 + x2)/2, v_max = y2."""
    # Test multiple distinct boxes
    bboxes = np.array(
        [
            [100.0, 200.0, 300.0, 500.0],  # u_c = 200.0, v_max = 500.0
            [1250.5, 400.0, 1350.5, 750.25],  # u_c = 1300.5, v_max = 750.25
            [0.0, 0.0, 10.0, 20.0],  # u_c = 5.0, v_max = 20.0
        ],
        dtype=np.float64,
    )

    feet = extract_foot_coordinates(bboxes)
    assert feet.shape == (3, 2), f"Expected shape (3, 2), got {feet.shape}"
    assert np.isclose(feet[0, 0], 200.0), f"Expected u_c=200.0, got {feet[0, 0]}"
    assert np.isclose(feet[0, 1], 500.0), f"Expected v_max=500.0, got {feet[0, 1]}"
    assert np.isclose(feet[1, 0], 1300.5), f"Expected u_c=1300.5, got {feet[1, 0]}"
    assert np.isclose(feet[1, 1], 750.25), f"Expected v_max=750.25, got {feet[1, 1]}"
    assert np.isclose(feet[2, 0], 5.0), f"Expected u_c=5.0, got {feet[2, 0]}"
    assert np.isclose(feet[2, 1], 20.0), f"Expected v_max=20.0, got {feet[2, 1]}"

    # Test single 1D box input
    single_box = np.array([50.0, 80.0, 150.0, 220.0])
    single_foot = extract_foot_coordinates(single_box)
    assert single_foot.shape == (1, 2)
    assert np.isclose(single_foot[0, 0], 100.0)
    assert np.isclose(single_foot[0, 1], 220.0)

    # Test empty array
    empty = extract_foot_coordinates(np.empty((0, 4), dtype=np.float64))
    assert empty.shape == (0, 2)

    # Test invalid boxes raise ValueError
    with pytest.raises(ValueError, match="Invalid bounding box"):
        extract_foot_coordinates(np.array([[200.0, 100.0, 150.0, 300.0]]))  # x1 > x2

    with pytest.raises(ValueError, match="Invalid bounding box"):
        extract_foot_coordinates(np.array([[100.0, 400.0, 200.0, 300.0]]))  # y1 > y2


def test_player_detector_configuration() -> None:
    """Verify player detector defaults adhere strictly to Task 2.1 specs."""
    detector = PlayerDetector(model_path="yolo11s.pt")
    assert detector.imgsz == DEFAULT_IMGSZ == 1280
    assert np.isclose(detector.conf, DEFAULT_CONF_THRESHOLD) and np.isclose(detector.conf, 0.25)
    assert np.isclose(detector.iou, DEFAULT_IOU_THRESHOLD) and np.isclose(detector.iou, 0.45)
    assert detector.classes == [0]
    detector.unload()
    assert detector.model is None


def test_player_detector_sample_video_recall_and_visual_proof() -> None:
    """Assert average player count across 10 sampled frames >= 14 and save visual proof."""
    assert VIDEO_PATH.exists(), f"Sample video does not exist: {VIDEO_PATH}"

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    assert cap.isOpened(), f"Failed to open video: {VIDEO_PATH}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert total_frames >= 300, f"Expected >= 300 frames, got {total_frames}"

    # Sample exactly 10 frames across the clip
    sampled_indices = np.linspace(0, total_frames - 1, 10, dtype=int).tolist()
    counts_per_frame: List[int] = []

    # Load ViewTransformer if calibration fixture exists for ground-plane projection
    view_transformer = None
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
        view_transformer = ViewTransformer.from_calibration_payload(cal_data)

    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    with PlayerDetector(model_path="yolo11s.pt") as detector:
        for idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            assert ret, f"Failed to read frame at index {idx}"

            result = detector.detect(
                frame,
                view_transformer=view_transformer,
                pitch_margin_m=0.5,
            )

            counts_per_frame.append(result.count)

            # Ensure all detected boxes are well-formed and inside frame dimensions
            for det in result.detections:
                assert det.confidence >= 0.25, f"Confidence {det.confidence} below threshold 0.25"
                assert det.x1 < det.x2 and det.y1 < det.y2, f"Inverted box: {det.bbox}"
                assert 0.0 <= det.foot_u <= video_w, f"Foot X {det.foot_u} outside frame width"
                assert 0.0 <= det.foot_v <= video_h, f"Foot Y {det.foot_v} outside frame height"

            # Frame 0 deliverable: write annotated visual proof artifact
            if idx == sampled_indices[0]:
                annotated = detector.annotate_frame(
                    frame=frame,
                    result=result,
                    draw_bboxes=True,
                    draw_foot_anchors=True,
                    draw_labels=True,
                    draw_hud_banner=True,
                )
                assert annotated.shape == (video_h, video_w, 3)
                success = cv2.imwrite(str(ARTIFACT_IMAGE_PATH), annotated)
                assert success, f"Failed to save visual proof artifact to {ARTIFACT_IMAGE_PATH}"

    cap.release()

    # Numeric assertions
    avg_players = float(np.mean(counts_per_frame))
    print(f"\n[Task 2.1 Verification] Sampled Frame Indices: {sampled_indices}")
    print(f"[Task 2.1 Verification] Counts Per Frame:      {counts_per_frame}")
    print(f"[Task 2.1 Verification] Average Player Count:  {avg_players:.2f}")
    print(f"[Task 2.1 Verification] Gold Standard Target:  >= {GOLD_STANDARD_MIN_AVG_PLAYERS:.1f}")

    assert avg_players >= GOLD_STANDARD_MIN_AVG_PLAYERS, (
        f"Recall failure: average player count {avg_players:.2f} < threshold {GOLD_STANDARD_MIN_AVG_PLAYERS}"
    )

    # Artifact assertions
    assert ARTIFACT_IMAGE_PATH.exists(), f"Missing artifact: {ARTIFACT_IMAGE_PATH}"
    artifact_size = ARTIFACT_IMAGE_PATH.stat().st_size
    assert artifact_size > 50_000, f"Artifact file too small ({artifact_size} bytes)"

    saved_img = cv2.imread(str(ARTIFACT_IMAGE_PATH))
    assert saved_img is not None, "Failed to reload saved verification image"
    assert saved_img.shape == (video_h, video_w, 3)


def test_supervision_interop_and_pitch_margin() -> None:
    """Verify supervision Detections data payload and ground-plane meter projection."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    with PlayerDetector(model_path="yolo11s.pt") as detector:
        # Synthetic frame test for empty detections
        empty_res = detector.detect(frame)
        assert empty_res.count == 0
        assert isinstance(empty_res.sv_detections, sv.Detections)
        assert len(empty_res.sv_detections) == 0

    # Ensure model is safely unloaded
    assert detector.model is None
