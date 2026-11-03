"""Unit tests and boundary validation for data contract schemas (Pydantic v2)."""

from typing import Any, Dict
import pytest
from pydantic import ValidationError

from schemas import (
    PitchCalibrationPayload,
    CameraIntrinsics,
    PitchCorrespondence,
    FrameTelemetryRecord,
    BallTelemetry,
    PlayerTelemetry,
    RoleEnum,
    Goal3DReconstructionClip,
    Goal3DFrame,
    Goal3DSkeleton,
)


def make_valid_calibration_payload() -> Dict[str, Any]:
    return {
        "match_id": "match_test_001",
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


def make_valid_frame_telemetry() -> Dict[str, Any]:
    return {
        "frame_index": 120,
        "timestamp_sec": 4.0,
        "ball": {
            "detected": True,
            "interpolated": False,
            "pixel_bbox": [1820.5, 940.0, 1842.0, 961.5],
            "meter_x": 24.82,
            "meter_y": 15.91,
            "meter_z": 0.0,
            "possessor_track_id": 7,
        },
        "players": [
            {
                "track_id": 7,
                "team_id": 0,
                "role": "PLAYER",
                "confidence": 0.91,
                "pixel_bbox": [1790.0, 810.0, 1855.0, 965.0],
                "meter_x": 24.50,
                "meter_y": 16.10,
                "speed_kmh": 14.2,
                "is_sprinting": False,
            }
        ],
    }


def make_valid_goal_clip() -> Dict[str, Any]:
    joints_17 = [[0.0, 0.0, float(i) * 0.1] for i in range(17)]
    return {
        "clip_id": "goal_01",
        "fps": 30,
        "start_timestamp_sec": 12.0,
        "end_timestamp_sec": 20.0,
        "frames": [
            {
                "frame_offset": 0,
                "ball_xyz": [38.4, 15.2, 0.15],
                "skeletons": [
                    {
                        "track_id": 7,
                        "team_id": 0,
                        "root_xyz": [38.1, 15.5, 0.0],
                        "joints_3d": joints_17,
                    }
                ],
            }
        ],
    }


class TestPitchCalibrationPayload:
    def test_valid_payload(self) -> None:
        payload = make_valid_calibration_payload()
        model = PitchCalibrationPayload.model_validate(payload)
        assert model.match_id == "match_test_001"
        assert len(model.correspondences) == 4
        assert model.pitch_length_m == 50.0

    def test_rejects_fewer_than_four_correspondences(self) -> None:
        payload = make_valid_calibration_payload()
        payload["correspondences"] = payload["correspondences"][:3]
        with pytest.raises(ValidationError):
            PitchCalibrationPayload.model_validate(payload)

    def test_rejects_negative_intrinsics(self) -> None:
        payload = make_valid_calibration_payload()
        payload["camera_intrinsics"]["fx"] = -100.0
        with pytest.raises(ValidationError):
            PitchCalibrationPayload.model_validate(payload)

    def test_rejects_out_of_bounds_correspondence_meter(self) -> None:
        payload = make_valid_calibration_payload()
        payload["correspondences"][0]["meter_x"] = 999.0
        with pytest.raises(ValidationError):
            PitchCalibrationPayload.model_validate(payload)


class TestFrameTelemetryRecord:
    def test_valid_record(self) -> None:
        payload = make_valid_frame_telemetry()
        model = FrameTelemetryRecord.model_validate(payload)
        assert model.frame_index == 120
        assert model.ball.detected is True
        assert len(model.players) == 1
        assert model.players[0].role == RoleEnum.PLAYER

    def test_rejects_out_of_bounds_meter_x_999(self) -> None:
        payload = make_valid_frame_telemetry()
        payload["players"][0]["meter_x"] = 999.0
        with pytest.raises(ValidationError):
            FrameTelemetryRecord.model_validate(payload)

    def test_rejects_out_of_bounds_ball_meter_x_999(self) -> None:
        payload = make_valid_frame_telemetry()
        payload["ball"]["meter_x"] = 999.0
        with pytest.raises(ValidationError):
            FrameTelemetryRecord.model_validate(payload)

    def test_rejects_excessive_speed_85_kmh(self) -> None:
        payload = make_valid_frame_telemetry()
        payload["players"][0]["speed_kmh"] = 85.0
        with pytest.raises(ValidationError):
            FrameTelemetryRecord.model_validate(payload)

    def test_rejects_negative_speed(self) -> None:
        payload = make_valid_frame_telemetry()
        payload["players"][0]["speed_kmh"] = -5.0
        with pytest.raises(ValidationError):
            FrameTelemetryRecord.model_validate(payload)

    def test_rejects_inverted_bounding_box(self) -> None:
        payload = make_valid_frame_telemetry()
        payload["players"][0]["pixel_bbox"] = [1855.0, 965.0, 1790.0, 810.0]
        with pytest.raises(ValidationError):
            FrameTelemetryRecord.model_validate(payload)

    def test_rejects_invalid_confidence(self) -> None:
        payload = make_valid_frame_telemetry()
        payload["players"][0]["confidence"] = 1.45
        with pytest.raises(ValidationError):
            FrameTelemetryRecord.model_validate(payload)


class TestGoal3DReconstructionClip:
    def test_valid_clip(self) -> None:
        payload = make_valid_goal_clip()
        model = Goal3DReconstructionClip.model_validate(payload)
        assert model.clip_id == "goal_01"
        assert model.fps == 30
        assert len(model.frames[0].skeletons[0].joints_3d) == 17

    def test_rejects_skeleton_not_17_joints(self) -> None:
        payload = make_valid_goal_clip()
        payload["frames"][0]["skeletons"][0]["joints_3d"] = [
            [0.0, 0.0, 0.0] for _ in range(16)
        ]
        with pytest.raises(ValidationError):
            Goal3DReconstructionClip.model_validate(payload)

    def test_rejects_joint_with_wrong_dimensions(self) -> None:
        payload = make_valid_goal_clip()
        payload["frames"][0]["skeletons"][0]["joints_3d"][0] = [0.0, 0.0]
        with pytest.raises(ValidationError):
            Goal3DReconstructionClip.model_validate(payload)

    def test_rejects_end_before_start_timestamp(self) -> None:
        payload = make_valid_goal_clip()
        payload["start_timestamp_sec"] = 25.0
        payload["end_timestamp_sec"] = 20.0
        with pytest.raises(ValidationError):
            Goal3DReconstructionClip.model_validate(payload)

    def test_rejects_out_of_bounds_ball_z(self) -> None:
        payload = make_valid_goal_clip()
        payload["frames"][0]["ball_xyz"] = [38.4, 15.2, 999.0]
        with pytest.raises(ValidationError):
            Goal3DReconstructionClip.model_validate(payload)
