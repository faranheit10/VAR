"""Data contract schemas for 8v8 Turf AI analytics platform.

Defines Pydantic v2 schemas for:
1. PitchCalibrationPayload (Frontend -> Python Worker)
2. FrameTelemetryRecord (Python Worker -> Frontend Radar & Stats)
3. Goal3DReconstructionClip (Python Worker -> React Three Fiber)
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Pitch boundaries in meters (50m x 32m turf with safety margin)
PITCH_X_MIN: float = -10.0
PITCH_X_MAX: float = 60.0
PITCH_Y_MIN: float = -10.0
PITCH_Y_MAX: float = 42.0
BALL_Z_MIN: float = 0.0
BALL_Z_MAX: float = 20.0
MAX_HUMAN_TURF_SPEED_KMH: float = 36.0


class RoleEnum(str, Enum):
    PLAYER = "PLAYER"
    GOALKEEPER = "GOALKEEPER"
    REFEREE = "REFEREE"


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k1: float
    k2: float
    p1: float = 0.0
    p2: float = 0.0
    fx: Annotated[float, Field(gt=0)]
    fy: Annotated[float, Field(gt=0)]
    cx: Annotated[float, Field(gt=0)]
    cy: Annotated[float, Field(gt=0)]


class PitchCorrespondence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    pixel_u: Annotated[float, Field(ge=0.0)]
    pixel_v: Annotated[float, Field(ge=0.0)]
    meter_x: Annotated[float, Field(ge=PITCH_X_MIN, le=PITCH_X_MAX)]
    meter_y: Annotated[float, Field(ge=PITCH_Y_MIN, le=PITCH_Y_MAX)]


class PitchCalibrationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match_id: str
    pitch_length_m: Annotated[float, Field(gt=0.0, le=120.0)] = 50.0
    pitch_width_m: Annotated[float, Field(gt=0.0, le=90.0)] = 32.0
    camera_intrinsics: CameraIntrinsics
    correspondences: Annotated[List[PitchCorrespondence], Field(min_length=4)]


class BallTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detected: bool
    interpolated: bool
    pixel_bbox: Annotated[List[float], Field(min_length=4, max_length=4)]
    meter_x: Annotated[float, Field(ge=PITCH_X_MIN, le=PITCH_X_MAX)]
    meter_y: Annotated[float, Field(ge=PITCH_Y_MIN, le=PITCH_Y_MAX)]
    meter_z: Annotated[float, Field(ge=BALL_Z_MIN, le=BALL_Z_MAX)] = 0.0
    possessor_track_id: Optional[int] = None

    @field_validator("pixel_bbox")
    @classmethod
    def validate_bbox_coordinates(cls, v: List[float]) -> List[float]:
        x1, y1, x2, y2 = v
        if x1 > x2 or y1 > y2:
            raise ValueError(f"Invalid bounding box: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
        return v


class PlayerTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: int
    team_id: int
    role: RoleEnum
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    pixel_bbox: Annotated[List[float], Field(min_length=4, max_length=4)]
    meter_x: Annotated[float, Field(ge=PITCH_X_MIN, le=PITCH_X_MAX)]
    meter_y: Annotated[float, Field(ge=PITCH_Y_MIN, le=PITCH_Y_MAX)]
    speed_kmh: Annotated[float, Field(ge=0.0, le=MAX_HUMAN_TURF_SPEED_KMH)]
    is_sprinting: bool

    @field_validator("pixel_bbox")
    @classmethod
    def validate_bbox_coordinates(cls, v: List[float]) -> List[float]:
        x1, y1, x2, y2 = v
        if x1 > x2 or y1 > y2:
            raise ValueError(f"Invalid bounding box: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]")
        return v


class FrameTelemetryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: Annotated[int, Field(ge=0)]
    timestamp_sec: Annotated[float, Field(ge=0.0)]
    ball: BallTelemetry
    players: List[PlayerTelemetry]


class Goal3DSkeleton(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: int
    team_id: int
    root_xyz: Annotated[List[float], Field(min_length=3, max_length=3)]
    joints_3d: Annotated[List[List[float]], Field(min_length=17, max_length=17)]

    @field_validator("joints_3d")
    @classmethod
    def validate_joint_dimensions(cls, joints: List[List[float]]) -> List[List[float]]:
        for idx, joint in enumerate(joints):
            if len(joint) != 3:
                raise ValueError(
                    f"Joint {idx} must have 3 coordinates [x, y, z], got {len(joint)}"
                )
        return joints

    @field_validator("root_xyz")
    @classmethod
    def validate_root_xyz(cls, root: List[float]) -> List[float]:
        x, y, z = root
        if not (PITCH_X_MIN <= x <= PITCH_X_MAX and PITCH_Y_MIN <= y <= PITCH_Y_MAX and -1.0 <= z <= 5.0):
            raise ValueError(f"Root XYZ coordinates out of physical bounds: {root}")
        return root


class Goal3DFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_offset: Annotated[int, Field(ge=0)]
    ball_xyz: Annotated[List[float], Field(min_length=3, max_length=3)]
    skeletons: List[Goal3DSkeleton]

    @field_validator("ball_xyz")
    @classmethod
    def validate_ball_xyz(cls, ball: List[float]) -> List[float]:
        x, y, z = ball
        if not (PITCH_X_MIN <= x <= PITCH_X_MAX and PITCH_Y_MIN <= y <= PITCH_Y_MAX and BALL_Z_MIN <= z <= BALL_Z_MAX):
            raise ValueError(f"Ball XYZ coordinates out of bounds: {ball}")
        return ball


class Goal3DReconstructionClip(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_id: str
    fps: Annotated[int, Field(gt=0)]
    start_timestamp_sec: Annotated[float, Field(ge=0.0)]
    end_timestamp_sec: Annotated[float, Field(ge=0.0)]
    frames: List[Goal3DFrame]

    @model_validator(mode="after")
    def validate_timestamp_range(self) -> Goal3DReconstructionClip:
        if self.end_timestamp_sec < self.start_timestamp_sec:
            raise ValueError(
                f"end_timestamp_sec ({self.end_timestamp_sec}) cannot be less than "
                f"start_timestamp_sec ({self.start_timestamp_sec})"
            )
        return self
