# DATA_CONTRACTS.md — STRICT INTER-SERVICE SCHEMAS

Both Python (`pydantic` v2) and TypeScript (`zod`) MUST implement these exact schemas. Any extra or missing field fails verification.

## 1. `PitchCalibrationPayload` (Frontend -> Python Worker)
```json
{
  "match_id": "uuid-v4",
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
    "cy": 1080.0
  },
  "correspondences": [
    { "label": "top_left_corner", "pixel_u": 412.5, "pixel_v": 380.2, "meter_x": 0.0, "meter_y": 0.0 },
    { "label": "top_right_corner", "pixel_u": 3410.0, "pixel_v": 375.8, "meter_x": 50.0, "meter_y": 0.0 },
    { "label": "bottom_right_corner", "pixel_u": 3810.0, "pixel_v": 1980.0, "meter_x": 50.0, "meter_y": 32.0 },
    { "label": "bottom_left_corner", "pixel_u": 95.0, "pixel_v": 1995.0, "meter_x": 0.0, "meter_y": 32.0 }
  ]
}
```

## 2. `FrameTelemetryRecord` (Python Worker -> Frontend Radar & Stats)
```json
{
  "frame_index": 120,
  "timestamp_sec": 4.000,
  "ball": {
    "detected": true,
    "interpolated": false,
    "pixel_bbox": [1820.5, 940.0, 1842.0, 961.5],
    "meter_x": 24.82,
    "meter_y": 15.91,
    "meter_z": 0.00,
    "possessor_track_id": 7
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
      "is_sprinting": false
    }
  ]
}
```

## 3. `Goal3DReconstructionClip` (Python Worker -> React Three Fiber)
```json
{
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
          "joints_3d": [
            [0.0, 0.0, 0.92],
            [0.12, 0.0, 0.48],
            [0.14, 0.05, 0.02]
          ]
        }
      ]
    }
  ]
}
```

> **Note:** `joints_3d` MUST contain exactly 17 COCO-format 3D keypoints relative to `root_xyz`, with the lowest ankle joint clamped such that $\min(Z_{ankle}) = 0.0\text{ m}$ during ground contact.
