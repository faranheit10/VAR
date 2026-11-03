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

## 4. `MatchSummaryReport` (Python Worker -> Frontend Dashboard & Share Cards)
```json
{
  "match_id": "uuid-v4",
  "duration_analysed_sec": 3600.0,
  "ball_visibility_pct": 94.5,
  "motm_track_id": 7,
  "team_stats": {
    "team_0": {
      "possession_pct": 54.2,
      "goals": 3,
      "total_xg": 2.41,
      "shots_total": 11,
      "shots_on_target": 7,
      "passes_completed": 118,
      "passes_progressive": 24,
      "balls_won": 31,
      "balls_lost": 28,
      "avg_convex_hull_m2": 310.5,
      "avg_defensive_line_m": 18.4,
      "territory_thirds_pct": [28.0, 41.0, 31.0]
    }
  },
  "player_stats": [
    {
      "track_id": 7,
      "team_id": 0,
      "role": "PLAYER",
      "time_on_pitch_sec": 3500.0,
      "match_rating": 9.2,
      "goals": 2,
      "assists": 1,
      "xg": 1.15,
      "shots": 4,
      "max_shot_speed_kmh": 68.4,
      "passes_completed": 29,
      "progressive_passes": 6,
      "line_breaking_passes": 3,
      "balls_won": 5,
      "balls_lost": 3,
      "total_distance_km": 4.82,
      "meters_per_min": 82.6,
      "top_speed_kmh": 26.1,
      "sprint_count": 14,
      "accelerations_high": 22,
      "decelerations_high": 19,
      "speed_zones_pct": {
        "walking": 42.0,
        "jogging": 38.5,
        "hard_running": 14.5,
        "sprinting": 5.0
      }
    }
  ],
  "events": [
    {
      "event_id": "evt_01",
      "type": "GOAL",
      "timestamp_sec": 1380.0,
      "team_id": 0,
      "scorer_track_id": 7,
      "assist_track_id": 4,
      "shot_speed_kmh": 64.2,
      "distance_to_goal_m": 11.4,
      "xg": 0.42,
      "clip_url": "artifacts/highlights/evt_01.mp4"
    }
  ]
}
```
