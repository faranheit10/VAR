# PHASED_TASKS.md — ATOMIC EXECUTION PLAN

Execute ONE `[ ]` task at a time per `AGENT_RULES.md`.

---

## PHASE 0: REPOSITORY BOOTSTRAP & TEST FIXTURES
**Goal:** Establish strict typing, monorepo tooling, and deterministic sample data before any CV logic is written.

- [ ] **Task 0.1: Monorepo & Python GPU Environment Setup**
  - **Action:** Initialize `apps/web` (Next.js + TS strict), `apps/api` (NestJS + TS strict), and `workers/cv-pipeline` (`pyproject.toml` with `torch`, `opencv-python-headless`, `ultralytics`, `supervision`, `transformers`, `umap-learn`, `scikit-learn`, `scipy`, `pydantic`, `pytest`, `mypy`).
  - **Deliverable:** `workers/cv-pipeline/scripts/check_env.py` that verifies CUDA availability and imports all libraries.
  - **Verification Command:** `python workers/cv-pipeline/scripts/check_env.py`
  - **Gold Standard Pass Criteria:** Script prints `CUDA_AVAILABLE: True` (or explicit CPU fallback flag) and exits with code `0`.

- [ ] **Task 0.2: Shared Data Contracts & Schema Validators**
  - **Action:** Implement all 3 schemas from `DATA_CONTRACTS.md` in `workers/cv-pipeline/src/schemas.py` (Pydantic v2) and `apps/web/src/lib/schemas.ts` (Zod).
  - **Deliverable:** Unit tests that validate valid payloads and reject out-of-bounds coordinates (e.g., `meter_x: 999` or `speed_kmh: 85.0`).
  - **Verification Command:** `pytest workers/cv-pipeline/tests/test_schemas.py && pnpm --filter web test:schemas`
  - **Gold Standard Pass Criteria:** 100% pass on boundary validation tests.

- [ ] **Task 0.3: Deterministic Test Video & Ground-Truth Fixture Generator**
  - **Action:** Write `workers/cv-pipeline/scripts/fetch_sample_match.py` to download a sample wide-angle football video clip (15 seconds, 1080p/4K) into `artifacts/sample_data/sample_8v8.mp4` and generate a reference `sample_calibration.json` with 4 known pitch corner pixel coordinates.
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_0_3.py`
  - **Gold Standard Pass Criteria:** Asserts `sample_8v8.mp4` exists, `fps >= 25`, `width >= 1920`, and frame count $\ge 300$.

---

## PHASE 1: CAMERA UNDISTORTION & HOMOGRAPHY ENGINE
**Goal:** Convert curved ultra-wide phone pixels into accurate $(X, Y)$ turf meters.

- [ ] **Task 1.1: Fisheye / Ultra-Wide Lens Undistortion Module**
  - **Action:** Create `workers/cv-pipeline/src/ingest/undistort.py`. Implement `undistort_frame(frame, intrinsics)` using `cv2.initUndistortRectifyMap` and `cv2.remap` (caching the map in memory so 4K frames undistort in $< 8\text{ ms}$).
  - **Deliverable:** Save before/after comparison image to `artifacts/verification/task_1_1/undistorted_check.jpg`.
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_1_1.py`
  - **Gold Standard Pass Criteria:** Collinearity RMSE of points sampled along the near touchline is $< 2.5\text{ pixels}$, and per-frame execution time after warmup is $< 15\text{ ms}$.

- [ ] **Task 1.2: 4-Point & Multi-Keypoint Homography Transformer (`ViewTransformer`)**
  - **Action:** Create `workers/cv-pipeline/src/calibration/homography.py`. Implement `ViewTransformer` with `pixel_to_meter(points_uv)` and `meter_to_pixel(points_xy)` using `cv2.findHomography(..., method=cv2.RANSAC)`.
  - **Deliverable:** Render a top-down $50\text{m} \times 32\text{m}$ green pitch image (`artifacts/verification/task_1_2/pitch_grid.jpg`) and project a $5\text{m} \times 5\text{m}$ meter grid back onto the camera frame (`artifacts/verification/task_1_2/camera_grid_overlay.jpg`).
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_1_2.py`
  - **Gold Standard Pass Criteria:** Round-trip reprojection RMSE $\|p - H^{-1}(H(p))\|_2 < 0.15\text{ meters}$ across all 4 corners and penalty spots.

---

## PHASE 2: DETECTION, SMALL-BALL SLICING & GROUND-PLANE TRACKING
**Goal:** Detect all 16 players and the ball, maintaining persistent IDs across frames.

- [ ] **Task 2.1: High-Resolution Player & Goalkeeper Detector**
  - **Action:** Create `workers/cv-pipeline/src/detection/player_detector.py` wrapping Ultralytics `YOLO11` (at `imgsz=1280`, `conf=0.25`, `iou=0.45`). Extract the bottom-center foot coordinate $[u_c, v_{max}]$ for each bounding box.
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_2_1.py`
  - **Gold Standard Pass Criteria:** Across 10 sampled frames of `sample_8v8.mp4`, average player count per frame is $\ge 14$, and `artifacts/verification/task_2_1/detections_frame_0.jpg` is written with foot anchor points drawn.

- [ ] **Task 2.2: SAHI Slicing Ball Detector + Kalman Gap Interpolator**
  - **Action:** Create `workers/cv-pipeline/src/detection/ball_tracker.py`. Implement sliced inference (2x2 overlapping tiles) for small ball detection, plus a 2D Kalman filter + cubic spline interpolator for gaps up to 15 frames where the ball is occluded behind a player's leg.
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_2_2.py`
  - **Gold Standard Pass Criteria:** Raw + interpolated ball trajectory has $100\%$ frame coverage across a 300-frame sequence, and flag `interpolated: true` is accurately set on occluded frames.

- [ ] **Task 2.3: Multi-Object Tracker with Ground-Plane Meter Projection**
  - **Action:** Create `workers/cv-pipeline/src/tracking/tracker.py` using `supervision.ByteTrack` (`track_activation_threshold=0.25`, `lost_track_buffer=60`, `minimum_matching_threshold=0.8`). Project every tracked player's foot anchor through `ViewTransformer.pixel_to_meter`.
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_2_3.py`
  - **Gold Standard Pass Criteria:** All projected $(X, Y)$ player coordinates lie within $[-2.0\text{m}, 52.0\text{m}] \times [-2.0\text{m}, 34.0\text{m}]$ (pitch + 2m touchline margin), and unique track IDs across a 5-second clip do not exceed $22$ (proving minimal ID fragmentation).

---

## PHASE 3: UNSUPERVISED TEAM CLUSTERING (SIGLIP + UMAP + K-MEANS)
**Goal:** Automatically separate Team A and Team B without manual tagging.

- [ ] **Task 3.1: Upper-Torso Cropping & SigLIP Embedding Extractor**
  - **Action:** Create `workers/cv-pipeline/src/clustering/team_classifier.py`. For every player bounding box, crop the top $15\%\text{–}55\%$ vertical region (torso/kit only, excluding head and turf between legs). Extract embeddings using `google/siglip-base-patch16-224` in batches of 32.
  - **Deliverable:** Save a montage of torso crops grouped by predicted cluster (`artifacts/verification/task_3_1/team_clusters.jpg`).
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_3_1.py`
  - **Gold Standard Pass Criteria:** UMAP (3 components) + K-Means (`n_clusters=2`) yields a Silhouette Score $> 0.35$, and temporal majority voting locks each `track_id` to a single `team_id` with $0$ single-frame team flips.

---

## PHASE 4: KINEMATICS, POSSESSION, PASSING & TACTICAL RADAR
**Goal:** Compute physically accurate match analytics that pass the anti-jitter gold standards.

- [ ] **Task 4.1: Savitzky-Golay Trajectory Smoother & Velocity/Distance Engine**
  - **Action:** Create `workers/cv-pipeline/src/kinematics/metrics.py`. Apply `scipy.signal.savgol_filter(window_length=15, polyorder=2)` to $(X_t, Y_t)$ per track. Compute total distance (meters), instantaneous speed (km/h), and sprint events (speed $> 20.0\text{ km/h}$ sustained for $\ge 1.0\text{s}$). Hard-clamp outliers caused by tracker jumps ($> 34.0\text{ km/h}$).
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_4_1.py`
  - **Gold Standard Pass Criteria:** Synthetic stationary test track with $\pm 3\text{ px}$ Gaussian noise records $< 0.3\text{m}$ total distance over $3.0\text{s}$ (vs $> 2.5\text{m}$ unsmoothed), and max speed across `sample_8v8.mp4` is $\le 34.0\text{ km/h}$.

- [ ] **Task 4.2: Ball Possession, Pass Graph & Voronoi Pitch Control**
  - **Action:** Create `workers/cv-pipeline/src/kinematics/tactics.py`. Assign ball possession to the player within $1.5\text{m}$ of the ball for $\ge 3$ consecutive frames. Detect passes (possession shift `Team A -> Team A`) and interceptions (`Team A -> Team B`). Compute Voronoi tessellation polygons for pitch control percentage.
  - **Deliverable:** Render a split-screen video (`artifacts/verification/task_4_2/tactical_radar_preview.mp4`) showing camera view on top and 2D Voronoi radar on bottom.
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_4_2.py`
  - **Gold Standard Pass Criteria:** Team A Possession % + Team B Possession % = $100.0\% \pm 0.01\%$, Voronoi area sum equals $1600.0\text{ m}^2$ ($50 \times 32$), and output MP4 is readable with $\ge 300$ frames.

---

## PHASE 5: 3D GOAL RECONSTRUCTION ENGINE
**Goal:** Lift a 10-second goal clip from 2D video into 3D world-grounded skeletons and a 3D ball trajectory.

- [ ] **Task 5.1: 3D Ball Projectile Trajectory Solver**
  - **Action:** Create `workers/cv-pipeline/src/reconstruction/ball_3d.py`. Given shot start frame $t_0$ (foot contact, $Z=0$) and end frame $t_1$ (goal line crossing), compute $Z(t)$ using projectile physics + vertical pixel offset relative to the $2.0\text{m}$ goalpost height.
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_5_1.py`
  - **Gold Standard Pass Criteria:** $Z(t_0) = 0.0\text{m}$, $Z(t) \ge 0.0\text{m}$ for all $t \in [t_0, t_1]$, and peak height for a standard shot is within $[0.0\text{m}, 4.5\text{m}]$.

- [ ] **Task 5.2: Monocular 3D Human Pose Lifting & Ground-Plane Clamping**
  - **Action:** Create `workers/cv-pipeline/src/reconstruction/pose_3d.py` using `RTMPose3D` (via `mmpose` or ONNX runtime). For each player in the 10-second goal window, extract 17 3D keypoints, translate the root pelvis to $(X_{meter}, Y_{meter})$, and translate vertically so $\min(Z_{left\_ankle}, Z_{right\_ankle}) = 0.0\text{m}$. Export `Goal3DReconstructionClip` JSON.
  - **Verification Command:** `pytest workers/cv-pipeline/tests/verify_task_5_2.py`
  - **Gold Standard Pass Criteria:** Validates exported JSON against `Goal3DReconstructionClip` schema; asserts 17 joints per skeleton and ankle ground contact error $\le 0.05\text{m}$.

---

## PHASE 6: WEB DASHBOARD, 4-CLICK CALIBRATOR & R3F 3D REPLAY VIEWER
**Goal:** Build the interactive Next.js + React Three Fiber frontend and NestJS BullMQ orchestration API.

- [ ] **Task 6.1: NestJS BullMQ Job Pipeline & Supabase Storage Wiring**
  - **Action:** Implement `POST /matches/upload`, `POST /matches/:id/calibrate`, and `POST /matches/:id/merge-tracks` (for 1-tap human-in-the-loop ID swap fixing) in `apps/api`.
  - **Verification Command:** `pnpm --filter api test:e2e`
  - **Gold Standard Pass Criteria:** E2E test submits calibration payload, triggers worker, merges two track IDs (`track_id: 19 -> track_id: 7`), and verifies recalculated distance/speed stats.

- [ ] **Task 6.2: Interactive 4-Click Pitch Calibrator & ID-Swap Timeline UI**
  - **Action:** Build `<PitchCalibrator />` in `apps/web` (HTML5 Canvas magnifying loupe allowing sub-pixel clicking of the 4 turf corners on Frame 1 with live homography grid preview) and `<TrackMergerTimeline />` (click two player avatars to merge swapped IDs).
  - **Verification Command:** `pnpm --filter web typecheck && pnpm --filter web test`
  - **Gold Standard Pass Criteria:** Zero TypeScript errors; unit test verifies clicking 4 corners generates a valid `PitchCalibrationPayload`.

- [ ] **Task 6.3: React Three Fiber (`R3F`) 3D Goal Replay & 2D Radar Player**
  - **Action:** Build `<Goal3DCanvas clip={goalClip} />` in `apps/web/src/components/replay/` using `@react-three/fiber` and `@react-three/drei`. Render the $50\text{m} \times 32\text{m}$ turf pitch, $5\text{m} \times 2\text{m}$ 3D goals, animated 17-joint skeletal mannequins (colored by `team_id`), 3D ball with trail ribbon, and camera presets (`Orbit`, `Striker POV`, `Goalkeeper POV`).
  - **Verification Command:** `pnpm --filter web build && pnpm --filter web test:r3f`
  - **Gold Standard Pass Criteria:** Production Next.js build succeeds with 0 warnings, and headless scene graph test confirms 3D pitch mesh, ball mesh, and skeletal bone segments mount at exact coordinates from `Goal3DReconstructionClip`.
