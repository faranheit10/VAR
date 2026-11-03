# ARCHITECTURE_AND_DESIGN.md — 8v8 TURF AI ANALYTICS PLATFORM ("TURFVISION")

## 1. System Overview & Tech Stack
A monocular (single 4K 60fps ultra-wide smartphone) 8v8 football match analytics and 3D goal reconstruction platform modeled after Replayfield.

- **Monorepo Layout:**
  - `apps/web`: Next.js (App Router), TypeScript, Tailwind CSS, React Three Fiber (`@react-three/fiber`, `@react-three/drei`) for the 2D tactical radar, 4-point calibration UI, ID-swap merger, 3D interactive goal replay viewer, and shareable social stat cards (`html-to-image`).
  - `apps/api`: NestJS (TypeScript), BullMQ + Redis (job orchestration), Supabase (PostgreSQL + Storage for match metadata, player profiles, and telemetry JSON blobs).
  - `workers/cv-pipeline`: Python 3.11+, PyTorch (CUDA), OpenCV, Ultralytics (`YOLO11`), `supervision` (`ByteTrack`), `transformers` (`google/siglip-base-patch16-224`), `umap-learn`, `scikit-learn`, `scipy`, and `MMPose` (`RTMPose3D`).

## 2. Directory Structure (Strict Enforcement)
```text
root/
├── AGENT_RULES.md
├── ARCHITECTURE_AND_DESIGN.md
├── DATA_CONTRACTS.md
├── GOLD_STANDARDS.md
├── MODEL_AND_HARDWARE_MANIFEST.md
├── EDGE_CASES_AND_GUARDRAILS.md
├── PHASED_TASKS.md
├── PROGRESS_LOG.md
├── artifacts/
│   ├── sample_data/             # Raw 8v8 test clip (15s, 4K 60fps)
│   ├── verification/            # Output images/JSON from verification scripts
│   ├── highlights/              # Sliced 12s goal MP4 clips & concatenated reels
│   └── cache/                   # Raw intermediate parquet detections
├── apps/
│   ├── api/                     # NestJS + BullMQ backend
│   └── web/                     # Next.js + React Three Fiber frontend
└── workers/
    └── cv-pipeline/
        ├── pyproject.toml
        ├── src/
        │   ├── ingest/          # Frame extraction, lens undistortion & highlight_cutter
        │   ├── calibration/     # 4-point & keypoint homography (pixels <-> meters)
        │   ├── detection/       # YOLO11 player/referee + SAHI ball detection
        │   ├── tracking/        # ByteTrack + ground-plane Kalman smoothing
        │   ├── clustering/      # SigLIP + UMAP + K-Means + track_stitcher (auto-heal IDs)
        │   ├── kinematics/      # Savitzky-Golay, Voronoi, advanced_stats (xG, speed zones)
        │   └── reconstruction/  # 3D ball parabola + RTMPose3D skeletal lifting
        └── tests/               # Deterministic verification scripts per task
```

## 3. Core Mathematical Specifications (Do Not Deviate)

### Pitch Coordinate System:
Standard 8v8 Turf Pitch Dimensions: Length $L = 50.0\text{ m}$ ($X \in [0, 50]$), Width $W = 32.0\text{ m}$ ($Y \in [0, 32]$), Goal Height $H_g = 2.0\text{ m}$, Goal Width $W_g = 5.0\text{ m}$ ($Y \in [13.5, 18.5]$).
Origin $(0, 0)$ is the top-left corner of the pitch in the 2D top-down radar.

### Ground-Plane Projection (Homography):
Given pixel coordinates of a player's feet $p = [u_{bottom\_center}, v_{bottom\_center}, 1]^T$ and the $3 \times 3$ homography matrix $H$:

$$p' = H \cdot p, \quad X_{meters} = \frac{p'_x}{p'_z}, \quad Y_{meters} = \frac{p'_y}{p'_z}$$

### Kinematics Smoothing (Anti-Jitter Law):
Raw bounding box bottom-center coordinates jitter by $\pm 3\text{ px}$ per frame, which artificially adds $1.5\text{ km}$ of fake distance over a match.
You MUST apply a Savitzky-Golay filter (`scipy.signal.savgol_filter` with `window_length=15`, `polyorder=2` at 30fps) to each track's $(X_t, Y_t)$ meter trajectory BEFORE computing Euclidean distance $\Delta d = \sqrt{\Delta X^2 + \Delta Y^2}$ or instantaneous velocity $v_t = \Delta d / \Delta t$.

### 4-Tier Physical Speed Zones:
- **Zone 1 (Walking/Standing):** $0\text{–}7\text{ km/h}$
- **Zone 2 (Jogging/Cruising):** $7\text{–}14\text{ km/h}$
- **Zone 3 (High-Intensity Running / "Ran Hard"):** $14\text{–}20\text{ km/h}$
- **Zone 4 (Sprinting):** $> 20\text{ km/h}$ sustained $\ge 1.0\text{s}$

### Geometric Tracking Expected Goals (xG) & Visible Goal Angle:
For a shot originating at coordinates $(X_s, Y_s)$ aimed at the goal mouth at $X = 50.0$ bounded by goalposts at $(50, 13.5)$ and $(50, 18.5)$ with distances $d_1 = \|(X_s, Y_s) - (50, 13.5)\|_2$ and $d_2 = \|(X_s, Y_s) - (50, 18.5)\|_2$:

$$\theta = \arccos\left(\frac{d_1^2 + d_2^2 - W_{goal}^2}{2 d_1 d_2}\right)$$

Shot distance $d = \|(X_s, Y_s) - (50, 16.0)\|_2$.
The geometric xG incorporates distance $d$, visible angle $\theta$, and defender pressure count $N_{\text{blockers}}$ (opponents within the triangle formed by $(X_s, Y_s)$ and the goalposts):

$$\text{xG} = \sigma\left(\beta_0 + \beta_1 d + \beta_2 \theta + \beta_3 N_{\text{blockers}}\right)$$

### 3D Ball Trajectory Physics:
For a shot starting at frame $t_0$ at ground point $(X_0, Y_0, 0)$ and ending at frame $t_1$ at $(X_1, Y_1, Z_1)$:

$$Z(t) = v_{z0}(t - t_0) - \frac{1}{2}g(t - t_0)^2 \quad \text{where } g = 9.81\text{ m/s}^2$$
