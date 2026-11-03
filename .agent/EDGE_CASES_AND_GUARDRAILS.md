# EDGE_CASES_AND_GUARDRAILS.md — 8v8 TURF PHYSICS & FILTERING LAWS

Your code in Phases 2, 3, and 4 MUST explicitly implement these 5 filters:

## 1. The "Subs & Spectators on the Touchline" Filter
- In 8v8 turf cages, substitutes, waiting players, or spectators stand right against the boundary net. YOLO will detect them as players.
- **Rule:** After projecting every detected person's foot coordinate $(u, v)$ through `ViewTransformer.pixel_to_meter(u, v)`, immediately discard any detection where:
  $$X < -0.5\text{ m} \quad \text{or} \quad X > 50.5\text{ m} \quad \text{or} \quad Y < -0.5\text{ m} \quad \text{or} \quad Y > 32.5\text{ m}$$
- Only players physically inside the $50\text{m} \times 32\text{m}$ pitch (plus a $0.5\text{m}$ line margin) may enter `ByteTrack`.

## 2. Goalkeeper Isolation in Team Clustering
- Goalkeepers often wear a different colored jersey or bib from both outfield teams, or swap roles mid-game in 8v8.
- **Rule:** Before running K-Means (`n_clusters=2`) on SigLIP torso embeddings, classify any track whose median $X$ position is within $4.5\text{m}$ of either goal line ($X \in [0, 4.5]$ or $X \in [45.5, 50.0]$) as `role = "GOALKEEPER"`. Assign their `team_id` based on which goal they defend ($X < 25 \implies \text{Team 0}$, $X \ge 25 \implies \text{Team 1}$), rather than kit color clustering.

## 3. Roof-Net & Cage-Wall Ball Ricochets
- Enclosed 8v8 pitches have overhead nets. When the ball hits the roof net, a naive 2D homography projection maps that high pixel coordinate as if the ball teleported 40 meters behind the far touchline.
- **Rule:** If the ball's projected 2D ground-plane velocity between two consecutive frames exceeds $130\text{ km/h}$ ($36.1\text{ m/s}$), flag the ball state as `in_air_ungrounded = True` and clamp its 2D radar coordinate to a linear interpolation between the takeoff point and landing point rather than projecting its aerial pixel onto the grass.

## 4. Video Chunking for Long Matches
- Never process a 60-minute match as a single monolithic job.
- **Rule:** The ingestion worker (`workers/cv-pipeline/src/ingest/chunker.py`) must split any video longer than 60 seconds into **60-second segments** using `ffmpeg -c copy -f segment -segment_time 60` so BullMQ can track progress per minute and retry failed segments without restarting from minute 0.

## 5. Coordinate Handedness (OpenCV vs. React Three Fiber)
- OpenCV / 2D Radar uses $(X, Y)$ where $X \in [0, 50]$ (left-to-right), $Y \in [0, 32]$ (top-to-bottom), and $Z \ge 0$ (upward height).
- Three.js / React Three Fiber uses a **Y-up** coordinate system by default.
- **Rule:** In `<Goal3DCanvas />` (Task 6.3), you MUST map telemetry coordinates as:
  $$\text{Three.js } (x, y, z) = \left(X_{meter} - 25.0,\; Z_{meter},\; Y_{meter} - 16.0\right)$$
  This centers the $50\text{m} \times 32\text{m}$ pitch at the 3D world origin $(0, 0, 0)$ with height along the Three.js $Y$-axis.
