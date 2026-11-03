# GOLD_STANDARDS.md — DETERMINISTIC ACCEPTANCE CRITERIA

Every verification script (`tests/verify_task_X_Y.py`) must assert these exact mathematical thresholds. If a metric falls outside these bounds, the task FAILS.

| Domain | Metric | Gold Standard Threshold | Why Skimming Fails Here |
| :--- | :--- | :--- | :--- |
| **Lens Undistortion** | Collinearity Error of Pitch Touchline | RMSE $< 2.5\text{ pixels}$ from a straight line fit ($y = mx + b$) | If undistortion is skipped or uses wrong coefficients, the touchline bows by $30+\text{ pixels}$. |
| **Homography Matrix** | Inverse Reprojection Error | $\frac{1}{N}\sum \| p_i - H^{-1}(H(p_i)) \|_2 < 0.15\text{ meters}$ | Proves pixel-to-meter and meter-to-pixel matrices are invertible and accurate. |
| **Player Detection** | Recall on 8v8 Sample Frames | $\ge 14$ of $16$ active players detected per frame ($\ge 87.5\%$) | Default thresholds miss far-side players ($25\text{ px}$ tall). Requires `imgsz=1280` or SAHI. |
| **Team Clustering** | Silhouette Score & Balance | Cluster split between $6\text{–}9$ players per team; Silhouette $> 0.35$ | Prevents K-Means from grouping by shadow/lighting instead of jersey color. |
| **Kinematics Speed Cap** | Maximum Human Turf Velocity | $0.0 \le v_{max} \le 34.0\text{ km/h}$ ($9.44\text{ m/s}$) | Unsmoothed tracker ID swaps cause instant $120\text{ km/h}$ teleportation spikes. |
| **Stationary Jitter** | Distance Walked by Stationary GK | $< 0.3\text{ meters}$ over a $3.0\text{s}$ stationary window | Without Savitzky-Golay smoothing, bounding box flicker adds $2\text{–}4\text{ m}$ of fake movement. |
| **3D Ground Locking** | Foot Penetration / Floating Error | $\min(Z_{left\_ankle}, Z_{right\_ankle}) \in [0.00\text{ m}, 0.08\text{ m}]$ during stance phase | Prevents 3D players from sinking under the turf or floating $0.5\text{ m}$ in the air. |
