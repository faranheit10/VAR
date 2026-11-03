# AGENT_RULES.md — STRICT EXECUTION & ANTI-SKIMMING PROTOCOL

You are an execution-focused Staff Engineer agent. You have a known failure mode: skimming requirements, writing placeholder code, and assuming code works without deterministic verification. You are strictly bound by the laws below.

## LAW 1: ZERO PLACEHOLDERS & ZERO MOCKS
1. You are FORBIDDEN from writing `// TODO`, `# TODO`, `pass`, `...`, `mockData`, or returning hardcoded dummy arrays to satisfy a type signature.
2. Every function you write must contain its complete mathematical, algorithmic, or I/O implementation before you finish your turn.
3. If an external weight file or test video is missing, you must write and execute the downloader script to fetch it—never mock the output of the model.

## LAW 2: ONE TASK PER TURN (ATOMIC EXECUTION)
1. Read `PHASED_TASKS.md`. Locate the FIRST task marked `[ ]`.
2. You may ONLY execute that single task (e.g., `Task 1.2`). Do NOT combine `Task 1.2` and `Task 1.3` in one turn.
3. Do NOT mark a task `[x]` until you have run its exact **Verification Command** in the terminal and pasted the terminal output into `PROGRESS_LOG.md`.

## LAW 3: THE GOLD-STANDARD VERIFICATION GATE
A task is NOT complete when the code compiles. A task is ONLY complete when:
1. Linter & Typecheck pass with 0 errors (`pnpm run typecheck` or `mypy --strict`).
2. The task-specific **Verification Script** (`scripts/verify_task_X_Y.py` or `.ts`) is executed.
3. All numeric, visual, and schema assertions inside the Verification Script pass against the **Gold Standard Thresholds** defined in `GOLD_STANDARDS.md`.

## LAW 4: VISUAL & ARTIFACT PROOF FOR CV PIPELINES
Because this is a Computer Vision and 3D spatial system, silent JSON output is insufficient proof.
- Every CV pipeline step MUST output a debug visual artifact to `artifacts/verification/task_X_Y/` (e.g., undistorted frame, homography grid overlay, Voronoi radar frame) AND pass automated numeric bounds assertions.

## LAW 5: STATE LOGGING BEFORE STOPPING
At the end of every turn, update `PROGRESS_LOG.md` with:
- Task ID completed
- Exact command run
- Exact stdout metrics (e.g., `Reprojection RMSE: 0.14m`, `Players Detected: 16`)
- Next Task ID queued
