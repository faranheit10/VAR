# MODEL_AND_HARDWARE_MANIFEST.md — EXACT WEIGHTS, URLS & VRAM BUDGET

## 1. Strict Hardware & Memory Budget
The local development and worker environment may run on constrained GPUs (4GB to 8GB VRAM) or CPU fallback (e.g., AMD Radeon RX 6400 / DirectML / ONNX Runtime or standard cloud T4).
- **LAW OF SEQUENTIAL MODEL LOADING:** You are FORBIDDEN from keeping YOLO, SigLIP, and RTMPose3D loaded in GPU/RAM simultaneously.
- Every pipeline stage must follow the **Load -> Batch Infer -> Offload (`del model; gc.collect()`)** pattern:
  1. Pass 1: Load Detection (`YOLO11s` / `RF-DETR` via ONNX Runtime) -> Write raw bounding boxes to disk (`artifacts/cache/raw_detections.parquet`) -> Unload model.
  2. Pass 2: Load `google/siglip-base-patch16-224` -> Extract torso embeddings for sampled frames -> Unload model.
  3. Pass 3: Load 3D Pose model ONLY for the 10-second goal clip windows -> Export 3D joints -> Unload model.

## 2. Exact Model Checkpoints & Runtimes (No Compiling `mmcv` From Source!)
Do NOT install full `mmpose` + `mmcv-full` from source—it consistently breaks on Windows/Linux CUDA mismatches. Use **ONNX Runtime (`onnxruntime-gpu` or `onnxruntime-directml`)** and **`rtmlib`** (lightweight ONNX wrapper for RTMPose/RTMPose3D).

| Stage | Exact Library / Package | Exact Model Checkpoint ID / URL | Input Resolution |
| :--- | :--- | :--- | :--- |
| **Player/Ref Detection** | `ultralytics` | `yolo11s.pt` (COCO classes: `0: person`, `32: sports ball`) or fine-tuned ONNX | `1280x1280` |
| **Ball Slicing** | `sahi` + `ultralytics` | `yolo11s.pt` (filtered strictly to class `32: sports ball`, `conf=0.15`) | `640x640` tiles (0.2 overlap) |
| **Team Embeddings** | `transformers` | `google/siglip-base-patch16-224` (HuggingFace Hub auto-download) | `224x224` |
| **2D/3D Pose Lifting** | `rtmlib` + `onnxruntime` | `rtmpose3d-l` ONNX checkpoint via `rtmlib` (`Body3d` class) or `MotionBERT-lite` | `256x192` crop |

## 3. Video Decoding Memory Rule
- NEVER load an entire 4K video into a Python list (`frames = [f for f in cap]`). A 60-second 4K clip uncompressed in RAM requires **44 GB of RAM** and will instantly crash the machine.
- You MUST use a generator (`sv.get_video_frames_generator(source_path)` from `supervision`) and yield one frame at a time.
