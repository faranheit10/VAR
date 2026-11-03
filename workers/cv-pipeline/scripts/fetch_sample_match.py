"""Script to obtain or transcode the sample 8v8 football match clip and generate reference calibration JSON.

Produces:
1. artifacts/sample_data/sample_8v8.mp4 (15s, >= 25fps, >= 1920 width, >= 300 frames)
2. artifacts/sample_data/sample_calibration.json (valid PitchCalibrationPayload)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import cv2


SAMPLE_DATA_DIR = Path("artifacts/sample_data")
OUTPUT_VIDEO_PATH = SAMPLE_DATA_DIR / "sample_8v8.mp4"
OUTPUT_CALIBRATION_PATH = SAMPLE_DATA_DIR / "sample_calibration.json"
LOCAL_WEBM_SOURCE = SAMPLE_DATA_DIR / "sample_8v8.webm"


def generate_reference_calibration() -> Dict[str, Any]:
    """Generate reference pitch calibration payload matching PitchCalibrationPayload data contract."""
    return {
      "match_id": "8v8-sample-match-001",
      "pitch_length_m": 50.0,
      "pitch_width_m": 32.0,
      "camera_intrinsics": {
        "k1": -0.28,
        "k2": 0.07,
        "p1": 0.0,
        "p2": 0.0,
        "fx": 1850.0,
        "fy": 1850.0,
        "cx": 960.0,
        "cy": 540.0,
      },
      "correspondences": [
        {"label": "top_left_corner", "pixel_u": 320.0, "pixel_v": 210.0, "meter_x": 0.0, "meter_y": 0.0},
        {"label": "top_right_corner", "pixel_u": 1600.0, "pixel_v": 210.0, "meter_x": 50.0, "meter_y": 0.0},
        {"label": "bottom_right_corner", "pixel_u": 1820.0, "pixel_v": 980.0, "meter_x": 50.0, "meter_y": 32.0},
        {"label": "bottom_left_corner", "pixel_u": 100.0, "pixel_v": 980.0, "meter_x": 0.0, "meter_y": 32.0},
      ],
    }


def transcode_with_ffmpeg(source_path: Path, dest_path: Path, duration_sec: int = 15) -> bool:
    """Use ffmpeg to transcode a segment of source video into MP4."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        return False

    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss",
        "00:00:05",
        "-i",
        str(source_path),
        "-t",
        str(duration_sec),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(dest_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 0


def transcode_with_opencv(source_path: Path, dest_path: Path, duration_sec: int = 15) -> bool:
    """Fallback transcoder using OpenCV VideoCapture/VideoWriter."""
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(fps * duration_sec)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(dest_path), fourcc, fps, (width, height))

    # Skip first 5 seconds
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 5))

    written = 0
    while written < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    return dest_path.exists() and written >= 300


def ensure_sample_video() -> Path:
    """Ensure artifacts/sample_data/sample_8v8.mp4 exists with required specs."""
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_VIDEO_PATH.exists() and OUTPUT_VIDEO_PATH.stat().st_size > 1000:
        cap = cv2.VideoCapture(str(OUTPUT_VIDEO_PATH))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if count >= 300 and w >= 1920 and fps >= 25.0:
            print(f"Sample video already exists and meets criteria: {OUTPUT_VIDEO_PATH}")
            return OUTPUT_VIDEO_PATH

    if LOCAL_WEBM_SOURCE.exists():
        print(f"Transcoding from local source {LOCAL_WEBM_SOURCE}...")
        success = transcode_with_ffmpeg(LOCAL_WEBM_SOURCE, OUTPUT_VIDEO_PATH, duration_sec=15)
        if not success:
            print("FFmpeg transcode failed or not found, falling back to OpenCV...")
            success = transcode_with_opencv(LOCAL_WEBM_SOURCE, OUTPUT_VIDEO_PATH, duration_sec=15)
        if success:
            print(f"Transcoded {OUTPUT_VIDEO_PATH} successfully.")
            return OUTPUT_VIDEO_PATH
        raise RuntimeError("Failed to transcode local source into sample_8v8.mp4")

    # If no local webm exists, raise an informative error
    raise FileNotFoundError(
        f"Source video {LOCAL_WEBM_SOURCE} not found. Please provide source footage."
    )


def ensure_calibration_fixture() -> Path:
    """Ensure artifacts/sample_data/sample_calibration.json exists and is valid."""
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = generate_reference_calibration()
    with open(OUTPUT_CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Written calibration fixture to {OUTPUT_CALIBRATION_PATH}")
    return OUTPUT_CALIBRATION_PATH


def main() -> int:
    """Main execution entry point."""
    print("=" * 60)
    print("FETCH / GENERATE SAMPLE 8v8 MATCH FIXTURES")
    print("=" * 60)
    try:
        video_path = ensure_sample_video()
        cal_path = ensure_calibration_fixture()

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"Video Path:        {video_path}")
        print(f"Resolution:        {width}x{height}")
        print(f"FPS:               {fps}")
        print(f"Frame Count:       {frame_count}")
        print(f"Calibration Path:  {cal_path}")
        print("=" * 60)
        print("FIXTURE SETUP COMPLETE")
        print("=" * 60)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
