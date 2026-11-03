"""High-Resolution Player & Goalkeeper Detector module.

Wraps Ultralytics YOLO11 (imgsz=1280, conf=0.25, iou=0.45) for detecting
players, goalkeepers, and referees on 8v8 turf football pitches, and extracts
precise bottom-center foot contact anchor points [u_c, v_max] for ground-plane
homography projection.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO  # type: ignore[attr-defined]

from calibration.homography import ViewTransformer


# Strict algorithmic hyperparameter defaults per PHASED_TASKS.md Task 2.1
DEFAULT_IMGSZ: int = 1280
DEFAULT_CONF_THRESHOLD: float = 0.25
DEFAULT_IOU_THRESHOLD: float = 0.45
DEFAULT_PERSON_CLASS_ID: int = 0  # COCO class 0: person


def extract_foot_coordinates(bboxes: np.ndarray) -> np.ndarray:
    """Extract bottom-center foot anchor coordinates [u_c, v_max] from bounding boxes.

    For each bounding box [x1, y1, x2, y2]:
        u_c = (x1 + x2) / 2.0
        v_max = y2

    Args:
        bboxes: Array of shape (N, 4) containing [x1, y1, x2, y2] bounding boxes.

    Returns:
        Array of shape (N, 2) of type float64 representing [u_c, v_max] pixel coordinates.
    """
    arr = np.asarray(bboxes, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == 4:
        arr = arr.reshape(1, 4)
    elif arr.ndim != 2 or (arr.shape[0] > 0 and arr.shape[1] != 4):
        raise ValueError(f"Bounding boxes array must have shape (N, 4), got {arr.shape}")

    if arr.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]

    if np.any(x1 > x2) or np.any(y1 > y2):
        raise ValueError("Invalid bounding box detected: x1 > x2 or y1 > y2")

    u_c = (x1 + x2) / 2.0
    v_max = y2.copy()

    foot_coords = np.column_stack([u_c, v_max]).astype(np.float64)
    return foot_coords


@dataclass
class PlayerDetection:
    """Container for an individual detected player bounding box and foot anchor."""

    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    foot_xy: Tuple[float, float]
    meter_xy: Optional[Tuple[float, float]] = None
    is_inside_pitch: Optional[bool] = None

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def foot_u(self) -> float:
        return self.foot_xy[0]

    @property
    def foot_v(self) -> float:
        return self.foot_xy[1]


@dataclass
class PlayerDetectionResult:
    """Container for detection results across a single video frame."""

    detections: List[PlayerDetection]
    bboxes: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    foot_coordinates: np.ndarray
    sv_detections: sv.Detections
    meter_coordinates: Optional[np.ndarray] = None
    inside_pitch_mask: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return len(self.detections)

    def __len__(self) -> int:
        return self.count


class PlayerDetector:
    """High-Resolution Player and Goalkeeper Detector wrapping Ultralytics YOLO11."""

    def __init__(
        self,
        model_path: Union[str, Path] = "yolo11s.pt",
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF_THRESHOLD,
        iou: float = DEFAULT_IOU_THRESHOLD,
        classes: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the YOLO11 player detector.

        Args:
            model_path: Path or identifier of the YOLO11 model checkpoint (default 'yolo11s.pt').
            imgsz: Input resolution for YOLO inference (default 1280).
            conf: Minimum confidence threshold (default 0.25).
            iou: NMS IoU threshold (default 0.45).
            classes: Sequence of COCO class IDs to retain (default [0] for 'person').
            device: Compute device ('cuda', 'cpu', or None for auto-selection).
        """
        self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes: List[int] = list(classes) if classes is not None else [DEFAULT_PERSON_CLASS_ID]

        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        self.model: Optional[YOLO] = YOLO(str(self.model_path))

    def detect(
        self,
        frame: np.ndarray,
        view_transformer: Optional[ViewTransformer] = None,
        pitch_margin_m: float = 0.5,
    ) -> PlayerDetectionResult:
        """Run high-resolution player detection on a single video frame.

        Args:
            frame: Input BGR image frame as numpy array (H, W, 3).
            view_transformer: Optional ViewTransformer instance to compute ground-plane turf meters.
            pitch_margin_m: Turf boundary margin in meters for inside-pitch filtering.

        Returns:
            PlayerDetectionResult containing structured detections, foot anchors, and sv.Detections.
        """
        if self.model is None:
            raise RuntimeError("Model is unloaded. Re-instantiate or reload before calling detect().")

        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError(f"Expected 3D BGR image array (H, W, 3), got {type(frame)}")

        raw_predictions = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )
        prediction_list: List[Any] = list(raw_predictions)

        first_res = prediction_list[0]
        boxes_obj = getattr(first_res, "boxes", None)

        if boxes_obj is None or len(boxes_obj) == 0:
            empty_bboxes = np.empty((0, 4), dtype=np.float64)
            empty_conf = np.empty((0,), dtype=np.float64)
            empty_cls = np.empty((0,), dtype=np.int64)
            empty_feet = np.empty((0, 2), dtype=np.float64)
            empty_sv = sv.Detections.empty()
            return PlayerDetectionResult(
                detections=[],
                bboxes=empty_bboxes,
                confidences=empty_conf,
                class_ids=empty_cls,
                foot_coordinates=empty_feet,
                sv_detections=empty_sv,
                meter_coordinates=None,
                inside_pitch_mask=None,
            )

        raw_xyxy = getattr(boxes_obj, "xyxy", None)
        raw_conf = getattr(boxes_obj, "conf", None)
        raw_cls = getattr(boxes_obj, "cls", None)

        if raw_xyxy is None or raw_conf is None or raw_cls is None:
            empty_bboxes = np.empty((0, 4), dtype=np.float64)
            empty_conf = np.empty((0,), dtype=np.float64)
            empty_cls = np.empty((0,), dtype=np.int64)
            empty_feet = np.empty((0, 2), dtype=np.float64)
            empty_sv = sv.Detections.empty()
            return PlayerDetectionResult(
                detections=[],
                bboxes=empty_bboxes,
                confidences=empty_conf,
                class_ids=empty_cls,
                foot_coordinates=empty_feet,
                sv_detections=empty_sv,
                meter_coordinates=None,
                inside_pitch_mask=None,
            )

        bboxes_xyxy = (
            raw_xyxy.cpu().numpy().astype(np.float64)
            if hasattr(raw_xyxy, "cpu")
            else np.asarray(raw_xyxy, dtype=np.float64)
        )
        confidences = (
            raw_conf.cpu().numpy().astype(np.float64)
            if hasattr(raw_conf, "cpu")
            else np.asarray(raw_conf, dtype=np.float64)
        )
        class_ids = (
            raw_cls.cpu().numpy().astype(np.int64)
            if hasattr(raw_cls, "cpu")
            else np.asarray(raw_cls, dtype=np.int64)
        )

        foot_coordinates = extract_foot_coordinates(bboxes_xyxy)

        # Build supervision Detections object for ByteTrack compatibility
        sv_detections = sv.Detections(
            xyxy=bboxes_xyxy.astype(np.float32),
            confidence=confidences.astype(np.float32),
            class_id=class_ids.astype(int),
            data={"foot_coordinates": foot_coordinates},
        )

        meter_coordinates: Optional[np.ndarray] = None
        inside_mask: Optional[np.ndarray] = None

        if view_transformer is not None:
            meter_coordinates = view_transformer.pixel_to_meter(foot_coordinates)
            inside_mask = view_transformer.is_inside_pitch(meter_coordinates, margin_m=pitch_margin_m)

        detections: List[PlayerDetection] = []
        for i in range(len(bboxes_xyxy)):
            box = tuple(float(c) for c in bboxes_xyxy[i])
            conf_val = float(confidences[i])
            cls_val = int(class_ids[i])
            foot_pt = (float(foot_coordinates[i, 0]), float(foot_coordinates[i, 1]))

            meter_pt: Optional[Tuple[float, float]] = None
            is_inside: Optional[bool] = None

            if meter_coordinates is not None:
                meter_pt = (float(meter_coordinates[i, 0]), float(meter_coordinates[i, 1]))
            if inside_mask is not None:
                is_inside = bool(inside_mask[i])

            detections.append(
                PlayerDetection(
                    bbox=(box[0], box[1], box[2], box[3]),
                    confidence=conf_val,
                    class_id=cls_val,
                    foot_xy=foot_pt,
                    meter_xy=meter_pt,
                    is_inside_pitch=is_inside,
                )
            )

        return PlayerDetectionResult(
            detections=detections,
            bboxes=bboxes_xyxy,
            confidences=confidences,
            class_ids=class_ids,
            foot_coordinates=foot_coordinates,
            sv_detections=sv_detections,
            meter_coordinates=meter_coordinates,
            inside_pitch_mask=inside_mask,
        )

    def detect_batch(
        self,
        frames: Sequence[np.ndarray],
        view_transformer: Optional[ViewTransformer] = None,
        pitch_margin_m: float = 0.5,
    ) -> List[PlayerDetectionResult]:
        """Run player detection across a batch of frames."""
        results: List[PlayerDetectionResult] = []
        for frame in frames:
            res = self.detect(
                frame,
                view_transformer=view_transformer,
                pitch_margin_m=pitch_margin_m,
            )
            results.append(res)
        return results

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: PlayerDetectionResult,
        draw_bboxes: bool = True,
        draw_foot_anchors: bool = True,
        draw_labels: bool = True,
        draw_hud_banner: bool = True,
        box_color: Tuple[int, int, int] = (0, 220, 255),
        anchor_color: Tuple[int, int, int] = (0, 255, 0),
        anchor_radius: int = 7,
    ) -> np.ndarray:
        """Draw bounding boxes, confidence labels, and foot anchor points onto frame.

        Args:
            frame: Input BGR image frame.
            result: Detection results from detect().
            draw_bboxes: Whether to draw bounding box rectangles.
            draw_foot_anchors: Whether to draw prominent circular foot anchor markers.
            draw_labels: Whether to draw text labels with confidence.
            draw_hud_banner: Whether to overlay top HUD banner with detection statistics.
            box_color: BGR color for bounding boxes (default amber/gold: (0, 220, 255)).
            anchor_color: BGR color for foot anchor points (default bright green: (0, 255, 0)).
            anchor_radius: Radius in pixels for foot anchor circle.

        Returns:
            Annotated BGR image as numpy array.
        """
        annotated = frame.copy()
        height, width = annotated.shape[:2]

        for det in result.detections:
            x1, y1, x2, y2 = (int(round(det.x1)), int(round(det.y1)), int(round(det.x2)), int(round(det.y2)))
            u_c, v_max = (int(round(det.foot_u)), int(round(det.foot_v)))

            # 1. Draw bounding box
            if draw_bboxes:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness=2, lineType=cv2.LINE_AA)

            # 2. Draw foot anchor point (high-visibility concentric target marker)
            if draw_foot_anchors:
                # Black outer halo ring for contrast against any turf grass shade
                cv2.circle(annotated, (u_c, v_max), anchor_radius + 2, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                # Colored filled disk
                cv2.circle(annotated, (u_c, v_max), anchor_radius, anchor_color, thickness=-1, lineType=cv2.LINE_AA)
                # Small white center dot
                cv2.circle(annotated, (u_c, v_max), 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                # Horizontal ground contact hairline
                cv2.line(annotated, (u_c - 12, v_max), (u_c + 12, v_max), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.line(annotated, (u_c - 10, v_max), (u_c + 10, v_max), anchor_color, thickness=1, lineType=cv2.LINE_AA)

            # 3. Draw label
            if draw_labels:
                label_text = f"Player {det.confidence:.2f}"
                if det.is_inside_pitch is not None:
                    label_text += " [Pitch]" if det.is_inside_pitch else " [Touchline]"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                label_bg_y1 = max(0, y1 - text_h - baseline - 4)
                label_bg_y2 = y1
                cv2.rectangle(
                    annotated,
                    (x1, label_bg_y1),
                    (x1 + text_w + 6, label_bg_y2),
                    (20, 20, 20),
                    thickness=-1,
                )
                cv2.putText(
                    annotated,
                    label_text,
                    (x1 + 3, label_bg_y2 - baseline - 1),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        # 4. Draw HUD banner
        if draw_hud_banner:
            banner_height = 42
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (width, banner_height), (15, 23, 42), thickness=-1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            cv2.line(annotated, (0, banner_height), (width, banner_height), (56, 189, 248), thickness=1)

            hud_title = "TURFVISION YOLO11 PLAYER DETECTOR"
            hud_stats = (
                f"Players Detected: {result.count} | "
                f"Model: {self.model_path.name} | "
                f"imgsz: {self.imgsz} | "
                f"conf >= {self.conf} | "
                f"iou: {self.iou}"
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated, hud_title, (16, 26), font, 0.65, (56, 189, 248), 2, lineType=cv2.LINE_AA)
            cv2.putText(annotated, hud_stats, (420, 26), font, 0.52, (241, 245, 249), 1, lineType=cv2.LINE_AA)

        return annotated

    def unload(self) -> None:
        """Offload model weights from memory per MODEL_AND_HARDWARE_MANIFEST.md."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self) -> PlayerDetector:
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.unload()
