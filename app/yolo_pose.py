from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import MODELS_DIR
from app.geometry import clamp_box


Point = Tuple[int, int]


YOLO_POSE_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass
class YoloPoseResult:
    bbox: List[int]
    confidence: float
    landmarks: Dict[str, Point]


def box_iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


class YoloPoseEstimator:
    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        confidence: float = 0.25,
        keypoint_confidence: float = 0.25,
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.keypoint_confidence = keypoint_confidence
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        local_model = MODELS_DIR / self.model_name
        model_path = str(local_model) if local_model.exists() else self.model_name
        self.model = YOLO(model_path)

    def estimate(self, frame_bgr: np.ndarray, max_people: int = 3) -> List[YoloPoseResult]:
        self._load_model()

        height, width = frame_bgr.shape[:2]

        results = self.model.predict(
            frame_bgr,
            conf=self.confidence,
            device="cpu",
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)

        if boxes is None or boxes.xyxy is None or keypoints is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(xyxy)

        keypoints_xy = keypoints.xy.cpu().numpy()
        keypoints_conf = None

        if getattr(keypoints, "conf", None) is not None:
            keypoints_conf = keypoints.conf.cpu().numpy()

        pose_results: List[YoloPoseResult] = []

        for person_idx, (box, conf) in enumerate(zip(xyxy, confs)):
            landmarks: Dict[str, Point] = {}

            if person_idx >= len(keypoints_xy):
                continue

            for kp_idx, name in enumerate(YOLO_POSE_KEYPOINTS):
                if kp_idx >= keypoints_xy.shape[1]:
                    continue

                if keypoints_conf is not None:
                    kp_conf = float(keypoints_conf[person_idx, kp_idx])
                    if kp_conf < self.keypoint_confidence:
                        continue

                x, y = keypoints_xy[person_idx, kp_idx]
                if x <= 0 or y <= 0:
                    continue

                landmarks[name] = (int(x), int(y))

            bbox = clamp_box(box.tolist(), width, height)

            pose_results.append(
                YoloPoseResult(
                    bbox=bbox,
                    confidence=float(conf),
                    landmarks=landmarks,
                )
            )

        pose_results.sort(
            key=lambda p: (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1]),
            reverse=True,
        )

        return pose_results[:max_people]


def find_best_pose_for_detection(
    detection_bbox: List[int],
    pose_results: List[YoloPoseResult],
    min_iou: float = 0.20,
) -> Optional[YoloPoseResult]:
    best_pose = None
    best_iou = 0.0

    for pose in pose_results:
        current_iou = box_iou(detection_bbox, pose.bbox)

        if current_iou > best_iou:
            best_iou = current_iou
            best_pose = pose

    if best_pose is None or best_iou < min_iou:
        return None

    return best_pose