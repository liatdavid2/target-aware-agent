from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.geometry import bbox_from_points, fallback_shirt_polygon_from_box


@dataclass
class PoseResult:
    pose_detected: bool
    landmarks: Dict[str, Tuple[int, int]]
    shirt_polygon: np.ndarray
    bbox_from_pose: Optional[List[int]]


class PoseEstimator:
    def __init__(self):
        try:
            import mediapipe as mp
        except Exception as exc:
            raise RuntimeError("MediaPipe is not installed. Run: pip install -r requirements.txt") from exc

        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.45,
        )
        self.landmark_enum = mp.solutions.pose.PoseLandmark

    def close(self):
        self.pose.close()

    def estimate_on_crop(self, frame_bgr: np.ndarray, bbox: List[int]) -> PoseResult:
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return PoseResult(False, {}, fallback_shirt_polygon_from_box(bbox), None)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.pose.process(crop_rgb)
        if not result.pose_landmarks:
            return PoseResult(False, {}, fallback_shirt_polygon_from_box(bbox), None)

        crop_h, crop_w = crop.shape[:2]
        landmarks = {}
        for lm_name in self.landmark_enum:
            lm = result.pose_landmarks.landmark[lm_name.value]
            if lm.visibility < 0.25:
                continue
            px = int(x1 + lm.x * crop_w)
            py = int(y1 + lm.y * crop_h)
            if 0 <= px < width and 0 <= py < height:
                landmarks[lm_name.name.lower()] = (px, py)

        required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if all(name in landmarks for name in required):
            shirt_polygon = np.array(
                [
                    landmarks["left_shoulder"],
                    landmarks["right_shoulder"],
                    landmarks["right_hip"],
                    landmarks["left_hip"],
                ],
                dtype=np.int32,
            )
        else:
            shirt_polygon = fallback_shirt_polygon_from_box(bbox)

        bbox_points = list(landmarks.values())
        pose_box = bbox_from_points(bbox_points, width, height, padding=24) if bbox_points else bbox

        return PoseResult(
            pose_detected=True,
            landmarks=landmarks,
            shirt_polygon=shirt_polygon,
            bbox_from_pose=pose_box,
        )
