from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

from app.config import MODELS_DIR
from app.geometry import clamp_box


@dataclass
class Detection:
    bbox: List[int]
    confidence: float
    class_name: str = "person"


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.35):
        self.model_name = model_name
        self.confidence = confidence
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

    def detect(self, frame_bgr: np.ndarray, max_people: int = 3) -> List[Detection]:
        self._load_model()
        height, width = frame_bgr.shape[:2]
        results = self.model.predict(
            frame_bgr,
            conf=self.confidence,
            classes=[0],
            device="cpu",
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(xyxy)

        for box, conf in zip(xyxy, confs):
            detections.append(
                Detection(
                    bbox=clamp_box(box.tolist(), width, height),
                    confidence=float(conf),
                )
            )

        detections.sort(key=lambda det: (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]), reverse=True)
        return detections[:max_people]


def full_frame_detection(frame_bgr: np.ndarray) -> List[Detection]:
    height, width = frame_bgr.shape[:2]
    return [Detection(bbox=[0, 0, width - 1, height - 1], confidence=1.0, class_name="full_frame")]
