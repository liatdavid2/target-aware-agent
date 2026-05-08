from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.config import MODELS_DIR
from app.geometry import clamp_box


@dataclass
class Detection:
    bbox: List[int]
    confidence: float
    class_name: str = "person"
    mask_polygon: Optional[List[List[int]]] = None
    track_id: Optional[int] = None


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n-seg.pt", confidence: float = 0.35):
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

        results = self.model.track(
            frame_bgr,
            conf=self.confidence,
            classes=[0],
            device="cpu",
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        detections: List[Detection] = []

        if not results:
            return detections

        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)

        if boxes is None or boxes.xyxy is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(xyxy)

        track_ids = [None] * len(xyxy)
        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()

        mask_polygons = []

        if masks is not None and getattr(masks, "xy", None) is not None:
            for poly in masks.xy:
                poly = np.asarray(poly, dtype=np.int32)

                if poly.ndim == 2 and poly.shape[0] >= 3:
                    mask_polygons.append(poly.tolist())
                else:
                    mask_polygons.append(None)

        for idx, (box, conf, track_id) in enumerate(zip(xyxy, confs, track_ids)):
            mask_polygon = None

            if idx < len(mask_polygons):
                mask_polygon = mask_polygons[idx]

            detections.append(
                Detection(
                    bbox=clamp_box(box.tolist(), width, height),
                    confidence=float(conf),
                    class_name="person",
                    mask_polygon=mask_polygon,
                    track_id=track_id,
                )
            )

        detections.sort(
            key=lambda det: (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]),
            reverse=True,
        )

        return detections[:max_people]


def full_frame_detection(frame_bgr: np.ndarray) -> List[Detection]:
    height, width = frame_bgr.shape[:2]

    return [
        Detection(
            bbox=[0, 0, width - 1, height - 1],
            confidence=1.0,
            class_name="full_frame",
            mask_polygon=None,
            track_id=None,
        )
    ]