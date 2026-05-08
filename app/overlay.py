from typing import Dict, List, Tuple
import cv2
import numpy as np


SKELETON_PAIRS = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def draw_pose(frame, landmarks: Dict[str, Tuple[int, int]]):
    for a, b in SKELETON_PAIRS:
        if a in landmarks and b in landmarks:
            cv2.line(frame, landmarks[a], landmarks[b], (60, 255, 60), 2, cv2.LINE_AA)
    for point in landmarks.values():
        cv2.circle(frame, point, 3, (0, 255, 255), -1)

def draw_person_segmentation(frame, candidate: dict):
    polygon = candidate.get("mask_polygon")

    if polygon is None:
        return

    polygon = np.array(polygon, dtype=np.int32)

    if polygon.ndim != 2 or polygon.shape[0] < 3:
        return

    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (80, 160, 255))
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    cv2.polylines(frame, [polygon], True, (80, 160, 255), 2)

def draw_target_overlay(frame, candidate: dict, target_color: str):
    x1, y1, x2, y2 = candidate["bbox"]
    score = candidate["match_score"]
    label = f"TARGET | {target_color} shirt | match {score:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 255), 3)

    draw_person_segmentation(frame, candidate)

    polygon = np.array(candidate["shirt_polygon"], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (40, 40, 40))
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.polylines(frame, [polygon], True, (255, 255, 255), 2)

    draw_pose(frame, candidate.get("landmarks", {}))

    y_label_top = max(0, y1 - 38)
    #cv2.rectangle(frame, (x1, y_label_top), (min(frame.shape[1] - 1, x1 + 390), y1), (0, 0, 0), -1)
    #cv2.putText(frame, label, (x1 + 7, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA)


def draw_non_target_overlay(frame, candidate: dict):
    x1, y1, x2, y2 = candidate["bbox"]
    #cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
    #cv2.putText(frame, "person", (x1 + 5, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_header(frame, target_query: str, frame_id: int):
    h, w = frame.shape[:2]
    #cv2.rectangle(frame, (0, 0), (w, 66), (18, 20, 25), -1)
    #cv2.putText(frame, "Target-aware video analysis", (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    #cv2.putText(frame, f"target: {target_query} | frame: {frame_id}", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (215, 215, 215), 1, cv2.LINE_AA)
