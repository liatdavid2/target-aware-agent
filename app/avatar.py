import cv2
import numpy as np


POSE_LINES = [
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


def blank_avatar(height: int, width: int = 360) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 28)
    return canvas


def normalize_to_avatar(landmarks: dict, out_w: int, out_h: int) -> dict:
    if not landmarks:
        return {}

    xs = [p[0] for p in landmarks.values()]
    ys = [p[1] for p in landmarks.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    body_w = max(max_x - min_x, 1)
    body_h = max(max_y - min_y, 1)

    scale = min(out_w * 0.62 / body_w, out_h * 0.78 / body_h)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    mapped = {}

    for name, point in landmarks.items():
        x, y = point
        nx = int((x - center_x) * scale + out_w / 2)
        ny = int((y - center_y) * scale + out_h / 2 + 20)
        mapped[name] = (nx, ny)

    return mapped


def draw_avatar(landmarks: dict, height: int, width: int = 360) -> np.ndarray:
    canvas = blank_avatar(height, width)

    if not landmarks:
        return canvas

    points = normalize_to_avatar(landmarks, width, height)

    for a, b in POSE_LINES:
        if a in points and b in points:
            cv2.line(canvas, points[a], points[b], (80, 220, 120), 8, cv2.LINE_AA)

    for point in points.values():
        cv2.circle(canvas, point, 7, (0, 230, 255), -1)

    if "nose" in points:
        cv2.circle(canvas, points["nose"], 18, (210, 210, 230), -1)

    return canvas


def combine_with_avatar(frame: np.ndarray, avatar: np.ndarray) -> np.ndarray:
    if avatar.shape[0] != frame.shape[0]:
        avatar = cv2.resize(avatar, (avatar.shape[1], frame.shape[0]))

    return np.hstack([frame, avatar])