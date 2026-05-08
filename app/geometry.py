from typing import Dict, List, Optional, Tuple
import numpy as np


def clamp_box(box: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width - 1, x2)))
    y2 = int(max(0, min(height - 1, y2)))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]


def resize_keep_aspect(frame, resize_width: int):
    height, width = frame.shape[:2]
    if resize_width <= 0 or width <= resize_width:
        return frame, 1.0
    scale = resize_width / float(width)
    new_height = int(height * scale)
    import cv2
    return cv2.resize(frame, (resize_width, new_height)), scale


def fallback_shirt_polygon_from_box(box: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    px1 = x1 + int(0.22 * w)
    px2 = x2 - int(0.22 * w)
    py1 = y1 + int(0.22 * h)
    py2 = y1 + int(0.56 * h)
    return np.array([[px1, py1], [px2, py1], [px2, py2], [px1, py2]], dtype=np.int32)


def bbox_from_points(points: List[Tuple[int, int]], width: int, height: int, padding: int = 20) -> List[int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return clamp_box([min(xs) - padding, min(ys) - padding, max(xs) + padding, max(ys) + padding], width, height)
