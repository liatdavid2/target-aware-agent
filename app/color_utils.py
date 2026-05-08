import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


COLOR_ALIASES = {
    "black": ["black", "שחור", "שחורה", "חולצה שחורה", "black shirt"],
    "red": ["red", "אדום", "אדומה", "חולצה אדומה", "red shirt"],
    "blue": ["blue", "כחול", "כחולה", "חולצה כחולה", "blue shirt"],
    "white": ["white", "לבן", "לבנה", "חולצה לבנה", "white shirt"],
}


@dataclass
class ColorMatchResult:
    color_name: str
    match_score: float
    ratios: Dict[str, float]
    mean_hsv: Tuple[float, float, float]
    pixel_count: int


def parse_target_color(target_query: str) -> str:
    normalized = (target_query or "").strip().lower()
    for color, aliases in COLOR_ALIASES.items():
        for alias in aliases:
            if alias.lower() in normalized:
                return color
    return "black"


def _hsv_mask_for_color(hsv_pixels: np.ndarray, color_name: str) -> np.ndarray:
    h = hsv_pixels[:, 0]
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]

    if color_name == "black":
        return (v < 80).astype(np.uint8)

    if color_name == "white":
        return ((s < 45) & (v > 170)).astype(np.uint8)

    if color_name == "red":
        lower_red = ((h <= 10) | (h >= 170)) & (s > 70) & (v > 50)
        return lower_red.astype(np.uint8)

    if color_name == "blue":
        return ((h >= 90) & (h <= 130) & (s > 60) & (v > 50)).astype(np.uint8)

    return np.zeros_like(h, dtype=np.uint8)


def evaluate_color_in_polygon(frame_bgr: np.ndarray, polygon: np.ndarray, color_name: str) -> ColorMatchResult:
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    polygon = polygon.astype(np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv[mask > 0]

    if hsv_pixels.size == 0:
        return ColorMatchResult(
            color_name=color_name,
            match_score=0.0,
            ratios={color_name: 0.0},
            mean_hsv=(0.0, 0.0, 0.0),
            pixel_count=0,
        )

    selected = _hsv_mask_for_color(hsv_pixels, color_name)
    score = float(selected.mean())
    mean_hsv = tuple(float(x) for x in hsv_pixels.mean(axis=0))

    return ColorMatchResult(
        color_name=color_name,
        match_score=score,
        ratios={color_name: score},
        mean_hsv=mean_hsv,
        pixel_count=int(hsv_pixels.shape[0]),
    )
