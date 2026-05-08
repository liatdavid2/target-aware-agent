import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


COLOR_ALIASES = {
    "black": ["black", "שחור", "שחורה", "חולצה שחורה", "black shirt"],
    "white": ["white", "לבן", "לבנה", "חולצה לבנה", "white shirt"],
    "gray": ["gray", "grey", "אפור", "אפורה", "חולצה אפורה", "gray shirt", "grey shirt"],

    "red": ["red", "אדום", "אדומה", "חולצה אדומה", "red shirt"],
    "orange": ["orange", "כתום", "כתומה", "חולצה כתומה", "orange shirt"],
    "yellow": ["yellow", "צהוב", "צהובה", "חולצה צהובה", "yellow shirt"],

    "green": ["green", "ירוק", "ירוקה", "חולצה ירוקה", "green shirt"],
    "cyan": ["cyan", "turquoise", "תכלת", "טורקיז", "חולצה תכלת", "cyan shirt", "turquoise shirt"],
    "blue": ["blue", "כחול", "כחולה", "חולצה כחולה", "blue shirt"],

    "purple": ["purple", "סגול", "סגולה", "חולצה סגולה", "purple shirt"],
    "pink": ["pink", "ורוד", "ורודה", "חולצה ורודה", "pink shirt"],
    "brown": ["brown", "חום", "חומה", "חולצה חומה", "brown shirt"],
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
        return (v < 85).astype(np.uint8)

    if color_name == "white":
        return ((s < 45) & (v > 170)).astype(np.uint8)

    if color_name == "gray":
        return ((s < 45) & (v >= 70) & (v <= 190)).astype(np.uint8)

    if color_name == "red":
        return (((h <= 10) | (h >= 170)) & (s > 70) & (v > 50)).astype(np.uint8)

    if color_name == "orange":
        return ((h >= 8) & (h <= 24) & (s > 70) & (v > 70)).astype(np.uint8)

    if color_name == "yellow":
        strong_yellow = (h >= 22) & (h <= 42) & (s > 45) & (v > 75)
        beige_yellow = (h >= 12) & (h <= 45) & (s > 25) & (v > 55)
        return (strong_yellow | beige_yellow).astype(np.uint8)

    if color_name == "green":
        return ((h >= 40) & (h <= 85) & (s > 50) & (v > 50)).astype(np.uint8)

    if color_name == "cyan":
        return ((h >= 80) & (h <= 100) & (s > 45) & (v > 60)).astype(np.uint8)

    if color_name == "blue":
        return ((h >= 95) & (h <= 130) & (s > 50) & (v > 50)).astype(np.uint8)

    if color_name == "purple":
        return ((h >= 130) & (h <= 155) & (s > 45) & (v > 50)).astype(np.uint8)

    if color_name == "pink":
        return ((h >= 150) & (h <= 169) & (s > 40) & (v > 80)).astype(np.uint8)

    if color_name == "brown":
        return ((h >= 8) & (h <= 25) & (s > 45) & (v >= 35) & (v <= 170)).astype(np.uint8)

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
