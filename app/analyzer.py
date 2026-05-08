import csv
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.color_utils import evaluate_color_in_polygon, parse_target_color
from app.config import AnalyzerConfig, OUTPUTS_DIR
from app.detector import PersonDetector, full_frame_detection
from app.geometry import resize_keep_aspect
from app.overlay import draw_header, draw_non_target_overlay, draw_target_overlay
from app.yolo_pose import YoloPoseEstimator, find_best_pose_for_detection


AVATAR_CONNECTIONS = [
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


def shirt_polygon_from_landmarks(landmarks: dict, fallback_bbox: List[int]) -> np.ndarray:
    required = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]

    if all(name in landmarks for name in required):
        return np.array(
            [
                landmarks["left_shoulder"],
                landmarks["right_shoulder"],
                landmarks["right_hip"],
                landmarks["left_hip"],
            ],
            dtype=np.int32,
        )

    x1, y1, x2, y2 = fallback_bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)

    return np.array(
        [
            [x1 + int(0.22 * w), y1 + int(0.18 * h)],
            [x2 - int(0.22 * w), y1 + int(0.18 * h)],
            [x2 - int(0.18 * w), y1 + int(0.58 * h)],
            [x1 + int(0.18 * w), y1 + int(0.58 * h)],
        ],
        dtype=np.int32,
    )


def _candidate_to_csv_row(frame_id: int, timestamp_sec: float, candidate: dict) -> dict:
    x1, y1, x2, y2 = candidate["bbox"]
    mean_h, mean_s, mean_v = candidate["mean_hsv"]

    return {
        "frame_id": frame_id,
        "timestamp_sec": round(timestamp_sec, 3),
        "candidate_id": candidate["candidate_id"],
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "detector_confidence": round(candidate["detector_confidence"], 4),
        "pose_detected": candidate["pose_detected"],
        "target_color": candidate["target_color"],
        "match_score": round(candidate["match_score"], 4),
        "is_target": candidate["is_target"],
        "mean_h": round(mean_h, 2),
        "mean_s": round(mean_s, 2),
        "mean_v": round(mean_v, 2),
        "shirt_pixel_count": candidate["shirt_pixel_count"],
    }


def _point_from_landmark(value) -> Optional[Tuple[int, int]]:
    if value is None:
        return None

    if len(value) < 2:
        return None

    return int(value[0]), int(value[1])


def _get_landmark_points(landmarks: dict) -> dict:
    points = {}

    for name, value in landmarks.items():
        point = _point_from_landmark(value)

        if point is not None:
            points[name] = point

    return points


def _safe_bbox(candidate: dict) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = candidate["bbox"]

    return int(x1), int(y1), int(x2), int(y2)


def _draw_avatar_segmentation(canvas: np.ndarray, candidate: dict) -> None:
    mask_polygon = candidate.get("mask_polygon")

    if mask_polygon is None:
        return

    poly = np.asarray(mask_polygon, dtype=np.int32)

    if poly.ndim != 2 or poly.shape[0] < 3:
        return

    fill_color = (210, 230, 255)
    outline_color = (80, 130, 220)

    overlay = canvas.copy()

    cv2.fillPoly(
        overlay,
        [poly],
        fill_color,
        lineType=cv2.LINE_AA,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        canvas,
        0.45,
        0,
        canvas,
    )

    cv2.polylines(
        canvas,
        [poly],
        True,
        outline_color,
        2,
        lineType=cv2.LINE_AA,
    )


def _draw_fallback_avatar(canvas: np.ndarray, candidate: dict) -> None:
    x1, y1, x2, y2 = _safe_bbox(candidate)

    person_w = max(x2 - x1, 1)
    person_h = max(y2 - y1, 1)

    center_x = x1 + person_w // 2

    head_radius = max(8, int(person_w * 0.16))
    head_center = (center_x, y1 + int(person_h * 0.16))

    body_top = y1 + int(person_h * 0.28)
    body_bottom = y1 + int(person_h * 0.70)

    body_color = (100, 170, 255)
    limb_color = (70, 120, 220)
    head_color = (210, 210, 210)
    outline_color = (40, 40, 40)

    cv2.circle(canvas, head_center, head_radius, head_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, head_center, head_radius, outline_color, 2, lineType=cv2.LINE_AA)

    cv2.ellipse(
        canvas,
        (center_x, (body_top + body_bottom) // 2),
        (max(10, int(person_w * 0.22)), max(20, int(person_h * 0.22))),
        0,
        0,
        360,
        body_color,
        -1,
        lineType=cv2.LINE_AA,
    )

    thickness = max(3, int(person_w * 0.05))

    cv2.line(
        canvas,
        (center_x - int(person_w * 0.22), body_top),
        (center_x - int(person_w * 0.38), body_bottom),
        limb_color,
        thickness,
        lineType=cv2.LINE_AA,
    )

    cv2.line(
        canvas,
        (center_x + int(person_w * 0.22), body_top),
        (center_x + int(person_w * 0.38), body_bottom),
        limb_color,
        thickness,
        lineType=cv2.LINE_AA,
    )

    cv2.line(
        canvas,
        (center_x - int(person_w * 0.10), body_bottom),
        (center_x - int(person_w * 0.20), y2),
        limb_color,
        thickness,
        lineType=cv2.LINE_AA,
    )

    cv2.line(
        canvas,
        (center_x + int(person_w * 0.10), body_bottom),
        (center_x + int(person_w * 0.20), y2),
        limb_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_avatar_person(canvas: np.ndarray, candidate: dict) -> None:
    _draw_avatar_segmentation(canvas, candidate)

    landmarks = candidate.get("landmarks", {})
    points = _get_landmark_points(landmarks)

    x1, y1, x2, y2 = _safe_bbox(candidate)
    person_w = max(x2 - x1, 1)

    if not points:
        _draw_fallback_avatar(canvas, candidate)
        return

    body_color = (100, 170, 255)
    limb_color = (70, 120, 220)
    joint_color = (35, 35, 35)
    head_color = (220, 220, 220)

    line_thickness = max(3, int(person_w * 0.045))
    joint_radius = max(4, int(person_w * 0.035))

    torso_names = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]

    if all(name in points for name in torso_names):
        torso = np.array(
            [
                points["left_shoulder"],
                points["right_shoulder"],
                points["right_hip"],
                points["left_hip"],
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(canvas, [torso], body_color, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [torso], True, joint_color, 2, lineType=cv2.LINE_AA)

    for a, b in AVATAR_CONNECTIONS:
        if a not in points or b not in points:
            continue

        color = body_color if "shoulder" in a or "hip" in a else limb_color

        cv2.line(
            canvas,
            points[a],
            points[b],
            color,
            line_thickness,
            lineType=cv2.LINE_AA,
        )

    head_center = None

    if "nose" in points:
        head_center = points["nose"]
    elif "left_shoulder" in points and "right_shoulder" in points:
        lx, ly = points["left_shoulder"]
        rx, ry = points["right_shoulder"]
        head_center = ((lx + rx) // 2, min(ly, ry) - int(person_w * 0.22))

    if head_center is not None:
        head_radius = max(8, int(person_w * 0.14))

        cv2.circle(
            canvas,
            head_center,
            head_radius,
            head_color,
            -1,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            canvas,
            head_center,
            head_radius,
            joint_color,
            2,
            lineType=cv2.LINE_AA,
        )

    for point in points.values():
        cv2.circle(
            canvas,
            point,
            joint_radius,
            joint_color,
            -1,
            lineType=cv2.LINE_AA,
        )


def draw_natural_avatar_frame(
    frame_shape,
    candidates: List[dict],
    target_query: str,
    frame_id: int,
) -> np.ndarray:
    height, width = frame_shape[:2]

    avatar_frame = np.full((height, width, 3), 245, dtype=np.uint8)

    draw_header(avatar_frame, target_query, frame_id)

    for candidate in candidates:
        _draw_avatar_person(avatar_frame, candidate)

    return avatar_frame


class VideoAnalyzer:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.target_color = parse_target_color(config.target_query)

        self.detector = (
            PersonDetector(config.yolo_model_name, config.yolo_confidence)
            if config.use_yolo
            else None
        )

        self.yolo_pose = YoloPoseEstimator(
            model_name="yolov8n-pose.pt",
            confidence=0.25,
            keypoint_confidence=0.20,
        )

    def close(self):
        pass

    def _max_people_for_processing(self) -> int:
        if self.config.enable_avatar:
            return max(2, int(self.config.max_people))

        return int(self.config.max_people)

    def _detect_people(self, frame: np.ndarray):
        if self.config.use_yolo:
            return self.detector.detect(
                frame,
                max_people=self._max_people_for_processing(),
            )

        return full_frame_detection(frame)

    def _match_pose_to_detection(self, det, pose_results):
        if not pose_results:
            return None

        if getattr(det, "class_name", "") == "full_frame":
            return pose_results[0]

        return find_best_pose_for_detection(
            det.bbox,
            pose_results,
            min_iou=0.10,
        )

    def _analyze_candidates(self, frame: np.ndarray, detections, pose_results) -> List[dict]:
        candidates = []

        for idx, det in enumerate(detections):
            matched_pose = self._match_pose_to_detection(det, pose_results)

            if matched_pose is not None:
                landmarks = matched_pose.landmarks
                pose_detected = len(landmarks) > 0
            else:
                landmarks = {}
                pose_detected = False

            shirt_polygon = shirt_polygon_from_landmarks(
                landmarks=landmarks,
                fallback_bbox=det.bbox,
            )

            color_result = evaluate_color_in_polygon(
                frame,
                shirt_polygon,
                self.target_color,
            )

            is_target = color_result.match_score >= self.config.min_match_score

            candidate = {
                "candidate_id": idx,
                "bbox": [int(v) for v in det.bbox],
                "detector_bbox": [int(v) for v in det.bbox],
                "mask_polygon": getattr(det, "mask_polygon", None),
                "detector_confidence": float(det.confidence),
                "pose_detected": bool(pose_detected),
                "landmarks": {
                    k: [int(p[0]), int(p[1])]
                    for k, p in landmarks.items()
                },
                "shirt_polygon": shirt_polygon.astype(int).tolist(),
                "target_color": self.target_color,
                "match_score": float(color_result.match_score),
                "mean_hsv": [float(x) for x in color_result.mean_hsv],
                "shirt_pixel_count": int(color_result.pixel_count),
                "is_target": bool(is_target),
            }

            candidates.append(candidate)

        return candidates

    def analyze(self, input_video_path: Path) -> dict:
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = OUTPUTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        output_video_path = run_dir / "analyzed_video.mp4"
        report_path = run_dir / "frame_report.json"
        features_path = run_dir / "frame_features.csv"
        config_path = run_dir / "config.json"

        cap = cv2.VideoCapture(str(input_video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_video_path}")

        input_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        output_fps = min(15.0, input_fps) if input_fps > 0 else 15.0

        max_frames = (
            int(self.config.max_seconds * input_fps)
            if self.config.max_seconds > 0
            else 10**9
        )

        writer = None
        frame_report = []
        csv_rows = []
        last_candidates: List[dict] = []

        frames_processed = 0
        heavy_frames = 0
        start_time = time.perf_counter()

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            if frames_processed >= max_frames:
                break

            frame, _ = resize_keep_aspect(frame, self.config.resize_width)
            height, width = frame.shape[:2]

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                output_width = width * 2 if self.config.enable_avatar else width
                output_height = height

                writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    output_fps,
                    (output_width, output_height),
                )

            should_process = (
                frames_processed % max(1, self.config.process_every_n_frames) == 0
            )

            if should_process:
                detections = self._detect_people(frame)

                pose_results = self.yolo_pose.estimate(
                    frame,
                    max_people=self._max_people_for_processing(),
                )

                last_candidates = self._analyze_candidates(
                    frame,
                    detections,
                    pose_results,
                )

                heavy_frames += 1

            annotated = frame.copy()
            draw_header(annotated, self.config.target_query, frames_processed)

            targets = [c for c in last_candidates if c["is_target"]]
            non_targets = [c for c in last_candidates if not c["is_target"]]

            if self.config.draw_non_targets:
                for candidate in non_targets:
                    draw_non_target_overlay(annotated, candidate)

            for candidate in targets:
                draw_target_overlay(annotated, candidate, self.target_color)

            if self.config.enable_avatar:
                sorted_targets = sorted(
                    targets,
                    key=lambda c: c.get("match_score", 0.0),
                    reverse=True,
                )

                sorted_targets = sorted_targets[: self._max_people_for_processing()]

                avatar_frame = draw_natural_avatar_frame(
                    frame_shape=frame.shape,
                    candidates=sorted_targets,
                    target_query=self.config.target_query,
                    frame_id=frames_processed,
                )

                annotated = np.hstack([annotated, avatar_frame])

            writer.write(annotated)

            timestamp_sec = frames_processed / input_fps if input_fps else 0.0

            frame_report.append(
                {
                    "frame_id": frames_processed,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "processed_with_models": should_process,
                    "num_candidates": len(last_candidates),
                    "num_targets": len(targets),
                    "candidates": last_candidates,
                }
            )

            for candidate in last_candidates:
                csv_rows.append(
                    _candidate_to_csv_row(
                        frames_processed,
                        timestamp_sec,
                        candidate,
                    )
                )

            frames_processed += 1

        cap.release()

        if writer is not None:
            writer.release()

        self.close()

        elapsed = time.perf_counter() - start_time
        fps_effective = frames_processed / elapsed if elapsed > 0 else 0.0

        summary = {
            "run_id": run_id,
            "target_query": self.config.target_query,
            "target_color": self.target_color,
            "frames_processed": frames_processed,
            "heavy_model_frames": heavy_frames,
            "elapsed_seconds": round(elapsed, 3),
            "effective_fps": round(fps_effective, 3),
            "output_video": str(output_video_path),
            "frame_report": str(report_path),
            "frame_features": str(features_path),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "frames": frame_report,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        fieldnames = [
            "frame_id",
            "timestamp_sec",
            "candidate_id",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "detector_confidence",
            "pose_detected",
            "target_color",
            "match_score",
            "is_target",
            "mean_h",
            "mean_s",
            "mean_v",
            "shirt_pixel_count",
        ]

        with open(features_path, "w", encoding="utf-8", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(csv_rows)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                self.config.__dict__,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return summary