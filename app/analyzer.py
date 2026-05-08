import csv
import json
import time
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.avatar import blank_avatar, combine_with_avatar, draw_avatar
from app.color_utils import evaluate_color_in_polygon, parse_target_color
from app.config import AnalyzerConfig, OUTPUTS_DIR
from app.detector import PersonDetector, full_frame_detection
from app.geometry import resize_keep_aspect
from app.overlay import draw_header, draw_non_target_overlay, draw_target_overlay
from app.yolo_pose import YoloPoseEstimator, find_best_pose_for_detection


DEFAULT_AVATAR_PERSON_POSITION = "right"


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


def _candidate_center_x(candidate: dict) -> float:
    x1, _, x2, _ = candidate["bbox"]
    return (float(x1) + float(x2)) / 2.0


def _bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter_area / float(area_a + area_b - inter_area + 1e-6)


def _select_avatar_candidate_by_rank(
    candidates: List[dict],
    position: str,
    rank: int,
) -> Optional[dict]:
    if not candidates:
        return None

    position = (position or "right").lower()
    rank = max(1, int(rank))

    reverse = position != "left"

    ordered = sorted(
        candidates,
        key=_candidate_center_x,
        reverse=reverse,
    )

    index = min(rank - 1, len(ordered) - 1)

    return ordered[index]


def _select_closest_to_previous_bbox(
    candidates: List[dict],
    previous_bbox: Optional[List[int]],
) -> Optional[dict]:
    if not candidates or previous_bbox is None:
        return None

    prev_cx = (previous_bbox[0] + previous_bbox[2]) / 2.0

    def score(candidate: dict) -> float:
        cx = _candidate_center_x(candidate)
        iou = _bbox_iou(candidate["bbox"], previous_bbox)
        distance = abs(cx - prev_cx)

        return iou * 1000.0 - distance

    return max(candidates, key=score)


def _select_avatar_candidate(
    candidates: List[dict],
    mode: str = DEFAULT_AVATAR_PERSON_POSITION,
) -> Optional[dict]:
    if not candidates:
        return None

    mode = (mode or DEFAULT_AVATAR_PERSON_POSITION).lower()

    if mode == "right":
        return max(candidates, key=_candidate_center_x)

    if mode == "left":
        return min(candidates, key=_candidate_center_x)

    if mode == "best":
        return max(candidates, key=lambda c: c.get("match_score", 0.0))

    return max(candidates, key=_candidate_center_x)


class VideoAnalyzer:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.target_color = parse_target_color(config.target_query)
        self.avatar_person_position = getattr(
            config,
            "avatar_person_position",
            DEFAULT_AVATAR_PERSON_POSITION,
        )

        self.avatar_person_rank_from_right = max(
            1,
            int(getattr(config, "avatar_person_rank_from_right", 1)),
        )

        self.selected_avatar_bbox: Optional[List[int]] = None

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

    def _detect_people(self, frame: np.ndarray):
        if self.config.use_yolo:
            return self.detector.detect(frame, max_people=self.config.max_people)

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

                output_width = width
                if self.config.enable_avatar:
                    output_width = width + 360

                writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    output_fps,
                    (output_width, height),
                )

            should_process = (
                frames_processed % max(1, self.config.process_every_n_frames) == 0
            )

            if should_process:
                detections = self._detect_people(frame)

                pose_results = self.yolo_pose.estimate(
                    frame,
                    max_people=self.config.max_people,
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
                target_candidate: Optional[dict] = None

                if targets:
                    if self.selected_avatar_bbox is None:
                        target_candidate = _select_avatar_candidate_by_rank(
                            candidates=targets,
                            position=self.avatar_person_position,
                            rank=self.avatar_person_rank_from_right,
                        )
                    else:
                        target_candidate = _select_closest_to_previous_bbox(
                            candidates=targets,
                            previous_bbox=self.selected_avatar_bbox,
                        )

                    if target_candidate is not None:
                        self.selected_avatar_bbox = [
                            int(v) for v in target_candidate["bbox"]
                        ]

                if target_candidate is not None:
                    avatar = draw_avatar(
                        target_candidate.get("landmarks", {}),
                        height=annotated.shape[0],
                        width=360,
                    )
                else:
                    avatar = blank_avatar(
                        height=annotated.shape[0],
                        width=360,
                    )

                annotated = combine_with_avatar(annotated, avatar)

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