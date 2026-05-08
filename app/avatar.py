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

    for y in range(height):
        alpha = y / max(height - 1, 1)
        base = int(26 + 20 * alpha)
        canvas[y, :] = (base + 6, base + 2, base)

    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (58, 58, 68), 2)
    cv2.putText(
        canvas,
        "Pose view",
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (230, 230, 235),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Waiting for target pose",
        (24, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (155, 155, 165),
        1,
        cv2.LINE_AA,
    )

    return canvas


def _to_point(value):
    if value is None:
        return None

    if len(value) < 2:
        return None

    return int(value[0]), int(value[1])


def _normalize_to_avatar(landmarks: dict, out_w: int, out_h: int) -> dict:
    points = {}

    for name, value in landmarks.items():
        point = _to_point(value)
        if point is not None:
            points[name] = point

    if not points:
        return {}

    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    body_w = max(max_x - min_x, 1)
    body_h = max(max_y - min_y, 1)

    scale_x = out_w * 0.62 / body_w
    scale_y = out_h * 0.72 / body_h
    scale = min(scale_x, scale_y)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    mapped = {}

    for name, point in points.items():
        x, y = point

        nx = int((x - center_x) * scale + out_w / 2)
        ny = int((y - center_y) * scale + out_h / 2 + 28)

        nx = int(np.clip(nx, 16, out_w - 16))
        ny = int(np.clip(ny, 56, out_h - 16))

        mapped[name] = (nx, ny)

    return mapped


def _distance(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.0

    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def _midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)


def _draw_limb(canvas, p1, p2, color, thickness: int):
    if p1 is None or p2 is None:
        return

    cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

    radius = max(4, thickness // 2)
    cv2.circle(canvas, p1, radius, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, p2, radius, color, -1, cv2.LINE_AA)


def _draw_torso(canvas, points: dict):
    ls = points.get("left_shoulder")
    rs = points.get("right_shoulder")
    lh = points.get("left_hip")
    rh = points.get("right_hip")

    if ls and rs and lh and rh:
        torso = np.array([ls, rs, rh, lh], dtype=np.int32)

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [torso], (74, 126, 235))
        cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, canvas)

        cv2.polylines(canvas, [torso], True, (220, 230, 255), 2, cv2.LINE_AA)
        return

    if ls and rs:
        center = _midpoint(ls, rs)
        shoulder_width = max(36, int(_distance(ls, rs)))
        cv2.ellipse(
            canvas,
            (center[0], center[1] + 48),
            (int(shoulder_width * 0.45), 58),
            0,
            0,
            360,
            (74, 126, 235),
            -1,
            cv2.LINE_AA,
        )


def _draw_head(canvas, points: dict):
    nose = points.get("nose")
    ls = points.get("left_shoulder")
    rs = points.get("right_shoulder")

    if nose is None and ls and rs:
        shoulder_center = _midpoint(ls, rs)
        shoulder_width = _distance(ls, rs)
        nose = (shoulder_center[0], int(shoulder_center[1] - shoulder_width * 0.75))

    if nose is None:
        return

    radius = 18

    if ls and rs:
        radius = int(np.clip(_distance(ls, rs) * 0.20, 14, 28))

    cv2.circle(canvas, nose, radius + 4, (40, 40, 48), -1, cv2.LINE_AA)
    cv2.circle(canvas, nose, radius, (236, 198, 158), -1, cv2.LINE_AA)
    cv2.circle(canvas, (nose[0] - radius // 3, nose[1] - radius // 5), 2, (30, 30, 35), -1)
    cv2.circle(canvas, (nose[0] + radius // 3, nose[1] - radius // 5), 2, (30, 30, 35), -1)

    cv2.ellipse(
        canvas,
        (nose[0], nose[1] + radius // 4),
        (radius // 3, radius // 5),
        0,
        0,
        180,
        (70, 70, 75),
        1,
        cv2.LINE_AA,
    )


def _draw_limbs(canvas, points: dict):
    arm_color = (98, 202, 245)
    leg_color = (118, 230, 152)
    joint_color = (245, 245, 245)

    arm_pairs = [
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ]

    leg_pairs = [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    for a, b in leg_pairs:
        _draw_limb(canvas, points.get(a), points.get(b), leg_color, 12)

    for a, b in arm_pairs:
        _draw_limb(canvas, points.get(a), points.get(b), arm_color, 11)

    for name in [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]:
        point = points.get(name)
        if point is not None:
            cv2.circle(canvas, point, 5, joint_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 7, (45, 45, 52), 1, cv2.LINE_AA)


def draw_avatar(landmarks: dict, height: int, width: int = 360) -> np.ndarray:
    canvas = blank_avatar(height=height, width=width)

    points = _normalize_to_avatar(landmarks, out_w=width, out_h=height)

    if not points:
        return canvas

    cv2.putText(
        canvas,
        "Target pose",
        (24, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (155, 220, 255),
        1,
        cv2.LINE_AA,
    )

    _draw_limbs(canvas, points)
    _draw_torso(canvas, points)
    _draw_head(canvas, points)

    return canvas


def combine_with_avatar(frame: np.ndarray, avatar: np.ndarray) -> np.ndarray:
    if avatar.shape[0] != frame.shape[0]:
        avatar = cv2.resize(
            avatar,
            (avatar.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    return np.hstack([frame, avatar])


def draw_multi_avatar(candidates: list, height: int, width: int = 360, max_avatars: int = 2) -> np.ndarray:
    canvas = blank_avatar(height=height, width=width)

    selected = candidates[:max_avatars]

    if not selected:
        return canvas

    slot_height = max(120, height // len(selected))
    panels = []

    for idx, candidate in enumerate(selected):
        landmarks = candidate.get("landmarks", {})
        panel = draw_avatar(
            landmarks=landmarks,
            height=slot_height,
            width=width,
        )

        label = f"Target {idx + 1} | score {candidate.get('match_score', 0.0):.2f}"

        cv2.putText(
            panel,
            label,
            (24, min(104, slot_height - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 235),
            1,
            cv2.LINE_AA,
        )

        panels.append(panel)

    combined = np.vstack(panels)

    if combined.shape[0] < height:
        pad = blank_avatar(height=height - combined.shape[0], width=width)
        combined = np.vstack([combined, pad])

    if combined.shape[0] > height:
        combined = cv2.resize(combined, (width, height), interpolation=cv2.INTER_AREA)

    return combined