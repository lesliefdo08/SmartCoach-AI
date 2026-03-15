"""Bat detection and trajectory tracking utilities."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


Point2D = Tuple[int, int]


def detect_bat(frame: np.ndarray, wrist_points: Sequence[Point2D] | None = None) -> Dict[str, object]:
    """Detect cricket bat candidate and tip point from frame.

    Uses contour shape filtering and optional wrist proximity.
    """
    if frame is None or frame.size == 0:
        return {"tip": None, "bbox": None, "confidence": 0.0}

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # broad wood-like and bright bat tape ranges
    mask1 = cv2.inRange(hsv, (5, 20, 60), (35, 220, 255))
    mask2 = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"tip": None, "bbox": None, "confidence": 0.0}

    h, w = frame.shape[:2]
    wrists = list(wrist_points) if wrist_points else []

    best = None
    best_score = -1.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 80:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect < 2.0:
            continue

        proximity_bonus = 0.0
        if wrists:
            cx = x + bw // 2
            cy = y + bh // 2
            d = min(np.hypot(cx - wx, cy - wy) for wx, wy in wrists)
            proximity_bonus = max(0.0, 1.0 - (d / (0.6 * np.hypot(w, h))))

        score = (area / (w * h)) * 6.0 + min(aspect / 6.0, 1.0) + proximity_bonus
        if score > best_score:
            best = c
            best_score = score

    if best is None:
        return {"tip": None, "bbox": None, "confidence": 0.0}

    x, y, bw, bh = cv2.boundingRect(best)
    pts = best.reshape(-1, 2)
    # farthest contour point from nearest wrist (or from bbox center)
    if wrists:
        wrist_anchor = np.array(wrists[np.argmin([np.linalg.norm(np.array(wrists[i]) - np.array([x + bw / 2, y + bh / 2])) for i in range(len(wrists))])], dtype=np.float32)
    else:
        wrist_anchor = np.array([x + bw / 2.0, y + bh / 2.0], dtype=np.float32)

    dists = np.linalg.norm(pts.astype(np.float32) - wrist_anchor, axis=1)
    tip = tuple(pts[int(np.argmax(dists))].tolist())

    confidence = float(np.clip(best_score, 0.0, 1.0))
    return {"tip": (int(tip[0]), int(tip[1])), "bbox": (int(x), int(y), int(bw), int(bh)), "confidence": confidence}


def track_bat_trajectory(
    frames: Sequence[np.ndarray],
    wrist_points_sequence: Sequence[Sequence[Point2D]] | None = None,
    smooth_window: int = 5,
) -> Dict[str, object]:
    """Track bat tip coordinates and estimate swing speed/arc."""
    path: List[Point2D] = []

    for i, frame in enumerate(frames):
        wrists = wrist_points_sequence[i] if wrist_points_sequence and i < len(wrist_points_sequence) else None
        det = detect_bat(frame, wrists)
        tip = det.get("tip")
        if tip is not None:
            path.append(tip)

    if not path:
        return {
            "bat_path_coordinates": [],
            "smoothed_bat_path": [],
            "swing_speed_per_frame": [],
            "swing_speed": 0.0,
            "swing_arc_angle": 0.0,
        }

    smoothed = _moving_average_path(path, window=max(1, smooth_window))
    speeds = [float(np.linalg.norm(np.array(smoothed[i]) - np.array(smoothed[i - 1]))) for i in range(1, len(smoothed))]
    swing_speed = float(np.mean(speeds)) if speeds else 0.0
    swing_arc = compute_swing_arc(smoothed)

    return {
        "bat_path_coordinates": path,
        "smoothed_bat_path": smoothed,
        "swing_speed_per_frame": speeds,
        "swing_speed": round(swing_speed, 3),
        "swing_arc_angle": round(swing_arc, 3),
    }


def compute_swing_arc(path: Sequence[Point2D]) -> float:
    """Estimate swing arc angle from tracked path."""
    if len(path) < 3:
        return 0.0

    pts = np.array(path, dtype=np.float32)
    start, mid, end = pts[0], pts[len(pts) // 2], pts[-1]

    v1 = start - mid
    v2 = end - mid
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _moving_average_path(path: Sequence[Point2D], window: int = 5) -> List[Point2D]:
    pts = np.array(path, dtype=np.float32)
    out: List[Point2D] = []
    for i in range(len(pts)):
        s = max(0, i - window + 1)
        m = pts[s : i + 1].mean(axis=0)
        out.append((int(m[0]), int(m[1])))
    return out
