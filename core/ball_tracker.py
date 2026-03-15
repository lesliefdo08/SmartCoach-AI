"""Ball trajectory and impact alignment estimation."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


Point2D = Tuple[int, int]


def detect_ball(frame: np.ndarray) -> Dict[str, object]:
    """Detect cricket ball candidate using color and circular contour constraints."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # generic red ball + bright tennis ball support
    red1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    yellow = cv2.inRange(hsv, (20, 80, 80), (40, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), yellow)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    best_score = -1.0
    best_radius = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 8 or area > 900:
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius < 2 or radius > 20:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue

        score = circularity * (1.2 - abs(radius - 6.0) / 20.0)
        if score > best_score:
            best_score = score
            best_center = (int(x), int(y))
            best_radius = float(radius)

    if best_center is None:
        return {"center": None, "radius": 0.0, "confidence": 0.0}

    return {"center": best_center, "radius": best_radius, "confidence": float(np.clip(best_score, 0.0, 1.0))}


def track_ball(frames: Sequence[np.ndarray], bat_path: Sequence[Point2D] | None = None) -> Dict[str, object]:
    """Track ball trajectory and estimate bat-ball impact alignment."""
    trajectory: List[Point2D] = []

    for frame in frames:
        det = detect_ball(frame)
        c = det.get("center")
        if c is not None:
            trajectory.append(c)

    if not trajectory:
        return {
            "ball_trajectory": [],
            "impact_point_estimate": None,
            "bat_ball_alignment_score": 0.0,
        }

    impact = trajectory[-1]
    score = 0.0

    if bat_path:
        bat_arr = np.array(bat_path, dtype=np.float32)
        ball_arr = np.array(trajectory, dtype=np.float32)
        dists = np.linalg.norm(ball_arr[:, None, :] - bat_arr[None, :, :], axis=2)
        min_dist = float(np.min(dists)) if dists.size else 999.0
        score = float(np.clip(100.0 - (min_dist / 200.0) * 100.0, 0.0, 100.0))
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        impact = tuple(ball_arr[min_idx[0]].astype(int).tolist())

    return {
        "ball_trajectory": trajectory,
        "impact_point_estimate": impact,
        "bat_ball_alignment_score": round(score, 2),
    }
