from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


Point2D = Tuple[float, float]
Keypoint = Tuple[float, float, float]


def _point(keypoints: Dict[str, Keypoint], name: str) -> Point2D:
    p = keypoints[name]
    return float(p[0]), float(p[1])


def _vector_angle(a: Point2D, b: Point2D, c: Point2D) -> float:
    """Angle ABC using arccos((AB·CB)/(|AB||CB|))."""
    ab = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
    cb = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
    denom = float(np.linalg.norm(ab) * np.linalg.norm(cb))
    if denom < 1e-8:
        return 0.0
    cos_val = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def _spine_tilt_deg(nose: Point2D, hip_center: Point2D) -> float:
    dx = abs(float(nose[0] - hip_center[0]))
    dy = abs(float(nose[1] - hip_center[1])) + 1e-6
    return float(np.degrees(np.arctan2(dx, dy)))


def _head_position_vs_ball_line(nose: Point2D, ball_line_x: float) -> float:
    return float(abs(nose[0] - ball_line_x))


def compute_biomechanics_frame(keypoints: Dict[str, Keypoint], ball_line_x: float) -> Dict[str, float]:
    if not keypoints:
        return {
            "elbow_angle": 0.0,
            "knee_bend_angle": 0.0,
            "spine_tilt": 0.0,
            "head_position_relative_ball_line": 0.0,
        }

    left_elbow = _vector_angle(_point(keypoints, "left_shoulder"), _point(keypoints, "left_elbow"), _point(keypoints, "left_wrist"))
    right_elbow = _vector_angle(_point(keypoints, "right_shoulder"), _point(keypoints, "right_elbow"), _point(keypoints, "right_wrist"))
    elbow_angle = float((left_elbow + right_elbow) / 2.0)

    left_knee = _vector_angle(_point(keypoints, "left_hip"), _point(keypoints, "left_knee"), _point(keypoints, "left_ankle"))
    right_knee = _vector_angle(_point(keypoints, "right_hip"), _point(keypoints, "right_knee"), _point(keypoints, "right_ankle"))
    knee_bend = float((left_knee + right_knee) / 2.0)

    hip_center = (
        (_point(keypoints, "left_hip")[0] + _point(keypoints, "right_hip")[0]) / 2.0,
        (_point(keypoints, "left_hip")[1] + _point(keypoints, "right_hip")[1]) / 2.0,
    )
    nose = _point(keypoints, "nose")
    spine_tilt = _spine_tilt_deg(nose=nose, hip_center=hip_center)

    head_offset = _head_position_vs_ball_line(nose=nose, ball_line_x=ball_line_x)

    return {
        "elbow_angle": round(elbow_angle, 2),
        "knee_bend_angle": round(knee_bend, 2),
        "spine_tilt": round(spine_tilt, 2),
        "head_position_relative_ball_line": round(head_offset, 2),
    }


def _score_from_target(value: float, target: float, tolerance: float) -> float:
    delta = abs(value - target)
    return float(np.clip(100.0 - (delta / max(tolerance, 1e-6)) * 100.0, 0.0, 100.0))


def summarize_biomechanics(metrics_series: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_series:
        return {"technique_score": 0.0, "balance_score": 0.0, "consistency_score": 0.0}

    elbow_vals = np.array([m["elbow_angle"] for m in metrics_series], dtype=np.float32)
    knee_vals = np.array([m["knee_bend_angle"] for m in metrics_series], dtype=np.float32)
    spine_vals = np.array([m["spine_tilt"] for m in metrics_series], dtype=np.float32)
    head_vals = np.array([m["head_position_relative_ball_line"] for m in metrics_series], dtype=np.float32)

    technique = 0.45 * _score_from_target(float(np.mean(elbow_vals)), target=105.0, tolerance=45.0)
    technique += 0.35 * _score_from_target(float(np.mean(knee_vals)), target=145.0, tolerance=40.0)
    technique += 0.20 * _score_from_target(float(np.mean(spine_vals)), target=12.0, tolerance=15.0)

    balance = 0.6 * _score_from_target(float(np.mean(spine_vals)), target=10.0, tolerance=14.0)
    balance += 0.4 * _score_from_target(float(np.mean(head_vals)), target=35.0, tolerance=50.0)

    spread = float(np.mean([np.std(elbow_vals), np.std(knee_vals), np.std(spine_vals), np.std(head_vals)]))
    consistency = float(np.clip(100.0 - spread * 1.2, 0.0, 100.0))

    return {
        "technique_score": round(float(np.clip(technique, 0.0, 100.0)), 2),
        "balance_score": round(float(np.clip(balance, 0.0, 100.0)), 2),
        "consistency_score": round(consistency, 2),
    }
