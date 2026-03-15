"""Visualization helpers for pose overlays and annotations."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.pose_detector import get_default_connections


Keypoint = Tuple[float, float, float]

ANGLE_TO_JOINT = {
    "left_elbow_angle": "left_elbow",
    "right_elbow_angle": "right_elbow",
    "left_knee_angle": "left_knee",
    "right_knee_angle": "right_knee",
    "left_shoulder_angle": "left_shoulder",
    "right_shoulder_angle": "right_shoulder",
    "hip_alignment": "left_hip",
    "spine_tilt": "nose",
}


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    keypoints: Dict[str, Keypoint],
    joint_errors: List[Dict[str, float]] | None = None,
    angles: Dict[str, float] | None = None,
    error_threshold: float = 12.0,
) -> np.ndarray:
    """Draw skeleton, joints, and angle labels on frame."""
    output = frame_bgr.copy()
    if not keypoints:
        return output

    bad_joints = set()
    if joint_errors:
        for item in joint_errors:
            if item.get("abs_error", 0.0) >= error_threshold:
                mapped = ANGLE_TO_JOINT.get(item.get("joint", ""))
                if mapped:
                    bad_joints.add(mapped)

    # Draw limbs first
    for a, b in get_default_connections():
        if a in keypoints and b in keypoints:
            pa = (int(keypoints[a][0]), int(keypoints[a][1]))
            pb = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(output, pa, pb, (120, 255, 120), 2)

    # Draw joints
    for name, (x, y, vis) in keypoints.items():
        if vis < 0.2:
            continue
        color = (0, 0, 255) if name in bad_joints else (0, 220, 0)
        cv2.circle(output, (int(x), int(y)), 5, color, -1)

    # Annotate selected angles
    if angles:
        for angle_name, value in angles.items():
            if angle_name not in ANGLE_TO_JOINT or not np.isfinite(value):
                continue
            joint_name = ANGLE_TO_JOINT[angle_name]
            if joint_name not in keypoints:
                continue
            x, y, _ = keypoints[joint_name]
            cv2.putText(
                output,
                f"{angle_name.replace('_angle', '')}: {value:.1f}",
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return output


def draw_bat_trajectory(
    frame_bgr: np.ndarray,
    bat_path_coordinates: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Draw smoothed bat trajectory curve."""
    output = frame_bgr.copy()
    if len(bat_path_coordinates) < 2:
        return output

    pts = np.array(bat_path_coordinates, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(output, [pts], isClosed=False, color=color, thickness=2)
    cv2.circle(output, tuple(pts[-1][0]), 5, color, -1)
    return output


def draw_ball_trajectory(
    frame_bgr: np.ndarray,
    ball_path_coordinates: List[Tuple[int, int]],
    impact_point: Tuple[int, int] | None = None,
    color: Tuple[int, int, int] = (0, 165, 255),
) -> np.ndarray:
    """Draw ball trajectory and impact zone marker."""
    output = frame_bgr.copy()
    if len(ball_path_coordinates) >= 2:
        pts = np.array(ball_path_coordinates, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [pts], isClosed=False, color=color, thickness=2)
        cv2.circle(output, tuple(pts[-1][0]), 4, color, -1)

    if impact_point is not None:
        cv2.circle(output, impact_point, 12, (0, 0, 255), 2)
        cv2.putText(
            output,
            "Impact Zone",
            (impact_point[0] + 8, impact_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_3d_skeleton_projection(
    frame_bgr: np.ndarray,
    pose3d: Dict[str, Tuple[float, float, float]],
    color: Tuple[int, int, int] = (255, 120, 0),
) -> np.ndarray:
    """Project normalized 3D landmarks to 2D and draw a compact skeleton."""
    output = frame_bgr.copy()
    if not pose3d:
        return output

    h, w = output.shape[:2]

    def to_px(name: str) -> Tuple[int, int] | None:
        if name not in pose3d:
            return None
        x, y, _ = pose3d[name]
        return int(x * w), int(y * h)

    links = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
    ]

    for a, b in links:
        pa, pb = to_px(a), to_px(b)
        if pa is None or pb is None:
            continue
        cv2.line(output, pa, pb, color, 2)

    for name in pose3d:
        p = to_px(name)
        if p is not None:
            cv2.circle(output, p, 4, (255, 255, 255), -1)

    cv2.putText(output, "3D Projection", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output
