"""3D pose estimation and biomechanics metrics using MediaPipe landmarks."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from utils.mediapipe_compat import get_pose_model


Point3D = Tuple[float, float, float]

_REQUIRED = (
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
)

_POSE_3D = None


def _get_pose_model():
    global _POSE_3D
    if _POSE_3D is None:
        _POSE_3D = get_pose_model(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
    return _POSE_3D


def estimate_3d_pose(frame: np.ndarray) -> Dict[str, Point3D]:
    """Estimate 3D body coordinates from a BGR or RGB frame.

    Returns selected landmarks in normalized camera coordinates [x, y, z].
    """
    if frame.ndim != 3:
        return {}

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
    result = _get_pose_model().detect_pose(rgb)

    if not result.keypoints_3d:
        return {}

    pose3d: Dict[str, Point3D] = {}
    for name in _REQUIRED:
        if name in result.keypoints_3d:
            pose3d[name] = result.keypoints_3d[name]

    return pose3d


def compute_biomechanical_metrics(pose3d: Dict[str, Point3D]) -> Dict[str, float | list[float]]:
    """Compute advanced biomechanics metrics from 3D coordinates."""
    if not pose3d:
        return {
            "shoulder_rotation": 0.0,
            "torso_twist": 0.0,
            "bat_swing_plane_angle": 0.0,
            "center_of_gravity_estimate": [0.0, 0.0, 0.0],
        }

    def vec(a: str, b: str) -> np.ndarray:
        return np.array(pose3d[b], dtype=np.float32) - np.array(pose3d[a], dtype=np.float32)

    shoulder_axis = vec("left_shoulder", "right_shoulder")
    hip_axis = vec("left_hip", "right_hip")

    shoulder_yaw = np.degrees(np.arctan2(shoulder_axis[2], shoulder_axis[0] + 1e-8))
    hip_yaw = np.degrees(np.arctan2(hip_axis[2], hip_axis[0] + 1e-8))

    shoulder_rotation = float(abs(shoulder_yaw))
    torso_twist = float(abs(shoulder_yaw - hip_yaw))

    left_arm = vec("left_shoulder", "left_wrist")
    right_arm = vec("right_shoulder", "right_wrist")
    swing_vec = (left_arm + right_arm) / 2.0
    bat_swing_plane_angle = float(np.degrees(np.arctan2(abs(swing_vec[1]), np.sqrt(swing_vec[0] ** 2 + swing_vec[2] ** 2) + 1e-8)))

    cog_points = np.array(
        [
            pose3d["left_shoulder"],
            pose3d["right_shoulder"],
            pose3d["left_hip"],
            pose3d["right_hip"],
            pose3d["left_knee"],
            pose3d["right_knee"],
        ],
        dtype=np.float32,
    )
    cog = cog_points.mean(axis=0)

    return {
        "shoulder_rotation": round(shoulder_rotation, 3),
        "torso_twist": round(torso_twist, 3),
        "bat_swing_plane_angle": round(bat_swing_plane_angle, 3),
        "center_of_gravity_estimate": [round(float(cog[0]), 4), round(float(cog[1]), 4), round(float(cog[2]), 4)],
    }
