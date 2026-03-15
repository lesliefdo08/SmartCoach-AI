"""Pose feature extraction for cricket posture analysis."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from utils.angle_utils import angle_between_three_points, line_angle_degrees, midpoint


Keypoint = Tuple[float, float, float]


def _xy(keypoints: Dict[str, Keypoint], name: str) -> Tuple[float, float]:
    point = keypoints[name]
    return float(point[0]), float(point[1])


def extract_pose_features(keypoints: Dict[str, Keypoint]) -> Dict[str, float]:
    """Extract joint and alignment features from detected keypoints."""
    if not keypoints:
        return {}

    # Elbow angles
    left_elbow_angle = angle_between_three_points(
        _xy(keypoints, "left_shoulder"),
        _xy(keypoints, "left_elbow"),
        _xy(keypoints, "left_wrist"),
    )
    right_elbow_angle = angle_between_three_points(
        _xy(keypoints, "right_shoulder"),
        _xy(keypoints, "right_elbow"),
        _xy(keypoints, "right_wrist"),
    )

    # Knee angles
    left_knee_angle = angle_between_three_points(
        _xy(keypoints, "left_hip"),
        _xy(keypoints, "left_knee"),
        _xy(keypoints, "left_ankle"),
    )
    right_knee_angle = angle_between_three_points(
        _xy(keypoints, "right_hip"),
        _xy(keypoints, "right_knee"),
        _xy(keypoints, "right_ankle"),
    )

    # Shoulder angles (arm relative to torso)
    left_shoulder_angle = angle_between_three_points(
        _xy(keypoints, "left_elbow"),
        _xy(keypoints, "left_shoulder"),
        _xy(keypoints, "left_hip"),
    )
    right_shoulder_angle = angle_between_three_points(
        _xy(keypoints, "right_elbow"),
        _xy(keypoints, "right_shoulder"),
        _xy(keypoints, "right_hip"),
    )

    # Hip alignment (tilt to horizontal)
    hip_alignment = line_angle_degrees(
        _xy(keypoints, "left_hip"),
        _xy(keypoints, "right_hip"),
    )

    # Spine tilt: line from mid-hip to nose compared against vertical
    mid_hip = midpoint(_xy(keypoints, "left_hip"), _xy(keypoints, "right_hip"))
    nose = _xy(keypoints, "nose")

    dx = nose[0] - mid_hip[0]
    dy = nose[1] - mid_hip[1]
    spine_tilt = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-8)))

    # Derived semantic indicators
    head_position = spine_tilt
    shoulder_rotation = abs(left_shoulder_angle - right_shoulder_angle)

    return {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_shoulder_angle": left_shoulder_angle,
        "right_shoulder_angle": right_shoulder_angle,
        "hip_alignment": hip_alignment,
        "spine_tilt": spine_tilt,
        "head_position": head_position,
        "shoulder_rotation": shoulder_rotation,
    }
