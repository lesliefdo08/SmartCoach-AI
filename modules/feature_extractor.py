from __future__ import annotations

from typing import Dict, List

import numpy as np


FRAME_FEATURES = [
    "bat_swing_arc",
    "bat_angle",
    "bat_angle_impact",
    "bat_follow_through_height",
    "follow_through_height",
    "follow_through_height_max",
    "swing_direction_score",
    "swing_trend",
    "wrist_position_impact_x",
    "wrist_position_impact_y",
    "player_body_lean",
    "shoulder_rotation",
    "elbow_angle",
    "body_lean",
    "wrist_trajectory",
    "knee_bend",
    "torso_tilt",
    "head_position",
    "bat_velocity",
    "ball_direction",
    "pose_visibility",
    "pose_confidence",
    "bat_confidence",
    "motion_phase",
]


def merge_features(object_features: Dict[str, float], pose_features: Dict[str, float]) -> Dict[str, float]:
    follow_through_height = float(pose_features.get("follow_through_height", 0.0))
    follow_through_height_max = float(
        pose_features.get(
            "follow_through_height_max",
            max(
                follow_through_height,
                float(object_features.get("bat_follow_through_height", 0.0)),
            ),
        )
    )
    body_lean = float(pose_features.get("body_lean", pose_features.get("torso_tilt", 0.0)))

    merged = {
        "bat_swing_arc": float(object_features.get("bat_swing_arc", 0.0)),
        "bat_angle": float(object_features.get("bat_angle", 0.0)),
        "bat_angle_impact": float(object_features.get("bat_angle_impact", object_features.get("bat_angle", 0.0))),
        "bat_follow_through_height": float(object_features.get("bat_follow_through_height", 0.0)),
        "follow_through_height": follow_through_height,
        "follow_through_height_max": follow_through_height_max,
        "swing_direction_score": float(object_features.get("swing_direction_score", 0.0)),
        "swing_trend": float(object_features.get("swing_trend", 0.0)),
        "wrist_position_impact_x": float(pose_features.get("wrist_position_impact_x", 0.0)),
        "wrist_position_impact_y": float(pose_features.get("wrist_position_impact_y", 0.0)),
        "player_body_lean": body_lean,
        "shoulder_rotation": float(pose_features.get("shoulder_rotation", pose_features.get("body_rotation", 0.0))),
        "elbow_angle": float(pose_features.get("elbow_angle", 0.0)),
        "body_lean": body_lean,
        "wrist_trajectory": float(pose_features.get("wrist_trajectory", 0.0)),
        "knee_bend": float(pose_features.get("knee_bend", 0.0)),
        "torso_tilt": float(pose_features.get("torso_tilt", 0.0)),
        "head_position": float(pose_features.get("head_position", 0.0)),
        "bat_velocity": float(object_features.get("bat_velocity", 0.0)),
        "ball_direction": float(object_features.get("ball_direction", 0.0)),
        "pose_visibility": float(pose_features.get("pose_visibility", 0.0)),
        "pose_confidence": float(pose_features.get("pose_confidence", pose_features.get("pose_visibility", 0.0))),
        "bat_confidence": float(object_features.get("bat_confidence", 0.0)),
        "motion_phase": float(pose_features.get("motion_phase", 0.0)),
    }
    return merged


def fill_missing_features(features: Dict[str, float]) -> Dict[str, float]:
    out = dict(features) if isinstance(features, dict) else {}
    for key in FRAME_FEATURES:
        out[key] = float(out.get(key, 0.0))
    return out


def sliding_window_average(features_by_frame: List[Dict[str, float]], window_size: int = 15) -> List[Dict[str, float]]:
    if not features_by_frame:
        return []

    win = int(np.clip(window_size, 10, 20))
    if len(features_by_frame) < win:
        return [
            {
                key: float(np.mean([f.get(key, 0.0) for f in features_by_frame]))
                for key in FRAME_FEATURES
            }
        ]

    out: List[Dict[str, float]] = []
    for i in range(0, len(features_by_frame) - win + 1):
        chunk = features_by_frame[i : i + win]
        out.append({key: float(np.mean([f.get(key, 0.0) for f in chunk])) for key in FRAME_FEATURES})
    return out
