from __future__ import annotations

from typing import Dict, List

import numpy as np


FRAME_FEATURES = [
    "bat_swing_arc",
    "bat_angle",
    "follow_through_height",
    "body_rotation",
    "knee_bend",
    "head_position",
    "bat_velocity",
    "ball_direction",
]


def merge_features(object_features: Dict[str, float], pose_features: Dict[str, float]) -> Dict[str, float]:
    merged = {
        "bat_swing_arc": float(object_features.get("bat_swing_arc", 0.0)),
        "bat_angle": float(object_features.get("bat_angle", 0.0)),
        "follow_through_height": float(pose_features.get("follow_through_height", 0.0)),
        "body_rotation": float(pose_features.get("body_rotation", 0.0)),
        "knee_bend": float(pose_features.get("knee_bend", 0.0)),
        "head_position": float(pose_features.get("head_position", 0.0)),
        "bat_velocity": float(object_features.get("bat_velocity", 0.0)),
        "ball_direction": float(object_features.get("ball_direction", 0.0)),
    }
    return merged


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
