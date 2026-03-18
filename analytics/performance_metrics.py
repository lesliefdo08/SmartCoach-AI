"""Performance analytics for SmartCoach AI."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _default_metrics(message: str = "Insufficient data for analysis") -> Dict[str, object]:
    return {
        "status": "insufficient_data",
        "message": message,
        "posture_accuracy_score": 0.0,
        "joint_stability": 0.0,
        "consistency_across_frames": 0.0,
        "swing_efficiency": 0.0,
        "bat_plane_consistency": 0.0,
        "torso_rotation_power": 0.0,
        "impact_alignment_score": 0.0,
        "advanced_performance_score": 0.0,
        "frames_analyzed": 0,
        "per_joint_variability": {},
    }


def compute_performance_metrics(
    frame_results: List[Dict[str, object]],
    biomechanics_data: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Compute posture accuracy, joint stability, and frame consistency."""
    if not isinstance(frame_results, list) or not frame_results:
        return _default_metrics("Insufficient data for analysis")

    valid_results = [fr for fr in frame_results if isinstance(fr, dict)]
    if not valid_results:
        return _default_metrics("Insufficient data for analysis")

    similarities = np.array([float(item.get("similarity_score", 0.0)) for item in valid_results], dtype=np.float32)
    similarities = similarities[np.isfinite(similarities)]
    if len(similarities) == 0:
        similarities = np.array([0.0], dtype=np.float32)
    posture_accuracy = float(np.clip(np.mean(similarities), 0.0, 100.0))

    feature_keys = set()
    for item in valid_results:
        features = item.get("features", {})
        if isinstance(features, dict):
            feature_keys.update(features.keys())

    per_joint_std: Dict[str, float] = {}
    for key in sorted(feature_keys):
        values = []
        for item in valid_results:
            features = item.get("features", {})
            if isinstance(features, dict):
                values.append(features.get(key, np.nan))
        arr = np.array(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 1:
            per_joint_std[key] = float(np.std(arr))

    if per_joint_std:
        mean_std = float(np.mean(list(per_joint_std.values())))
        joint_stability = float(np.clip(100.0 - (mean_std / 45.0) * 100.0, 0.0, 100.0))
    else:
        joint_stability = 0.0

    sim_std = float(np.std(similarities)) if len(similarities) > 1 else 0.0
    consistency = float(np.clip(100.0 - sim_std * 2.0, 0.0, 100.0))

    advanced = _compute_advanced_metrics(biomechanics_data or {})

    return {
        "status": "ok",
        "message": "Metrics computed",
        "posture_accuracy_score": round(posture_accuracy, 2),
        "joint_stability": round(joint_stability, 2),
        "consistency_across_frames": round(consistency, 2),
        "swing_efficiency": advanced["swing_efficiency"],
        "bat_plane_consistency": advanced["bat_plane_consistency"],
        "torso_rotation_power": advanced["torso_rotation_power"],
        "impact_alignment_score": advanced["impact_alignment_score"],
        "advanced_performance_score": advanced["advanced_performance_score"],
        "frames_analyzed": len(valid_results),
        "per_joint_variability": {k: round(v, 2) for k, v in per_joint_std.items()},
    }


def _compute_advanced_metrics(biomechanics_data: Dict[str, object]) -> Dict[str, float]:
    bat = biomechanics_data.get("bat", {}) if isinstance(biomechanics_data, dict) else {}
    ball = biomechanics_data.get("ball", {}) if isinstance(biomechanics_data, dict) else {}
    bio3d = biomechanics_data.get("pose3d", {}) if isinstance(biomechanics_data, dict) else {}

    swing_speed = float(bat.get("swing_speed", 0.0))
    arc_angle = float(bat.get("swing_arc_angle", 0.0))
    plane_series = np.array(bio3d.get("bat_swing_plane_series", []) if isinstance(bio3d, dict) else [], dtype=np.float32)
    torso_series = np.array(bio3d.get("torso_twist_series", []) if isinstance(bio3d, dict) else [], dtype=np.float32)
    plane_series = plane_series[np.isfinite(plane_series)]
    torso_series = torso_series[np.isfinite(torso_series)]
    impact_alignment = float(ball.get("bat_ball_alignment_score", 0.0))

    # Heuristic scoring transforms to 0-100
    swing_speed_score = float(np.clip((swing_speed / 18.0) * 100.0, 0.0, 100.0))
    arc_score = float(np.clip(100.0 - abs(arc_angle - 95.0), 0.0, 100.0))
    swing_efficiency = round(0.6 * swing_speed_score + 0.4 * arc_score, 2)

    if len(plane_series) > 1:
        plane_std = float(np.std(plane_series))
        bat_plane_consistency = float(np.clip(100.0 - (plane_std / 20.0) * 100.0, 0.0, 100.0))
    else:
        bat_plane_consistency = 0.0

    if len(torso_series) > 0:
        mean_torso = float(np.mean(np.abs(torso_series)))
        torso_rotation_power = float(np.clip((mean_torso / 45.0) * 100.0, 0.0, 100.0))
    else:
        torso_rotation_power = 0.0

    impact_alignment_score = float(np.clip(impact_alignment, 0.0, 100.0))

    advanced_overall = float(
        np.clip(
            0.3 * swing_efficiency
            + 0.25 * bat_plane_consistency
            + 0.2 * torso_rotation_power
            + 0.25 * impact_alignment_score,
            0.0,
            100.0,
        )
    )

    return {
        "swing_efficiency": round(swing_efficiency, 2),
        "bat_plane_consistency": round(bat_plane_consistency, 2),
        "torso_rotation_power": round(torso_rotation_power, 2),
        "impact_alignment_score": round(impact_alignment_score, 2),
        "advanced_performance_score": round(advanced_overall, 2),
    }
