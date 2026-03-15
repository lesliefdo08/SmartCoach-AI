"""Pose comparison and similarity scoring module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class PoseComparator:
    """Compares current pose features against a reference profile."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "left_elbow_angle": 1.3,
            "right_elbow_angle": 1.3,
            "left_knee_angle": 1.1,
            "right_knee_angle": 1.1,
            "left_shoulder_angle": 1.2,
            "right_shoulder_angle": 1.2,
            "hip_alignment": 0.9,
            "spine_tilt": 1.0,
        }
    )

    def compare(
        self,
        detected_features: Dict[str, float],
        reference_profile: Dict[str, object],
    ) -> Dict[str, object]:
        """Return similarity metrics and per-joint errors.

        Supports two reference formats:
        1) Legacy fixed-value format: {joint_name: mean_angle}
        2) Statistical format:
           {
             "joint_angles_mean": {...},
             "joint_angles_std": {...},
             "ideal_ranges": {...}
           }
        """
        reference_features, reference_std, ideal_ranges = normalize_reference_profile(reference_profile)

        common_keys = [
            key
            for key in self.weights
            if key in detected_features
            and key in reference_features
            and np.isfinite(detected_features[key])
            and np.isfinite(reference_features[key])
        ]

        if not common_keys:
            return {
                "similarity_score": 0.0,
                "cosine_similarity": 0.0,
                "euclidean_distance": float("inf"),
                "weighted_joint_deviation": 100.0,
                "joint_errors": [],
            }

        detected_vec = np.array([detected_features[k] for k in common_keys], dtype=np.float32)
        reference_vec = np.array([reference_features[k] for k in common_keys], dtype=np.float32)

        cos_sim = float(cosine_similarity([detected_vec], [reference_vec])[0][0])
        cos_score = np.clip(((cos_sim + 1.0) / 2.0) * 100.0, 0.0, 100.0)

        euc_dist = float(np.linalg.norm(detected_vec - reference_vec))
        normalized_euc = min(euc_dist / (np.sqrt(len(common_keys)) * 180.0), 1.0)
        euc_score = (1.0 - normalized_euc) * 100.0

        total_weight = sum(self.weights[k] for k in common_keys)

        weighted_z_outside = 0.0
        range_compliance_sum = 0.0
        for k in common_keys:
            tolerance = max(float(reference_std.get(k, 0.0)), 4.0)
            lower, upper = ideal_ranges.get(
                k,
                (reference_features[k] - 1.5 * tolerance, reference_features[k] + 1.5 * tolerance),
            )
            value = float(detected_features[k])

            if value < lower:
                outside = lower - value
            elif value > upper:
                outside = value - upper
            else:
                outside = 0.0

            z_outside = outside / tolerance
            weighted_z_outside += z_outside * self.weights[k]

            compliance = max(0.0, 1.0 - (z_outside / 3.0))
            range_compliance_sum += compliance * self.weights[k]

        weighted_dev = (weighted_z_outside / total_weight) * (100.0 / 3.0)
        weighted_dev = float(np.clip(weighted_dev, 0.0, 100.0))
        range_score = float(np.clip((range_compliance_sum / total_weight) * 100.0, 0.0, 100.0))

        similarity_score = float(
            np.clip(0.35 * cos_score + 0.25 * euc_score + 0.25 * (100.0 - weighted_dev) + 0.15 * range_score, 0.0, 100.0)
        )

        joint_errors: List[Dict[str, float]] = []
        for k in common_keys:
            err = float(detected_features[k] - reference_features[k])
            tolerance = max(float(reference_std.get(k, 0.0)), 4.0)
            lower, upper = ideal_ranges.get(
                k,
                (reference_features[k] - 1.5 * tolerance, reference_features[k] + 1.5 * tolerance),
            )

            value = float(detected_features[k])
            if value < lower:
                outside = lower - value
            elif value > upper:
                outside = value - upper
            else:
                outside = 0.0

            joint_errors.append(
                {
                    "joint": k,
                    "detected": value,
                    "reference": float(reference_features[k]),
                    "error": err,
                    "abs_error": abs(err),
                    "std_tolerance": round(tolerance, 3),
                    "ideal_range": [round(float(lower), 3), round(float(upper), 3)],
                    "outside_deviation": round(float(outside), 3),
                    "outside_zscore": round(float(outside / tolerance), 3),
                }
            )

        joint_errors.sort(key=lambda item: item.get("outside_deviation", item["abs_error"]), reverse=True)

        return {
            "similarity_score": round(similarity_score, 2),
            "cosine_similarity": round(float(cos_score), 2),
            "euclidean_distance": round(euc_dist, 2),
            "weighted_joint_deviation": round(weighted_dev, 2),
            "range_compliance_score": round(range_score, 2),
            "joint_errors": joint_errors,
        }


def normalize_reference_profile(
    reference_profile: Dict[str, object],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Normalize reference profile to mean/std/range maps.

    Returns:
        means, stds, ranges(lower, upper)
    """
    if "joint_angles_mean" in reference_profile:
        means = {
            str(k): float(v)
            for k, v in dict(reference_profile.get("joint_angles_mean", {})).items()
            if np.isfinite(v)
        }
        stds = {
            str(k): float(v)
            for k, v in dict(reference_profile.get("joint_angles_std", {})).items()
            if np.isfinite(v)
        }
        raw_ranges = dict(reference_profile.get("ideal_ranges", {}))
    elif "angles" in reference_profile:
        means = {
            str(k): float(v)
            for k, v in dict(reference_profile.get("angles", {})).items()
            if np.isfinite(v)
        }
        stds = {k: 8.0 for k in means}
        raw_ranges = {}
    else:
        means = {str(k): float(v) for k, v in reference_profile.items() if np.isfinite(v)}
        stds = {k: 8.0 for k in means}
        raw_ranges = {}

    ranges: Dict[str, Tuple[float, float]] = {}
    for k, m in means.items():
        if k in raw_ranges and isinstance(raw_ranges[k], (list, tuple)) and len(raw_ranges[k]) == 2:
            low = float(raw_ranges[k][0])
            high = float(raw_ranges[k][1])
            if low <= high:
                ranges[k] = (low, high)
                continue

        tol = max(float(stds.get(k, 8.0)), 4.0)
        ranges[k] = (max(0.0, m - 1.5 * tol), min(180.0, m + 1.5 * tol))

    return means, stds, ranges


def get_reference_means(reference_profile: Dict[str, object]) -> Dict[str, float]:
    """Return only the mean angles map from any reference profile format."""
    means, _, _ = normalize_reference_profile(reference_profile)
    return means
