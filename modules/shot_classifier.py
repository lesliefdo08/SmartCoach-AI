from __future__ import annotations

from pathlib import Path
from collections import Counter
import re
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


LABELS = ["defensive", "drive", "lofted", "pull", "cut", "sweep"]
BASE_FEATURES = [
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
    "motion_phase",
    "frame_weight",
]

FEATURE_ORDER = BASE_FEATURES + [
    "elbow_angle_mean",
    "elbow_angle_max",
    "elbow_angle_min",
    "elbow_angle_velocity",
    "elbow_angle_peak",
    "elbow_angle_trend",
    "shoulder_rotation_mean",
    "shoulder_rotation_max",
    "shoulder_rotation_min",
    "shoulder_rotation_velocity",
    "shoulder_rotation_peak",
    "shoulder_rotation_trend",
    "wrist_trajectory_mean",
    "wrist_trajectory_max",
    "wrist_trajectory_min",
    "wrist_trajectory_velocity",
    "wrist_trajectory_peak",
    "wrist_trajectory_trend",
    "body_lean_mean",
    "body_lean_max",
    "body_lean_min",
    "body_lean_velocity",
    "body_lean_peak",
    "body_lean_trend",
    "bat_velocity_peak",
    "bat_velocity_trend",
    "bat_angle_impact_mean",
]

NORMALIZE_FEATURES = [
    "bat_angle_impact",
    "follow_through_height_max",
    "swing_direction_score",
    "player_body_lean",
]


def _window_average(series: Sequence[Dict[str, float]], window_size: int = 15) -> List[Dict[str, float]]:
    if not series:
        return []
    win = int(np.clip(window_size, 10, 20))
    if len(series) < win:
        return [
            {
                key: float(np.mean([s.get(key, 0.0) for s in series]))
                for key in FEATURE_ORDER
            }
        ]
    out: List[Dict[str, float]] = []
    temporal_keys = ["elbow_angle", "shoulder_rotation", "wrist_trajectory", "body_lean"]
    for i in range(0, len(series) - win + 1):
        chunk = series[i : i + win]
        weights = np.array([float(np.clip(s.get("frame_weight", 1.0), 0.25, 4.0)) for s in chunk], dtype=np.float32)
        if float(np.sum(weights)) <= 1e-6:
            weights = np.ones(len(chunk), dtype=np.float32)

        agg = {}
        for key in BASE_FEATURES:
            arr = np.array([float(s.get(key, 0.0)) for s in chunk], dtype=np.float32)
            agg[key] = float(np.average(arr, weights=weights)) if len(arr) else 0.0

        for key in temporal_keys:
            arr = np.array([float(s.get(key, 0.0)) for s in chunk], dtype=np.float32)
            agg[f"{key}_mean"] = float(np.mean(arr)) if len(arr) else 0.0
            agg[f"{key}_max"] = float(np.max(arr)) if len(arr) else 0.0
            agg[f"{key}_min"] = float(np.min(arr)) if len(arr) else 0.0
            agg[f"{key}_velocity"] = float(np.mean(np.abs(np.diff(arr)))) if len(arr) >= 2 else 0.0
            agg[f"{key}_peak"] = float(np.max(np.abs(np.diff(arr)))) if len(arr) >= 2 else 0.0
            if len(arr) >= 2:
                x = np.arange(len(arr), dtype=np.float32)
                coeff = np.polyfit(x, arr, deg=1)
                agg[f"{key}_trend"] = float(coeff[0])
            else:
                agg[f"{key}_trend"] = 0.0

        bat_vel = np.array([float(s.get("bat_velocity", 0.0)) for s in chunk], dtype=np.float32)
        agg["bat_velocity_peak"] = float(np.max(bat_vel)) if len(bat_vel) else 0.0
        if len(bat_vel) >= 2:
            xv = np.arange(len(bat_vel), dtype=np.float32)
            coeff_v = np.polyfit(xv, bat_vel, deg=1)
            agg["bat_velocity_trend"] = float(coeff_v[0])
        else:
            agg["bat_velocity_trend"] = 0.0

        impact_angles = np.array([float(s.get("bat_angle_impact", s.get("bat_angle", 0.0))) for s in chunk], dtype=np.float32)
        agg["bat_angle_impact_mean"] = float(np.mean(impact_angles)) if len(impact_angles) else 0.0
        out.append(agg)
    return out


def _vectorize(features: Dict[str, float], feature_order: Sequence[str]) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in feature_order], dtype=np.float32)


def _normalize_indices(feature_order: Sequence[str]) -> List[int]:
    return [i for i, name in enumerate(feature_order) if name in NORMALIZE_FEATURES]


def _fit_selected_scaler(x: np.ndarray, feature_order: Sequence[str]) -> tuple[StandardScaler | None, List[int]]:
    idx = _normalize_indices(feature_order)
    if not idx:
        return None, []
    scaler = StandardScaler()
    scaler.fit(x[:, idx])
    return scaler, idx


def _transform_selected(x: np.ndarray, scaler: StandardScaler | None, idx: Sequence[int]) -> np.ndarray:
    if scaler is None or not idx:
        return x
    x_out = np.array(x, dtype=np.float32, copy=True)
    x_out[:, list(idx)] = scaler.transform(x_out[:, list(idx)])
    return x_out


def _apply_feature_weights(x: np.ndarray, feature_weights: Dict[str, float] | None, feature_order: Sequence[str]) -> np.ndarray:
    if not feature_weights:
        return x
    x_out = np.array(x, dtype=np.float32, copy=True)
    for i, name in enumerate(feature_order):
        w = float(feature_weights.get(name, 1.0))
        if abs(w - 1.0) > 1e-9:
            x_out[:, i] = x_out[:, i] * w
    return x_out


def _renormalize(prob_map: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(prob_map.values()))
    if total <= 1e-9:
        return prob_map
    return {k: float(v / total) for k, v in prob_map.items()}


def _extract_summary(feature_series: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not feature_series:
        return {}
    arr = lambda key: np.array([float(f.get(key, 0.0)) for f in feature_series], dtype=np.float32)
    summary = {
        "follow_through_height_max": float(np.max(arr("follow_through_height_max"))),
        "bat_angle_impact": float(np.mean(arr("bat_angle_impact"))),
        "player_body_lean": float(np.mean(arr("player_body_lean"))),
        "swing_direction_score": float(np.mean(arr("swing_direction_score"))),
        "shoulder_rotation": float(np.mean(arr("shoulder_rotation"))),
        "bat_velocity_peak": float(np.max(arr("bat_velocity"))),
    }
    return summary


def _adaptive_boost_multiplier(model_confidence: float, calibration: Dict[str, object]) -> float:
    low_gate = float(calibration.get("heuristic_low_conf_gate", 0.70))
    high_gate = float(calibration.get("heuristic_high_conf_gate", 0.85))
    low_factor = float(calibration.get("heuristic_low_conf_multiplier", 1.35))
    high_factor = float(calibration.get("heuristic_high_conf_multiplier", 0.65))
    if model_confidence < low_gate:
        return low_factor
    if model_confidence > high_gate:
        return high_factor
    return 1.0


def _apply_soft_heuristics(
    prob_map: Dict[str, float],
    summary: Dict[str, float],
    model_confidence: float,
    calibration: Dict[str, object],
) -> Tuple[Dict[str, float], List[str]]:
    adjusted = dict(prob_map)
    reasons: List[str] = []
    mult = _adaptive_boost_multiplier(model_confidence, calibration)

    follow_h = float(summary.get("follow_through_height_max", 0.0))
    bat_ang = float(summary.get("bat_angle_impact", 0.0))
    body_lean = float(summary.get("player_body_lean", 0.0))
    swing_dir = float(summary.get("swing_direction_score", 0.0))
    shoulder_rot = float(summary.get("shoulder_rotation", 0.0))

    if follow_h > 0.18 and "lofted" in adjusted:
        adjusted["lofted"] += 0.06 * mult
        reasons.append("Boost lofted: high follow-through height")

    if bat_ang < 22.0 and body_lean > 10.0 and "defensive" in adjusted:
        adjusted["defensive"] += 0.05 * mult
        reasons.append("Boost defensive: low bat angle and forward lean")

    if abs(swing_dir) < 0.22 and shoulder_rot > 18.0 and "pull" in adjusted:
        adjusted["pull"] += 0.05 * mult
        reasons.append("Boost pull: horizontal swing with torso rotation")

    if reasons:
        reasons.append(f"Adaptive heuristic multiplier: {mult:.2f}")

    return _renormalize(adjusted), reasons


def _apply_tie_breaker(
    prob_map: Dict[str, float],
    summary: Dict[str, float],
    tie_break_threshold: float,
    model_confidence: float,
    calibration: Dict[str, object],
) -> Tuple[Dict[str, float], List[str], bool]:
    ranked = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) < 2:
        return prob_map, [], False

    top1, top2 = ranked[0], ranked[1]
    diff_pct = (top1[1] - top2[1]) * 100.0
    if diff_pct >= tie_break_threshold:
        return prob_map, [], False

    adjusted = dict(prob_map)
    reasons: List[str] = [f"Top-2 gap {diff_pct:.2f}% < {tie_break_threshold:.2f}%; applying cricket tie-breaker"]
    mult = _adaptive_boost_multiplier(model_confidence, calibration)

    follow_h = float(summary.get("follow_through_height_max", 0.0))
    bat_ang = float(summary.get("bat_angle_impact", 0.0))
    swing_dir = float(summary.get("swing_direction_score", 0.0))

    boost_label = top1[0]
    if follow_h > 0.20:
        boost_label = "lofted"
        reasons.append("Tie-break favor lofted due to high follow-through")
    elif bat_ang < 20.0:
        boost_label = "defensive"
        reasons.append("Tie-break favor defensive due to low impact bat angle")
    elif abs(swing_dir) < 0.20:
        boost_label = "pull"
        reasons.append("Tie-break favor pull due to horizontal swing")

    if boost_label in adjusted:
        adjusted[boost_label] += 0.04 * mult
    reasons.append(f"Adaptive tie-break multiplier: {mult:.2f}")
    return _renormalize(adjusted), reasons, True


def train_classifier(training_x: np.ndarray, training_y: np.ndarray, random_state: int = 42) -> Dict[str, object]:
    scaler, normalize_idx = _fit_selected_scaler(training_x, FEATURE_ORDER)
    x_scaled = _transform_selected(training_x, scaler, normalize_idx)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(x_scaled, training_y)
    return {
        "model": clf,
        "labels": LABELS,
        "feature_order": FEATURE_ORDER,
        "sklearn_version": sklearn.__version__,
        "scaler": scaler,
        "normalize_indices": normalize_idx,
        "feature_weights": {k: 1.0 for k in FEATURE_ORDER},
    }


def save_classifier(bundle: Dict[str, object], model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_classifier(model_path: str | Path) -> Dict[str, object]:
    return joblib.load(Path(model_path))


def save_classifier_versioned(bundle: Dict[str, object], models_dir: str | Path, prefix: str = "shot_model_v") -> Path:
    model_root = Path(models_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    existing = list_model_versions(model_root, prefix=prefix)
    next_version = (max([v for _, v in existing]) + 1) if existing else 1
    out_path = model_root / f"{prefix}{next_version}.pkl"
    save_classifier(bundle, out_path)
    return out_path


def list_model_versions(models_dir: str | Path, prefix: str = "shot_model_v") -> List[Tuple[Path, int]]:
    model_root = Path(models_dir)
    if not model_root.exists():
        return []

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.pkl$")
    out: List[Tuple[Path, int]] = []
    for p in model_root.glob(f"{prefix}*.pkl"):
        m = pattern.match(p.name)
        if m:
            out.append((p, int(m.group(1))))
    return sorted(out, key=lambda x: x[1])


def get_latest_model_path(models_dir: str | Path, legacy_path: str | Path | None = None, prefix: str = "shot_model_v") -> Path | None:
    versions = list_model_versions(models_dir, prefix=prefix)
    if versions:
        return versions[-1][0]

    if legacy_path is not None:
        legacy = Path(legacy_path)
        if legacy.exists():
            return legacy
    return None


def classify_shot_ml(
    feature_series: Sequence[Dict[str, float]],
    model_bundle: Dict[str, object],
    window_size: int = 15,
    smooth_window: int = 7,
    low_conf_threshold: float | None = None,
    tie_break_threshold: float | None = None,
    debug: bool = False,
) -> Dict[str, object]:
    windows = _window_average(feature_series, window_size=window_size)
    if not windows:
        return {
            "shot_type": "Uncertain shot",
            "confidence_score": 0.0,
            "probabilities": {},
            "insight": "Insufficient frame features for classification.",
        }

    model: RandomForestClassifier = model_bundle["model"]
    labels = list(model_bundle.get("labels", LABELS))
    model_feature_order = list(model_bundle.get("feature_order", FEATURE_ORDER))
    calibration = model_bundle.get("calibration", {}) if isinstance(model_bundle, dict) else {}
    if low_conf_threshold is None:
        low_conf_threshold = float(calibration.get("low_confidence_threshold", 60.0))
    if tie_break_threshold is None:
        tie_break_threshold = float(calibration.get("tie_break_threshold", 10.0))

    scaler: StandardScaler | None = model_bundle.get("scaler") if isinstance(model_bundle, dict) else None
    normalize_idx = list(model_bundle.get("normalize_indices", _normalize_indices(model_feature_order))) if isinstance(model_bundle, dict) else _normalize_indices(model_feature_order)
    feature_weights: Dict[str, float] = model_bundle.get("feature_weights", {}) if isinstance(model_bundle, dict) else {}

    probabilities = []
    frame_predictions: List[str] = []
    for w in windows:
        vec = _vectorize(w, model_feature_order).reshape(1, -1)
        vec = _apply_feature_weights(vec, feature_weights, model_feature_order)
        vec = _transform_selected(vec, scaler, normalize_idx)
        proba = model.predict_proba(vec)[0]
        probabilities.append(proba)
        pred_idx = int(np.argmax(proba))
        frame_predictions.append(str(model.classes_[pred_idx]))

    smoothed_predictions: List[str] = []
    vote_win = int(np.clip(smooth_window, 5, 10))
    for i in range(len(frame_predictions)):
        recent = frame_predictions[max(0, i - vote_win + 1) : i + 1]
        voted = Counter(recent).most_common(1)[0][0]
        smoothed_predictions.append(str(voted))

    avg_proba = np.mean(np.vstack(probabilities), axis=0)
    model_prob_map = {str(model.classes_[i]): float(avg_proba[i]) for i in range(len(avg_proba))}
    summary = _extract_summary(feature_series)
    base_model_conf = float(np.max(avg_proba)) if len(avg_proba) else 0.0

    adjusted_map, heuristic_reasons = _apply_soft_heuristics(model_prob_map, summary, base_model_conf, calibration)
    adjusted_map, tie_reasons, tie_applied = _apply_tie_breaker(adjusted_map, summary, float(tie_break_threshold), base_model_conf, calibration)

    best_label = max(adjusted_map.items(), key=lambda x: x[1])[0]
    label_to_idx = {str(model.classes_[i]): i for i in range(len(model.classes_))}
    best_idx = label_to_idx.get(best_label, int(np.argmax(avg_proba)))
    confidence_model = float(adjusted_map.get(best_label, avg_proba[best_idx]))

    consistency = float(smoothed_predictions.count(best_label) / max(1, len(smoothed_predictions)))
    pose_visibility = float(np.mean([f.get("pose_visibility", 0.0) for f in feature_series])) if feature_series else 0.0
    valid_frames = sum(1 for f in feature_series if float(f.get("motion_phase", 0.0)) > 0.0 or float(f.get("pose_visibility", 0.0)) >= 0.35)
    valid_ratio = float(valid_frames / max(1, len(feature_series)))
    motion_clarity = float(np.mean([abs(f.get("swing_direction_score", 0.0)) + abs(f.get("bat_velocity", 0.0)) * 0.05 for f in feature_series])) if feature_series else 0.0
    motion_clarity = float(np.clip(motion_clarity, 0.0, 1.0))

    missing_key_ratio = 0.0
    if feature_series:
        key_feats = ["bat_angle_impact", "follow_through_height_max", "swing_direction_score", "player_body_lean"]
        missing = 0
        total = len(feature_series) * len(key_feats)
        for f in feature_series:
            for k in key_feats:
                if abs(float(f.get(k, 0.0))) < 1e-6:
                    missing += 1
        missing_key_ratio = float(missing / max(1, total))

    confidence = float(
        np.clip(
            0.50 * confidence_model
            + 0.20 * consistency
            + 0.10 * pose_visibility
            + 0.08 * valid_ratio
            + 0.12 * motion_clarity
            - 0.15 * (1.0 - consistency)
            - 0.12 * missing_key_ratio,
            0.0,
            1.0,
        )
    )

    probability_map = {k: round(v * 100.0, 2) for k, v in sorted(adjusted_map.items(), key=lambda x: x[1], reverse=True)}
    top3_before = sorted({str(model.classes_[i]): float(avg_proba[i]) for i in range(len(avg_proba))}.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_after = sorted(adjusted_map.items(), key=lambda x: x[1], reverse=True)[:3]

    low_conf_warning = ""
    if confidence * 100.0 < low_conf_threshold:
        low_conf_warning = "Low confidence due to poor visibility or unstable camera"

    if confidence < 0.4:
        return {
            "shot_type": "Uncertain shot",
            "confidence_score": round(confidence * 100.0, 2),
            "probabilities": probability_map,
            "consistency_score": round(consistency * 100.0, 2),
            "valid_frame_count": int(valid_frames),
            "valid_frame_ratio": round(valid_ratio * 100.0, 2),
            "pose_visibility_score": round(pose_visibility * 100.0, 2),
            "insight": "Model confidence below threshold. Try clearer camera angle and complete follow-through.",
            "low_confidence_warning": low_conf_warning,
            "debug": {
                "summary_features": summary,
                "top3_before": [(k, round(v * 100.0, 2)) for k, v in top3_before],
                "top3_after": [(k, round(v * 100.0, 2)) for k, v in top3_after],
                "heuristic_reasons": heuristic_reasons + tie_reasons,
                "tie_break_applied": tie_applied,
                "tie_break_threshold": float(tie_break_threshold),
                "missing_key_ratio": round(missing_key_ratio, 4),
                "motion_clarity": round(motion_clarity, 4),
            } if debug else {},
        }

    insight = {
        "defensive": "Compact bat path and controlled body posture suggest a defensive shot.",
        "drive": "Forward extension and stable transfer indicate a drive.",
        "lofted": "High follow-through and upward bat path indicate a lofted shot attempt.",
        "pull": "Horizontal bat travel with torso rotation aligns with a pull shot.",
        "cut": "Lateral bat path with open shoulders suggests a cut shot.",
        "sweep": "Low stance and across-line bat movement indicate a sweep.",
    }.get(str(best_label), "Shot classified from temporal pose and object features.")

    return {
        "shot_type": str(best_label),
        "confidence_score": round(confidence * 100.0, 2),
        "probabilities": probability_map,
        "consistency_score": round(consistency * 100.0, 2),
        "valid_frame_count": int(valid_frames),
        "valid_frame_ratio": round(valid_ratio * 100.0, 2),
        "pose_visibility_score": round(pose_visibility * 100.0, 2),
        "insight": insight,
        "low_confidence_warning": low_conf_warning,
        "debug": {
            "summary_features": summary,
            "top3_before": [(k, round(v * 100.0, 2)) for k, v in top3_before],
            "top3_after": [(k, round(v * 100.0, 2)) for k, v in top3_after],
            "heuristic_reasons": heuristic_reasons + tie_reasons,
            "tie_break_applied": tie_applied,
            "tie_break_threshold": float(tie_break_threshold),
            "missing_key_ratio": round(missing_key_ratio, 4),
            "motion_clarity": round(motion_clarity, 4),
            "decision_reason": (heuristic_reasons + tie_reasons)[-1] if (heuristic_reasons or tie_reasons) else "ML probability dominated",
        } if debug else {},
    }


def generate_contextual_feedback(classification: Dict[str, object], biomech_scores: Dict[str, float], feature_series: Sequence[Dict[str, float]]) -> str:
    shot = str(classification.get("shot_type", "Uncertain shot"))
    confidence = float(classification.get("confidence_score", 0.0))
    technique = float(biomech_scores.get("technique_score", 0.0))
    balance = float(biomech_scores.get("balance_score", 0.0))

    if shot == "Uncertain shot":
        return "Shot classification is uncertain. Capture more side-on frames through full impact and follow-through."

    if not feature_series:
        return f"Detected {shot} at {confidence:.1f}% confidence."

    avg_follow = float(np.mean([f.get("follow_through_height", 0.0) for f in feature_series]))
    avg_head = float(np.mean([f.get("head_position", 0.0) for f in feature_series]))
    avg_elbow = float(np.mean([f.get("elbow_angle", 0.0) for f in feature_series]))
    avg_body_lean = float(np.mean([f.get("body_lean", 0.0) for f in feature_series]))
    avg_wrist_speed = float(np.mean([f.get("wrist_trajectory", 0.0) for f in feature_series]))

    if shot == "lofted" and balance < 65.0:
        return "Follow-through height indicates an attempted lofted shot, but balance was lost at impact. Stabilize head position and front-leg base."
    if shot == "drive" and technique < 70.0:
        return "Drive intent is clear, but bat-arm alignment is inconsistent. Keep elbow lead and maintain straighter bat path through contact."
    if shot in {"pull", "cut"} and avg_head > 35.0:
        return "Horizontal shot mechanics are present, but head movement is high. Keep head quieter to improve timing and control."
    if shot == "sweep" and avg_follow < 0.1:
        return "Sweep setup is detected, but follow-through is short. Extend through the line for better control."

    if avg_elbow < 95.0:
        return "Key issue: early elbow drop. Suggestion: keep front elbow higher through impact to maintain bat control."
    if avg_body_lean > 20.0:
        return "Key issue: excessive body lean. Suggestion: keep shoulder-hip alignment more upright for balance and timing."
    if avg_wrist_speed < 2.5 and shot in {"drive", "cut", "pull"}:
        return "Key issue: limited wrist acceleration through contact. Suggestion: accelerate wrists later and extend through the ball."

    return f"{shot.title()} classified at {confidence:.1f}% confidence with stable biomechanics signals."


def build_training_matrices(samples: Sequence[Tuple[Dict[str, float], str]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([_vectorize(features, FEATURE_ORDER) for features, _ in samples], dtype=np.float32)
    y = np.array([label for _, label in samples])
    return x, y
