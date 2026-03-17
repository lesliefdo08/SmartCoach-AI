from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier


LABELS = ["defensive", "drive", "lofted", "pull", "cut", "sweep"]
FEATURE_ORDER = [
    "bat_swing_arc",
    "bat_angle",
    "bat_follow_through_height",
    "follow_through_height",
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
    "elbow_angle_mean",
    "elbow_angle_max",
    "elbow_angle_velocity",
    "shoulder_rotation_mean",
    "shoulder_rotation_max",
    "shoulder_rotation_velocity",
    "wrist_trajectory_mean",
    "wrist_trajectory_max",
    "wrist_trajectory_velocity",
    "body_lean_mean",
    "body_lean_max",
    "body_lean_velocity",
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
        agg = {key: float(np.mean([s.get(key, 0.0) for s in chunk])) for key in FEATURE_ORDER}
        for key in temporal_keys:
            arr = np.array([float(s.get(key, 0.0)) for s in chunk], dtype=np.float32)
            agg[f"{key}_mean"] = float(np.mean(arr)) if len(arr) else 0.0
            agg[f"{key}_max"] = float(np.max(arr)) if len(arr) else 0.0
            agg[f"{key}_velocity"] = float(np.mean(np.abs(np.diff(arr)))) if len(arr) >= 2 else 0.0
        out.append(agg)
    return out


def _vectorize(features: Dict[str, float], feature_order: Sequence[str]) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in feature_order], dtype=np.float32)


def train_classifier(training_x: np.ndarray, training_y: np.ndarray, random_state: int = 42) -> Dict[str, object]:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(training_x, training_y)
    return {
        "model": clf,
        "labels": LABELS,
        "feature_order": FEATURE_ORDER,
        "sklearn_version": sklearn.__version__,
    }


def save_classifier(bundle: Dict[str, object], model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_classifier(model_path: str | Path) -> Dict[str, object]:
    return joblib.load(Path(model_path))


def classify_shot_ml(
    feature_series: Sequence[Dict[str, float]],
    model_bundle: Dict[str, object],
    window_size: int = 15,
    smooth_window: int = 7,
    low_conf_threshold: float = 60.0,
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

    probabilities = []
    frame_predictions: List[str] = []
    for w in windows:
        vec = _vectorize(w, model_feature_order).reshape(1, -1)
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
    best_label = Counter(smoothed_predictions).most_common(1)[0][0]
    label_to_idx = {str(model.classes_[i]): i for i in range(len(model.classes_))}
    best_idx = label_to_idx.get(best_label, int(np.argmax(avg_proba)))
    confidence_model = float(avg_proba[best_idx])

    consistency = float(smoothed_predictions.count(best_label) / max(1, len(smoothed_predictions)))
    pose_visibility = float(np.mean([f.get("pose_visibility", 0.0) for f in feature_series])) if feature_series else 0.0
    valid_frames = sum(1 for f in feature_series if float(f.get("motion_phase", 0.0)) > 0.0 or float(f.get("pose_visibility", 0.0)) >= 0.35)
    valid_ratio = float(valid_frames / max(1, len(feature_series)))
    confidence = float(
        np.clip(
            0.5 * confidence_model + 0.25 * consistency + 0.15 * pose_visibility + 0.10 * valid_ratio,
            0.0,
            1.0,
        )
    )

    probability_map = {str(model.classes_[i]): round(float(avg_proba[i]) * 100.0, 2) for i in range(len(avg_proba))}

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
