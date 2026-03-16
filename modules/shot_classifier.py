from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


LABELS = ["defensive", "drive", "lofted", "pull", "cut", "sweep"]
FEATURE_ORDER = [
    "bat_swing_arc",
    "bat_angle",
    "bat_follow_through_height",
    "follow_through_height",
    "shoulder_rotation",
    "knee_bend",
    "torso_tilt",
    "head_position",
    "bat_velocity",
    "ball_direction",
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
    for i in range(0, len(series) - win + 1):
        chunk = series[i : i + win]
        out.append({key: float(np.mean([s.get(key, 0.0) for s in chunk])) for key in FEATURE_ORDER})
    return out


def _vectorize(features: Dict[str, float]) -> np.ndarray:
    return np.array([float(features.get(k, 0.0)) for k in FEATURE_ORDER], dtype=np.float32)


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
    }


def save_classifier(bundle: Dict[str, object], model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_classifier(model_path: str | Path) -> Dict[str, object]:
    return joblib.load(Path(model_path))


def classify_shot_ml(feature_series: Sequence[Dict[str, float]], model_bundle: Dict[str, object], window_size: int = 15) -> Dict[str, object]:
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

    probabilities = []
    for w in windows:
        vec = _vectorize(w).reshape(1, -1)
        probabilities.append(model.predict_proba(vec)[0])

    avg_proba = np.mean(np.vstack(probabilities), axis=0)
    best_idx = int(np.argmax(avg_proba))
    best_label = model.classes_[best_idx]
    confidence = float(avg_proba[best_idx])

    probability_map = {str(model.classes_[i]): round(float(avg_proba[i]) * 100.0, 2) for i in range(len(avg_proba))}
    if confidence < 0.5:
        return {
            "shot_type": "Uncertain shot",
            "confidence_score": round(confidence * 100.0, 2),
            "probabilities": probability_map,
            "insight": "Model confidence below 50%. Try clearer camera angle and complete follow-through.",
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
        "insight": insight,
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

    if shot == "lofted" and balance < 65.0:
        return "Follow-through height indicates an attempted lofted shot, but balance was lost at impact. Stabilize head position and front-leg base."
    if shot == "drive" and technique < 70.0:
        return "Drive intent is clear, but bat-arm alignment is inconsistent. Keep elbow lead and maintain straighter bat path through contact."
    if shot in {"pull", "cut"} and avg_head > 35.0:
        return "Horizontal shot mechanics are present, but head movement is high. Keep head quieter to improve timing and control."
    if shot == "sweep" and avg_follow < 0.1:
        return "Sweep setup is detected, but follow-through is short. Extend through the line for better control."

    return f"{shot.title()} classified at {confidence:.1f}% confidence with stable biomechanics signals."


def build_training_matrices(samples: Sequence[Tuple[Dict[str, float], str]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([_vectorize(features) for features, _ in samples], dtype=np.float32)
    y = np.array([label for _, label in samples])
    return x, y
