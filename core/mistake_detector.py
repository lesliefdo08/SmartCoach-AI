"""AI-based technique mistake detection for SmartCoach AI."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


MISTAKE_LABELS = [
    "early_bat_swing",
    "low_elbow",
    "poor_head_alignment",
    "straight_front_leg",
    "imbalanced_stance",
]

FEATURE_COLUMNS = [
    "elbow_angle",
    "knee_angle",
    "hip_alignment",
    "spine_tilt",
    "head_position",
    "shoulder_rotation",
]


def to_mistake_feature_vector(pose_features: Dict[str, float]) -> Dict[str, float]:
    """Convert raw pose features into mistake-model feature vector."""
    left_elbow = float(pose_features.get("left_elbow_angle", np.nan))
    right_elbow = float(pose_features.get("right_elbow_angle", np.nan))
    left_knee = float(pose_features.get("left_knee_angle", np.nan))
    right_knee = float(pose_features.get("right_knee_angle", np.nan))
    left_shoulder = float(pose_features.get("left_shoulder_angle", np.nan))
    right_shoulder = float(pose_features.get("right_shoulder_angle", np.nan))
    hip_alignment = float(pose_features.get("hip_alignment", np.nan))
    spine_tilt = float(pose_features.get("spine_tilt", np.nan))

    elbow_angle = np.nanmean([left_elbow, right_elbow])
    knee_angle = np.nanmean([left_knee, right_knee])
    shoulder_rotation = abs(left_shoulder - right_shoulder) if np.isfinite(left_shoulder) and np.isfinite(right_shoulder) else np.nan
    head_position = spine_tilt

    vector = {
        "elbow_angle": float(elbow_angle if np.isfinite(elbow_angle) else 0.0),
        "knee_angle": float(knee_angle if np.isfinite(knee_angle) else 0.0),
        "hip_alignment": float(hip_alignment if np.isfinite(hip_alignment) else 0.0),
        "spine_tilt": float(spine_tilt if np.isfinite(spine_tilt) else 0.0),
        "head_position": float(head_position if np.isfinite(head_position) else 0.0),
        "shoulder_rotation": float(shoulder_rotation if np.isfinite(shoulder_rotation) else 0.0),
    }
    return vector


def _sequence_to_vector(sequence_or_features: Sequence[Dict[str, float]] | Dict[str, float]) -> Dict[str, float]:
    if isinstance(sequence_or_features, dict):
        return to_mistake_feature_vector(sequence_or_features)

    if len(sequence_or_features) == 0:
        return {k: 0.0 for k in FEATURE_COLUMNS}

    rows = [to_mistake_feature_vector(item) for item in sequence_or_features]
    df = pd.DataFrame(rows)
    means = df.mean(axis=0, numeric_only=True)
    return {k: float(means.get(k, 0.0)) for k in FEATURE_COLUMNS}


def train_mistake_model(
    feature_vectors: Sequence[Dict[str, float]],
    labels: Sequence[Sequence[str]],
    random_state: int = 42,
) -> Dict[str, object]:
    """Train a multi-label mistake classifier."""
    if len(feature_vectors) == 0:
        raise ValueError("No training feature vectors provided.")

    x = pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0.0).astype(np.float32)

    mlb = MultiLabelBinarizer(classes=MISTAKE_LABELS)
    y = mlb.fit_transform(labels)

    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    model = OneVsRestClassifier(base)
    model.fit(x, y)

    train_pred = model.predict(x)
    train_acc = float(np.mean((train_pred == y).all(axis=1)))

    return {
        "model": model,
        "mlb": mlb,
        "feature_columns": FEATURE_COLUMNS,
        "train_accuracy": train_acc,
    }


def predict_mistakes(
    pose_input: Sequence[Dict[str, float]] | Dict[str, float],
    model_bundle: Dict[str, object],
    threshold: float = 0.45,
) -> Dict[str, object]:
    """Predict technique mistakes and confidence scores."""
    vector = _sequence_to_vector(pose_input)
    x = pd.DataFrame([vector]).reindex(columns=model_bundle["feature_columns"], fill_value=0.0).astype(np.float32)

    model = model_bundle["model"]
    mlb = model_bundle["mlb"]

    probs = model.predict_proba(x)[0]
    labels = list(mlb.classes_)
    probability_map = {str(label): float(prob) for label, prob in zip(labels, probs)}

    detected = [
        {
            "label": label,
            "confidence": round(prob * 100.0, 2),
            "display_name": _display_name(label),
        }
        for label, prob in probability_map.items()
        if prob >= threshold
    ]
    detected.sort(key=lambda x: x["confidence"], reverse=True)

    if not detected:
        best_label = max(probability_map.items(), key=lambda x: x[1])[0]
        detected = [
            {
                "label": best_label,
                "confidence": round(float(probability_map[best_label]) * 100.0, 2),
                "display_name": _display_name(best_label),
            }
        ]

    return {
        "detected_mistakes": detected,
        "probabilities": probability_map,
        "feature_vector": vector,
    }


def save_model(model_bundle: Dict[str, object], model_path: str | Path) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model_bundle, f)


def load_model(model_path: str | Path) -> Dict[str, object]:
    path = Path(model_path)
    with path.open("rb") as f:
        return pickle.load(f)


def generate_synthetic_mistake_data(
    samples: int = 2000,
    random_state: int = 42,
) -> tuple[List[Dict[str, float]], List[List[str]]]:
    """Simulate training data for common cricket technique mistakes."""
    rng = np.random.default_rng(random_state)

    feature_vectors: List[Dict[str, float]] = []
    labels: List[List[str]] = []

    for _ in range(samples):
        vec = {
            "elbow_angle": float(np.clip(rng.normal(108, 20), 40, 175)),
            "knee_angle": float(np.clip(rng.normal(145, 15), 90, 180)),
            "hip_alignment": float(np.clip(abs(rng.normal(6, 4)), 0, 30)),
            "spine_tilt": float(np.clip(abs(rng.normal(10, 6)), 0, 40)),
            "head_position": 0.0,
            "shoulder_rotation": float(np.clip(abs(rng.normal(15, 10)), 0, 60)),
        }
        vec["head_position"] = vec["spine_tilt"]

        row_labels: List[str] = []
        if vec["elbow_angle"] < 95:
            row_labels.append("low_elbow")
        if vec["spine_tilt"] > 14 or vec["head_position"] > 14:
            row_labels.append("poor_head_alignment")
        if vec["knee_angle"] > 158:
            row_labels.append("straight_front_leg")
        if vec["hip_alignment"] > 9 or vec["shoulder_rotation"] > 24:
            row_labels.append("imbalanced_stance")
        if vec["shoulder_rotation"] > 22 and vec["elbow_angle"] < 105:
            row_labels.append("early_bat_swing")

        if rng.random() < 0.06:
            # random label noise for robustness
            row_labels.append(str(rng.choice(MISTAKE_LABELS)))

        row_labels = sorted(set(row_labels))
        if len(row_labels) == 0:
            # keep negative-ish examples by assigning mild issue with low prevalence
            if rng.random() < 0.15:
                row_labels = ["imbalanced_stance"]

        feature_vectors.append(vec)
        labels.append(row_labels)

    return feature_vectors, labels


def _display_name(label: str) -> str:
    mapping = {
        "early_bat_swing": "Early Bat Swing",
        "low_elbow": "Low Elbow",
        "poor_head_alignment": "Head Misalignment",
        "straight_front_leg": "Front Leg Too Straight",
        "imbalanced_stance": "Imbalanced Stance",
    }
    return mapping.get(label, label.replace("_", " ").title())
