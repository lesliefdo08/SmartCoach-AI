"""Shot classification utilities using temporal pose features."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


BASE_FEATURES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "hip_alignment",
    "spine_tilt",
]

SHOT_CLASSES = ["cover_drive", "straight_drive", "pull_shot", "defense"]


def extract_temporal_features(pose_sequence: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Convert per-frame pose features into a fixed-length temporal vector."""
    feature_vector: Dict[str, float] = {}

    for feature in BASE_FEATURES:
        series = np.array([frame.get(feature, np.nan) for frame in pose_sequence], dtype=np.float32)
        series = series[np.isfinite(series)]

        if len(series) == 0:
            stats = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "range": 0.0,
                "delta": 0.0,
                "velocity": 0.0,
            }
        else:
            diffs = np.diff(series) if len(series) > 1 else np.array([0.0], dtype=np.float32)
            stats = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "min": float(np.min(series)),
                "max": float(np.max(series)),
                "range": float(np.max(series) - np.min(series)),
                "delta": float(series[-1] - series[0]),
                "velocity": float(np.mean(np.abs(diffs))),
            }

        for k, v in stats.items():
            feature_vector[f"{feature}_{k}"] = v

    feature_vector["sequence_length"] = float(len(pose_sequence))
    return feature_vector


def create_pose_sequence_dataset(
    sequences: Sequence[Sequence[Dict[str, float]]],
    labels: Sequence[str],
) -> pd.DataFrame:
    """Create a tabular dataset format from temporal pose sequences.

    Each row corresponds to one pose sequence, represented by temporal summary
    features (mean/std/min/max/range/delta/velocity) for all base joint angles.
    """
    rows: List[Dict[str, float | str]] = []
    for sequence, label in zip(sequences, labels):
        row = extract_temporal_features(sequence)
        row["label"] = label
        rows.append(row)

    return pd.DataFrame(rows)


def train_model(
    sequences: Sequence[Sequence[Dict[str, float]]],
    labels: Sequence[str],
    model_type: str = "random_forest",
    random_state: int = 42,
) -> Dict[str, object]:
    """Train a shot classifier using sequence-level temporal features."""
    dataset = create_pose_sequence_dataset(sequences, labels)
    if dataset.empty:
        raise ValueError("Training dataset is empty.")

    x = dataset.drop(columns=["label"]).astype(np.float32)
    y = dataset["label"].astype(str)

    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=random_state)
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
        )

    model.fit(x, y)

    return {
        "model": model,
        "feature_columns": list(x.columns),
        "classes": list(model.classes_),
        "model_type": model_type,
        "train_accuracy": float(model.score(x, y)),
    }


def predict_shot(
    pose_sequence: Sequence[Dict[str, float]],
    model_bundle: Dict[str, object],
) -> Dict[str, object]:
    """Predict shot type and class probabilities from a sequence of pose features."""
    if not pose_sequence:
        return {
            "shot_type": "unknown",
            "confidence_score": 0.0,
            "probabilities": {shot: 0.0 for shot in SHOT_CLASSES},
        }

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    classes = model_bundle.get("classes", SHOT_CLASSES)

    temporal = extract_temporal_features(pose_sequence)
    x = pd.DataFrame([temporal]).reindex(columns=feature_columns, fill_value=0.0)

    probs = model.predict_proba(x)[0]
    best_idx = int(np.argmax(probs))
    shot_type = str(classes[best_idx])

    probability_map = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
    for shot in SHOT_CLASSES:
        probability_map.setdefault(shot, 0.0)

    return {
        "shot_type": shot_type,
        "confidence_score": round(float(probs[best_idx]) * 100.0, 2),
        "probabilities": probability_map,
    }


def save_model(model_bundle: Dict[str, object], model_path: str | Path) -> None:
    """Persist model bundle to disk."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model_bundle, f)


def load_model(model_path: str | Path) -> Dict[str, object]:
    """Load model bundle from disk."""
    path = Path(model_path)
    with path.open("rb") as f:
        return pickle.load(f)


def load_reference_profiles(reference_dir: str | Path) -> Dict[str, Dict[str, float]]:
    """Load reference angle profiles by shot type."""
    ref_dir = Path(reference_dir)
    profiles: Dict[str, Dict[str, float]] = {}

    for shot in SHOT_CLASSES:
        pro_path = ref_dir / f"{shot}_pro.json"
        ref_path = pro_path if pro_path.exists() else (ref_dir / f"{shot}.json")
        with ref_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "joint_angles_mean" in data:
            profiles[shot] = {k: float(v) for k, v in data["joint_angles_mean"].items()}
        else:
            profiles[shot] = {k: float(v) for k, v in data["angles"].items()}

    return profiles


def generate_synthetic_training_data(
    reference_profiles: Dict[str, Dict[str, float]],
    samples_per_class: int = 180,
    min_seq_len: int = 18,
    max_seq_len: int = 38,
    random_state: int = 42,
) -> Tuple[List[List[Dict[str, float]]], List[str]]:
    """Generate simulated sequence data around reference poses for model training."""
    rng = np.random.default_rng(random_state)
    sequences: List[List[Dict[str, float]]] = []
    labels: List[str] = []

    for shot, reference in reference_profiles.items():
        for _ in range(samples_per_class):
            seq_len = int(rng.integers(min_seq_len, max_seq_len + 1))
            phase = np.linspace(0.0, 1.0, seq_len)
            sequence: List[Dict[str, float]] = []

            for t in phase:
                frame: Dict[str, float] = {}
                motion_intensity = float(np.sin(t * np.pi))

                for feature in BASE_FEATURES:
                    base = float(reference[feature])
                    trend = (t - 0.5) * float(rng.normal(0.0, 8.0))
                    wave = motion_intensity * float(rng.normal(0.0, 6.0))
                    noise = float(rng.normal(0.0, 5.0))
                    value = float(np.clip(base + trend + wave + noise, 0.0, 180.0))
                    frame[feature] = value

                sequence.append(frame)

            sequences.append(sequence)
            labels.append(shot)

    return sequences, labels
