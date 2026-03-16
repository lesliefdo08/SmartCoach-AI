from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.shot_classifier import build_training_matrices, save_classifier, train_classifier
from modules.video_processor import CricketAnalyticsPipeline


DATASET_DIR = ROOT_DIR / "dataset"
MODEL_PATH = ROOT_DIR / "models" / "shot_classifier.pkl"
LABELS = ["defensive", "drive", "lofted", "pull", "cut"]


def _collect_samples_from_dataset() -> List[Tuple[Dict[str, float], str]]:
    if not DATASET_DIR.exists():
        return []

    pipeline = CricketAnalyticsPipeline(sample_rate=2)
    samples: List[Tuple[Dict[str, float], str]] = []

    for label in LABELS:
        label_dir = DATASET_DIR / label
        if not label_dir.exists():
            continue
        videos = list(label_dir.glob("*.mp4")) + list(label_dir.glob("*.mkv")) + list(label_dir.glob("*.webm"))
        for video in videos:
            out = pipeline.process_video(video)
            for f in out.get("window_features", []):
                samples.append((f, label))

    pipeline.close()
    return samples


def _synthetic_samples_if_needed(samples: List[Tuple[Dict[str, float], str]]) -> List[Tuple[Dict[str, float], str]]:
    if samples:
        return samples

    rng = np.random.default_rng(42)
    priors = {
        "defensive": [80, 68, 0.05, 8, 155, 9, 20, 4, 15],
        "drive": [130, 42, 0.12, 14, 145, 12, 26, 7, 20],
        "lofted": [220, 18, 0.20, 18, 140, 14, 34, 11, 35],
        "pull": [170, 5, 0.09, 22, 135, 11, 30, 10, -5],
        "cut": [160, -8, 0.08, 24, 138, 10, 28, 9, -20],
    }
    keys = [
        "bat_swing_arc",
        "bat_angle",
        "follow_through_height",
        "shoulder_rotation",
        "knee_bend",
        "torso_tilt",
        "head_position",
        "bat_velocity",
        "ball_direction",
    ]

    generated: List[Tuple[Dict[str, float], str]] = []
    for label, base in priors.items():
        for _ in range(320):
            vals = rng.normal(loc=np.array(base, dtype=np.float32), scale=np.array([12, 10, 0.04, 5, 10, 4, 8, 2, 10], dtype=np.float32))
            generated.append(({k: float(v) for k, v in zip(keys, vals)}, label))

    return generated


def main() -> None:
    print("Collecting dataset features...")
    samples = _collect_samples_from_dataset()
    samples = _synthetic_samples_if_needed(samples)

    x, y = build_training_matrices(samples)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    bundle = train_classifier(x_train, y_train)
    model = bundle["model"]

    y_pred = model.predict(x_val)
    acc = float(accuracy_score(y_val, y_pred))

    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_val, y_pred, digits=3))

    save_classifier(bundle, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")

    if acc < 0.70:
        raise SystemExit("Validation accuracy below 70%. Add more labeled clips and retrain.")


if __name__ == "__main__":
    main()
