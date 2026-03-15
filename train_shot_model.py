from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from modules.shot_classifier import build_training_matrices, save_classifier, train_classifier
from modules.video_processor import CricketAnalyticsPipeline


ROOT_DIR = Path(__file__).parent
TRAINING_DIR = ROOT_DIR / "training_data"
MODEL_PATH = ROOT_DIR / "models" / "shot_classifier.pkl"
VALID_LABELS = ["defensive", "drive", "lofted", "pull", "cut", "sweep"]


def collect_samples_from_videos() -> List[Tuple[Dict[str, float], str]]:
    pipeline = CricketAnalyticsPipeline(sample_rate=2)
    samples: List[Tuple[Dict[str, float], str]] = []

    for label in VALID_LABELS:
        label_dir = TRAINING_DIR / label
        if not label_dir.exists():
            continue
        for video_file in sorted(label_dir.glob("*.mp4")):
            out = pipeline.process_video(video_file)
            for f in out.get("window_features", []):
                samples.append((f, label))

    pipeline.close()
    return samples


def augment_if_sparse(samples: List[Tuple[Dict[str, float], str]]) -> List[Tuple[Dict[str, float], str]]:
    if samples:
        return samples

    rng = np.random.default_rng(42)
    synthetic: List[Tuple[Dict[str, float], str]] = []
    priors = {
        "defensive": [80, 68, 0.05, 6, 155, 20, 4, 15],
        "drive": [130, 42, 0.12, 12, 145, 26, 7, 20],
        "lofted": [220, 18, 0.20, 14, 140, 34, 11, 35],
        "pull": [170, 5, 0.09, 20, 135, 30, 10, -5],
        "cut": [160, -8, 0.08, 22, 138, 28, 9, -20],
        "sweep": [145, -28, 0.06, 16, 122, 24, 8, -40],
    }
    keys = [
        "bat_swing_arc",
        "bat_angle",
        "follow_through_height",
        "body_rotation",
        "knee_bend",
        "head_position",
        "bat_velocity",
        "ball_direction",
    ]

    for label, base in priors.items():
        for _ in range(240):
            vals = rng.normal(loc=np.array(base, dtype=np.float32), scale=np.array([12, 10, 0.04, 5, 10, 8, 2, 10], dtype=np.float32))
            sample = {k: float(v) for k, v in zip(keys, vals)}
            synthetic.append((sample, label))
    return synthetic


def main() -> None:
    print("Collecting labeled training videos...")
    samples = collect_samples_from_videos()
    samples = augment_if_sparse(samples)

    x, y = build_training_matrices(samples)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    bundle = train_classifier(x_train, y_train)
    model = bundle["model"]
    train_acc = float(model.score(x_train, y_train))
    test_acc = float(model.score(x_test, y_test))

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(classification_report(y_test, model.predict(x_test), digits=3))

    save_classifier(bundle, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
