from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

from core.shot_classifier import (
    generate_synthetic_training_data,
    load_reference_profiles,
    predict_shot,
    save_model,
    train_model,
)


ROOT_DIR = Path(__file__).parent
REFERENCE_DIR = ROOT_DIR / "reference_data"
MODEL_PATH = ROOT_DIR / "assets" / "shot_classifier.pkl"


def evaluate_model(
    model_bundle: Dict[str, object],
    test_sequences: List[List[Dict[str, float]]],
    test_labels: List[str],
) -> float:
    preds = [predict_shot(sequence, model_bundle)["shot_type"] for sequence in test_sequences]
    if not test_labels:
        return 0.0
    return float(np.mean(np.array(preds) == np.array(test_labels)))


def main() -> None:
    print("Loading reference profiles...")
    reference_profiles = load_reference_profiles(REFERENCE_DIR)

    print("Generating synthetic pose-sequence dataset...")
    sequences, labels = generate_synthetic_training_data(
        reference_profiles=reference_profiles,
        samples_per_class=220,
        min_seq_len=18,
        max_seq_len=40,
        random_state=42,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        sequences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print("Training shot classifier...")
    model_bundle = train_model(
        sequences=x_train,
        labels=y_train,
        model_type="random_forest",
        random_state=42,
    )

    test_accuracy = evaluate_model(model_bundle, x_test, y_test)
    model_bundle["test_accuracy"] = test_accuracy

    print(f"Train Accuracy: {model_bundle['train_accuracy']:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Classes       : {model_bundle['classes']}")
    print(f"Samples       : {len(labels)} | Distribution: {dict(Counter(labels))}")

    save_model(model_bundle, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
