from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from modules.shot_classifier import FEATURE_ORDER, LABELS, save_classifier, save_classifier_versioned


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
LEGACY_MODEL_PATH = MODELS_DIR / "shot_classifier.pkl"


def _augment_rows_if_small(df: pd.DataFrame, min_rows: int = 400, random_state: int = 42) -> pd.DataFrame:
    if len(df) >= min_rows:
        return df

    rng = np.random.default_rng(random_state)
    numeric_cols = [c for c in FEATURE_ORDER if c in df.columns]
    if not numeric_cols or df.empty:
        return df

    needed = min_rows - len(df)
    picks = df.sample(n=needed, replace=True, random_state=random_state).copy()
    for col in numeric_cols:
        std = float(df[col].std()) if col in df.columns else 0.0
        noise_scale = 0.03 * max(1.0, abs(std))
        picks[col] = picks[col].astype(float) + rng.normal(0.0, noise_scale, size=len(picks))

    out = pd.concat([df, picks], ignore_index=True)
    return out


def _build_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_cols = [c for c in FEATURE_ORDER if c in df.columns]
    if not feature_cols:
        raise ValueError("No model features found in dataset. Run generate_dataset.py first.")

    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0.0

    x = df[FEATURE_ORDER].astype(float).to_numpy(dtype=np.float32)
    y = df["shot_type"].astype(str).to_numpy()
    return x, y, FEATURE_ORDER


def _train_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, random_state: int) -> object:
    if model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=random_state)
    else:
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=24,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    clf.fit(x_train, y_train)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shot classifier from generated feature dataset.")
    parser.add_argument("--dataset-csv", type=str, default="dataset/features_dataset.csv")
    parser.add_argument("--model", type=str, choices=["random_forest", "gradient_boosting"], default="random_forest")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--augment-if-small", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.dataset_csv)
    if not data_path.exists():
        raise SystemExit(f"Dataset file not found: {data_path}. Run generate_dataset.py first.")

    df = pd.read_csv(data_path)
    if "shot_type" not in df.columns:
        raise SystemExit("Dataset must include 'shot_type' column.")

    df = df[df["shot_type"].isin(LABELS)].copy()
    if args.augment_if_small:
        df = _augment_rows_if_small(df)

    x, y, feature_order = _build_xy(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=y,
    )

    clf = _train_model(args.model, x_train, y_train, random_state=int(args.random_state))
    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    print("=== Training Results ===")
    print(f"Model         : {args.model}")
    print(f"Samples       : {len(df)}")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision(M)  : {prec:.4f}")
    print(f"Recall(M)     : {rec:.4f}")
    print(f"F1(M)         : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    conf = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    print("Confusion Matrix:")
    print(conf)

    max_proba = np.max(y_proba, axis=1)
    correct_mask = (y_pred == y_test)
    if np.any(correct_mask):
        low_conf_threshold = float(np.percentile(max_proba[correct_mask], 20) * 100.0)
    else:
        low_conf_threshold = 60.0
    low_conf_threshold = float(np.clip(low_conf_threshold, 45.0, 75.0))

    bundle: Dict[str, object] = {
        "model": clf,
        "labels": LABELS,
        "feature_order": feature_order,
        "calibration": {
            "low_confidence_threshold": round(low_conf_threshold, 2),
            "model_name": args.model,
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    versioned_path = save_classifier_versioned(bundle, MODELS_DIR)
    save_classifier(bundle, LEGACY_MODEL_PATH)

    metrics_path = MODELS_DIR / "latest_training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "samples": int(len(df)),
                "accuracy": round(acc, 6),
                "precision_macro": round(float(prec), 6),
                "recall_macro": round(float(rec), 6),
                "f1_macro": round(float(f1), 6),
                "low_confidence_threshold": round(low_conf_threshold, 2),
            },
            f,
            indent=2,
        )

    print(f"Saved legacy model  : {LEGACY_MODEL_PATH}")
    print(f"Saved versioned model: {versioned_path}")
    print(f"Saved metrics       : {metrics_path}")


if __name__ == "__main__":
    main()
