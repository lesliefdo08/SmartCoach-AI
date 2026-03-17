from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from modules.shot_classifier import FEATURE_ORDER, classify_shot_ml, get_latest_model_path, load_classifier, save_classifier
from modules.video_processor import CricketAnalyticsPipeline


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
LEGACY_MODEL = MODELS_DIR / "shot_classifier.pkl"


def _apply_bundle_preprocess(model_bundle: Dict[str, object], x: np.ndarray) -> np.ndarray:
    feature_order = list(model_bundle.get("feature_order", FEATURE_ORDER))
    feature_weights = model_bundle.get("feature_weights", {})
    scaler = model_bundle.get("scaler")
    normalize_idx = list(model_bundle.get("normalize_indices", []))

    x_out = np.array(x, dtype=np.float32, copy=True)
    if feature_weights:
        for i, name in enumerate(feature_order):
            x_out[:, i] = x_out[:, i] * float(feature_weights.get(name, 1.0))
    if scaler is not None and normalize_idx:
        x_out[:, normalize_idx] = scaler.transform(x_out[:, normalize_idx])
    return x_out


def _evaluate_tabular(
    model_bundle: Dict[str, object],
    dataset_csv: Path,
    log_misclassified: bool = False,
    max_misclassified: int = 25,
) -> Dict[str, float]:
    df = pd.read_csv(dataset_csv)
    if "shot_type" not in df.columns:
        raise SystemExit("Dataset CSV must include shot_type.")

    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0.0

    x = df[FEATURE_ORDER].astype(float).to_numpy(dtype=np.float32)
    y_true = df["shot_type"].astype(str).to_numpy()

    model = model_bundle["model"]
    x_model = _apply_bundle_preprocess(model_bundle, x)
    y_pred = model.predict(x_model)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("=== Evaluation Metrics ===")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Precision(M)  : {prec:.4f}")
    print(f"Recall(M)     : {rec:.4f}")
    print(f"F1(M)         : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    if log_misclassified:
        mis_path = MODELS_DIR / "misclassified_samples.jsonl"
        count = 0
        with mis_path.open("w", encoding="utf-8") as f:
            for i in range(len(y_true)):
                if y_true[i] == y_pred[i]:
                    continue
                snapshot = {
                    "index": int(i),
                    "actual": str(y_true[i]),
                    "predicted": str(y_pred[i]),
                    "feature_snapshot": {
                        "bat_angle_impact": float(df.iloc[i].get("bat_angle_impact", 0.0)),
                        "follow_through_height_max": float(df.iloc[i].get("follow_through_height_max", 0.0)),
                        "swing_direction_score": float(df.iloc[i].get("swing_direction_score", 0.0)),
                        "player_body_lean": float(df.iloc[i].get("player_body_lean", 0.0)),
                    },
                }
                f.write(json.dumps(snapshot) + "\n")
                count += 1
                if count >= max_misclassified:
                    break
        print(f"Saved misclassified samples: {mis_path} (count={count})")

    return {
        "accuracy": acc,
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
    }


def _collect_labeled_videos(dataset_root: Path) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    for label_dir in sorted([d for d in dataset_root.iterdir() if d.is_dir()]):
        label = label_dir.name.lower()
        for ext in ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"):
            for video in sorted(label_dir.glob(ext)):
                out.append((video, label))
    return out


def _calibrate_from_real_videos(model_bundle: Dict[str, object], dataset_root: Path) -> Dict[str, float]:
    videos = _collect_labeled_videos(dataset_root)
    if not videos:
        print("No labeled videos found for calibration; keeping default thresholds.")
        return {
            "motion_start_floor": 1.5,
            "motion_start_percentile": 40.0,
            "motion_peak_percentile": 85.0,
            "motion_peak_multiplier": 1.8,
            "low_confidence_threshold": float(model_bundle.get("calibration", {}).get("low_confidence_threshold", 60.0)),
            "tie_break_threshold": float(model_bundle.get("calibration", {}).get("tie_break_threshold", 9.0)),
        }

    best = {
        "score": -1.0,
        "motion_start_floor": 1.5,
        "motion_start_percentile": 40.0,
        "motion_peak_percentile": 85.0,
        "motion_peak_multiplier": 1.8,
        "low_confidence_threshold": float(model_bundle.get("calibration", {}).get("low_confidence_threshold", 60.0)),
        "tie_break_threshold": float(model_bundle.get("calibration", {}).get("tie_break_threshold", 9.0)),
    }

    for start_p in (35.0, 40.0, 45.0):
        for peak_p in (80.0, 85.0, 90.0):
            for low_conf in (55.0, 60.0, 65.0):
                for tie_break in (7.0, 9.0, 11.0):
                    pipe = CricketAnalyticsPipeline(sample_rate=2)
                    pipe.motion_start_floor = 1.5
                    pipe.motion_start_percentile = start_p
                    pipe.motion_peak_percentile = peak_p
                    pipe.motion_peak_multiplier = 1.8

                    total = 0
                    correct = 0
                    uncertain = 0
                    try:
                        for video, expected in videos:
                            out = pipe.process_video(video)
                            pred = classify_shot_ml(
                                out.get("key_features", out.get("features_by_frame", [])),
                                model_bundle,
                                low_conf_threshold=low_conf,
                                tie_break_threshold=tie_break,
                            )
                            pred_label = str(pred.get("shot_type", "Uncertain shot")).lower()
                            total += 1
                            if pred_label == "uncertain shot":
                                uncertain += 1
                            elif pred_label == expected:
                                correct += 1
                    finally:
                        pipe.close()

                    if total == 0:
                        continue

                    acc = correct / total
                    uncertain_rate = uncertain / total
                    score = acc - 0.1 * uncertain_rate
                    if score > best["score"]:
                        best = {
                            "score": score,
                            "motion_start_floor": 1.5,
                            "motion_start_percentile": start_p,
                            "motion_peak_percentile": peak_p,
                            "motion_peak_multiplier": 1.8,
                            "low_confidence_threshold": low_conf,
                            "tie_break_threshold": tie_break,
                        }

    print("=== Calibrated Thresholds ===")
    print(json.dumps({k: v for k, v in best.items() if k != "score"}, indent=2))
    return {k: v for k, v in best.items() if k != "score"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latest shot model and optionally calibrate thresholds.")
    parser.add_argument("--dataset-csv", type=str, default="dataset/features_dataset.csv")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--log-misclassified", action="store_true")
    parser.add_argument("--max-misclassified", type=int, default=25)
    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else get_latest_model_path(MODELS_DIR, legacy_path=LEGACY_MODEL)
    if model_path is None or not Path(model_path).exists():
        raise SystemExit("No trained model found. Run train_model.py first.")

    model_bundle = load_classifier(model_path)
    metrics = _evaluate_tabular(
        model_bundle,
        Path(args.dataset_csv),
        log_misclassified=bool(args.log_misclassified),
        max_misclassified=int(args.max_misclassified),
    )

    if args.calibrate:
        calibration = _calibrate_from_real_videos(model_bundle, Path(args.dataset_root))
        model_bundle["calibration"] = calibration
        save_classifier(model_bundle, model_path)
        save_classifier(model_bundle, LEGACY_MODEL)

        calibration_path = MODELS_DIR / "calibration.json"
        with calibration_path.open("w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        print(f"Saved calibration config: {calibration_path}")

    metrics_path = MODELS_DIR / "latest_evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics: {metrics_path}")


if __name__ == "__main__":
    main()
