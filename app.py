from __future__ import annotations

import streamlit as st
st.write("Environment check")

import cv2
st.write("OpenCV version:", cv2.__version__)

from datetime import datetime
import json
import tempfile
import threading
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from analytics.dashboard import (
    build_analysis_frames,
    export_report_csv,
    export_report_pdf,
    plot_joint_angle_trends,
    plot_joint_deviation_radar,
    plot_posture_heatmap,
    plot_similarity_timeline,
    summarize_mistakes_and_suggestions,
)
from analytics.biomechanics_dashboard import (
    plot_bat_arc_angle,
    plot_swing_speed_vs_frame,
    plot_torso_rotation,
)
from analytics.performance_metrics import compute_performance_metrics
from core.bat_tracker import compute_swing_arc
from core.feedback_engine import generate_feedback
from core.frame_pipeline import FramePipeline
from core.mistake_detector import (
    generate_synthetic_mistake_data,
    load_model as load_mistake_model,
    predict_mistakes,
    save_model as save_mistake_model,
    train_mistake_model,
)
from core.pose_comparator import PoseComparator, get_reference_means
from core.realtime_coach import run_realtime_coaching
from core.shot_classifier import (
    generate_synthetic_training_data,
    load_model,
    load_reference_profiles,
    predict_shot,
    save_model,
    train_model,
)
from core.video_processor import extract_frames, load_video
from database.database import init_database
from modules.biomechanics import compute_biomechanics_frame, summarize_biomechanics
from modules.session_manager import save_analysis_session
from modules.shot_classifier import (
    FEATURE_ORDER,
    build_training_matrices,
    classify_shot_ml,
    get_latest_model_path,
    generate_contextual_feedback,
    load_classifier,
    save_classifier,
    save_classifier_versioned,
    train_classifier,
)
from modules.video_processor import CricketAnalyticsPipeline
from utils.visualization import (
    draw_3d_skeleton_projection,
    draw_ball_trajectory,
    draw_bat_trajectory,
    draw_pose_overlay,
)


ROOT_DIR = Path(__file__).parent
REFERENCE_DIR = ROOT_DIR / "reference_data"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "shot_classifier.pkl"
MISTAKE_MODEL_PATH = ROOT_DIR / "assets" / "mistake_detector.pkl"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_VIDEO_SIZE_BYTES = 300 * 1024 * 1024

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _load_reference_profile(shot_type: str) -> Dict[str, object]:
    pro_path = REFERENCE_DIR / f"{shot_type}_pro.json"
    ref_path = pro_path if pro_path.exists() else (REFERENCE_DIR / f"{shot_type}.json")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference dataset missing for shot: {shot_type}")

    with ref_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not any(k in data for k in ("angles", "joint_angles_mean")):
        raise ValueError(f"Invalid reference dataset schema in {ref_path.name}")
    return data


def _init_state() -> None:
    init_database()
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
    if "player_name" not in st.session_state:
        st.session_state.player_name = "Athlete"
    if "current_section" not in st.session_state:
        st.session_state.current_section = "Home"
    if "shot_model" not in st.session_state:
        try:
            st.session_state.shot_model = _load_or_train_shot_model()
        except Exception:
            st.session_state.shot_model = None


def _aggregate_features(frame_results: List[Dict[str, object]]) -> Dict[str, float]:
    if not frame_results:
        return {}
    keys = set()
    for fr in frame_results:
        keys.update(fr["features"].keys())

    result: Dict[str, float] = {}
    for k in sorted(keys):
        vals = [fr["features"].get(k, np.nan) for fr in frame_results]
        arr = np.array(vals, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            result[k] = float(np.mean(arr))
    return result


def _load_or_train_shot_model() -> Dict[str, object]:
    latest_model = get_latest_model_path(MODEL_DIR, legacy_path=MODEL_PATH)
    if latest_model is not None:
        bundle = load_classifier(latest_model)
        existing_order = list(bundle.get("feature_order", []))
        sklearn_version = str(bundle.get("sklearn_version", ""))
        if existing_order and set(FEATURE_ORDER).issubset(set(existing_order)) and sklearn_version == sklearn.__version__:
            return bundle

    rng = np.random.default_rng(42)
    priors = {
        "defensive": {
            "bat_swing_arc": 80, "bat_angle": 68, "bat_follow_through_height": 0.08, "follow_through_height": 0.05,
            "shoulder_rotation": 8, "elbow_angle": 150, "body_lean": 8, "wrist_trajectory": 2.8,
            "knee_bend": 155, "torso_tilt": 9, "head_position": 20, "bat_velocity": 4, "ball_direction": 15,
            "pose_visibility": 0.82, "motion_phase": 1.0,
        },
        "drive": {
            "bat_swing_arc": 130, "bat_angle": 42, "bat_follow_through_height": 0.14, "follow_through_height": 0.12,
            "shoulder_rotation": 14, "elbow_angle": 140, "body_lean": 11, "wrist_trajectory": 4.8,
            "knee_bend": 145, "torso_tilt": 12, "head_position": 26, "bat_velocity": 7, "ball_direction": 20,
            "pose_visibility": 0.80, "motion_phase": 1.5,
        },
        "lofted": {
            "bat_swing_arc": 220, "bat_angle": 18, "bat_follow_through_height": 0.24, "follow_through_height": 0.20,
            "shoulder_rotation": 18, "elbow_angle": 128, "body_lean": 16, "wrist_trajectory": 6.5,
            "knee_bend": 140, "torso_tilt": 14, "head_position": 34, "bat_velocity": 11, "ball_direction": 35,
            "pose_visibility": 0.78, "motion_phase": 2.0,
        },
        "pull": {
            "bat_swing_arc": 170, "bat_angle": 5, "bat_follow_through_height": 0.10, "follow_through_height": 0.09,
            "shoulder_rotation": 22, "elbow_angle": 132, "body_lean": 14, "wrist_trajectory": 5.6,
            "knee_bend": 135, "torso_tilt": 11, "head_position": 30, "bat_velocity": 10, "ball_direction": -5,
            "pose_visibility": 0.79, "motion_phase": 1.8,
        },
        "cut": {
            "bat_swing_arc": 160, "bat_angle": -8, "bat_follow_through_height": 0.09, "follow_through_height": 0.08,
            "shoulder_rotation": 24, "elbow_angle": 130, "body_lean": 13, "wrist_trajectory": 5.3,
            "knee_bend": 138, "torso_tilt": 10, "head_position": 28, "bat_velocity": 9, "ball_direction": -20,
            "pose_visibility": 0.78, "motion_phase": 1.8,
        },
        "sweep": {
            "bat_swing_arc": 145, "bat_angle": -28, "bat_follow_through_height": 0.07, "follow_through_height": 0.06,
            "shoulder_rotation": 16, "elbow_angle": 126, "body_lean": 15, "wrist_trajectory": 4.1,
            "knee_bend": 122, "torso_tilt": 7, "head_position": 24, "bat_velocity": 8, "ball_direction": -40,
            "pose_visibility": 0.77, "motion_phase": 1.6,
        },
    }
    samples = []
    for label, base in priors.items():
        for _ in range(280):
            row = {k: 0.0 for k in FEATURE_ORDER}
            for key, value in base.items():
                scale = 0.05 * max(1.0, abs(float(value)))
                row[key] = float(rng.normal(loc=float(value), scale=scale))

            for key in ("elbow_angle", "shoulder_rotation", "wrist_trajectory", "body_lean"):
                row[f"{key}_mean"] = row[key]
                row[f"{key}_max"] = row[key] * float(rng.uniform(1.02, 1.25))
                row[f"{key}_velocity"] = abs(row[key]) * float(rng.uniform(0.02, 0.08))

            row["pose_visibility"] = float(np.clip(row["pose_visibility"], 0.2, 1.0))
            row["motion_phase"] = float(np.clip(row["motion_phase"], 0.0, 2.0))
            samples.append((row, label))

    x, y = build_training_matrices(samples)
    bundle = train_classifier(x, y)
    save_classifier(bundle, MODEL_PATH)
    save_classifier_versioned(bundle, MODEL_DIR)
    return bundle


def _load_or_train_mistake_model() -> Dict[str, object]:
    if MISTAKE_MODEL_PATH.exists():
        return load_mistake_model(MISTAKE_MODEL_PATH)

    x, y = generate_synthetic_mistake_data(samples=2600, random_state=42)
    bundle = train_mistake_model(feature_vectors=x, labels=y, random_state=42)
    save_mistake_model(bundle, MISTAKE_MODEL_PATH)
    return bundle


def _plot_shot_probabilities(probability_map: Dict[str, float]):
    vals = [float(v) for v in probability_map.values()]
    scale = 100.0 if (vals and max(vals) <= 1.0) else 1.0
    prob_df = pd.DataFrame(
        {
            "Shot": list(probability_map.keys()),
            "Probability": [float(v) * scale for v in probability_map.values()],
        }
    ).sort_values("Probability", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(prob_df["Shot"], prob_df["Probability"], color=["#00C2FF", "#22C55E", "#F59E0B", "#EF4444"])
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Predicted Shot Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, prob_df["Probability"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def _plot_player_vs_ideal(avg_features: Dict[str, float], reference_features: Dict[str, float]):
    keys = [k for k in sorted(reference_features.keys()) if k in avg_features]
    if not keys:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No comparison data", ha="center", va="center")
        ax.axis("off")
        return fig

    player_vals = [float(avg_features[k]) for k in keys]
    ideal_vals = [float(reference_features[k]) for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    y = np.arange(len(keys))

    axes[0].barh(y, player_vals, color="#38BDF8")
    axes[0].set_title("Player Pose Profile")
    axes[0].set_xlim(0, 180)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([k.replace("_", " ") for k in keys], fontsize=8)
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    axes[1].barh(y, ideal_vals, color="#22C55E")
    axes[1].set_title("Ideal Pose Profile")
    axes[1].set_xlim(0, 180)
    axes[1].set_yticks(y)
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Player vs Ideal Pose (Side-by-Side)")
    fig.tight_layout()
    return fig


def _compute_score_card(metrics: Dict[str, object]) -> Dict[str, float]:
    technique = float(metrics.get("posture_accuracy_score", 0.0))
    balance = float(metrics.get("joint_stability", 0.0))
    consistency = float(metrics.get("consistency_across_frames", 0.0))
    overall = 0.4 * technique + 0.3 * balance + 0.3 * consistency
    return {
        "technique": round(technique, 2),
        "balance": round(balance, 2),
        "consistency": round(consistency, 2),
        "overall": round(overall, 2),
    }


def _build_bat_tracking_from_path(bat_tips: List[tuple[int, int]]) -> Dict[str, object]:
    if len(bat_tips) < 2:
        return {
            "bat_path_coordinates": bat_tips,
            "smoothed_bat_path": bat_tips,
            "swing_speed_per_frame": [],
            "swing_speed": 0.0,
            "swing_arc_angle": 0.0,
        }

    speeds = [
        float(np.linalg.norm(np.array(bat_tips[i]) - np.array(bat_tips[i - 1])))
        for i in range(1, len(bat_tips))
    ]
    return {
        "bat_path_coordinates": bat_tips,
        "smoothed_bat_path": bat_tips,
        "swing_speed_per_frame": speeds,
        "swing_speed": round(float(np.mean(speeds)) if speeds else 0.0, 3),
        "swing_arc_angle": round(float(compute_swing_arc(bat_tips)), 3),
    }


def _build_ball_tracking_from_path(ball_centers: List[tuple[int, int]], bat_path: List[tuple[int, int]]) -> Dict[str, object]:
    if not ball_centers:
        return {
            "ball_trajectory": [],
            "impact_point_estimate": None,
            "bat_ball_alignment_score": 0.0,
        }

    impact = ball_centers[-1]
    score = 0.0
    if bat_path:
        bat_arr = np.array(bat_path, dtype=np.float32)
        ball_arr = np.array(ball_centers, dtype=np.float32)
        dists = np.linalg.norm(ball_arr[:, None, :] - bat_arr[None, :, :], axis=2)
        min_dist = float(np.min(dists)) if dists.size else 999.0
        score = float(np.clip(100.0 - (min_dist / 200.0) * 100.0, 0.0, 100.0))
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        impact = tuple(ball_arr[min_idx[0]].astype(int).tolist())

    return {
        "ball_trajectory": ball_centers,
        "impact_point_estimate": impact,
        "bat_ball_alignment_score": round(score, 2),
    }


def _empty_analysis_result(player_name: str = "Athlete") -> Dict[str, object]:
    empty_df = pd.DataFrame()
    return {
        "status": "error",
        "message": "Unable to analyze video properly",
        "stage": "unknown",
        "player_name": player_name,
        "fps": 0.0,
        "detected_shot": "Uncertain shot",
        "confidence_score": 0.0,
        "shot_prediction": {
            "shot_type": "Uncertain shot",
            "confidence_score": 0.0,
            "probabilities": {},
            "insight": "Insufficient data for analysis",
            "low_confidence_warning": "Low confidence due to poor visibility or unstable camera",
        },
        "low_confidence_warning": "Low confidence due to poor visibility or unstable camera",
        "metrics": {
            "status": "insufficient_data",
            "message": "Insufficient data for analysis",
            "posture_accuracy_score": 0.0,
            "joint_stability": 0.0,
            "consistency_across_frames": 0.0,
            "swing_efficiency": 0.0,
            "bat_plane_consistency": 0.0,
            "torso_rotation_power": 0.0,
            "impact_alignment_score": 0.0,
            "advanced_performance_score": 0.0,
            "frames_analyzed": 0,
            "per_joint_variability": {},
        },
        "score_card": {"technique": 0.0, "balance": 0.0, "consistency": 0.0, "overall": 0.0},
        "similarity_series": [],
        "avg_features": {},
        "reference_features": {},
        "final_feedback": {
            "summary": "Insufficient data for analysis",
            "tips": [
                "Use a side-on camera angle.",
                "Ensure full body and bat are visible.",
                "Record in better lighting with steady camera.",
            ],
        },
        "detected_mistakes": [],
        "display_frames": [],
        "features_df": empty_df,
        "similarity_df": empty_df,
        "deviation_df": empty_df,
        "frame_table_df": empty_df,
        "mistake_summary": {"top_mistakes": [], "suggestions": ["Collect clearer video for full analysis."]},
        "biomechanics": {
            "bat": {
                "bat_path_coordinates": [],
                "smoothed_bat_path": [],
                "swing_speed_per_frame": [],
                "swing_speed": 0.0,
                "swing_arc_angle": 0.0,
            },
            "ball": {
                "ball_trajectory": [],
                "impact_point_estimate": None,
                "bat_ball_alignment_score": 0.0,
            },
            "pose3d": {"torso_twist_series": [], "bat_swing_plane_series": []},
        },
        "advanced_projection_frames": [],
        "last_frame_rgb": None,
        "analysis_stats": {},
    }


def _fallback_analysis_result(
    stage: str,
    message: str,
    player_name: str,
    partial: Dict[str, object] | None = None,
) -> Dict[str, object]:
    payload = _empty_analysis_result(player_name=player_name)
    payload["stage"] = stage
    payload["message"] = message
    if partial:
        payload.update(partial)
        payload["status"] = "partial" if partial.get("detected_shot") else "error"
    return payload


def _validate_uploaded_video(uploaded_video) -> tuple[bool, str]:
    if uploaded_video is None:
        return False, "No video selected."

    name = str(getattr(uploaded_video, "name", "")).strip()
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        return False, "Unsupported video format. Please upload mp4, mov, avi, or mkv."

    size = int(getattr(uploaded_video, "size", 0) or 0)
    if size <= 0:
        return False, "Uploaded file appears empty."
    if size > MAX_VIDEO_SIZE_BYTES:
        return False, "Video file is too large. Please upload a file under 300 MB."

    return True, ""


def _probe_video_file(video_path: str) -> tuple[bool, str]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return False, "Unable to open uploaded video."
        ok, _ = cap.read()
        cap.release()
        if not ok:
            return False, "Video has no readable frames."
        return True, ""
    except Exception:
        return False, "Uploaded file is not a valid video."


def _run_video_analysis(uploaded_video, sample_rate: int, user_id: int, player_name: str) -> Dict[str, object] | None:
    ok, validation_message = _validate_uploaded_video(uploaded_video)
    if not ok:
        logger.info("Upload validation failed: %s", validation_message)
        st.error(validation_message)
        return _fallback_analysis_result("input_validation", validation_message, player_name)

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_video.read())
            temp_path = temp_video.name
    except Exception as exc:
        logger.exception("Stage upload_write failed: %s", exc)
        return _fallback_analysis_result("upload_write", "Unable to read uploaded file.", player_name)

    can_open, probe_message = _probe_video_file(temp_path)
    if not can_open:
        logger.info("Video probe failed: %s", probe_message)
        st.error(probe_message)
        return _fallback_analysis_result("video_probe", probe_message, player_name)

    try:
        shot_model = st.session_state.get("shot_model") or _load_or_train_shot_model()
        st.session_state.shot_model = shot_model
    except Exception as exc:
        logger.exception("Stage model_load failed: %s", exc)
        return _fallback_analysis_result("model_load", "Unable to load analysis model.", player_name)

    out: Dict[str, object] = {}
    pipeline = None
    try:
        pipeline = CricketAnalyticsPipeline(sample_rate=sample_rate, target_size=(854, 480))
        out = pipeline.process_video(temp_path)
    except Exception as exc:
        logger.exception("Stage pipeline_process failed: %s", exc)
        return _fallback_analysis_result("pipeline_process", "Unable to analyze video properly", player_name)
    finally:
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass

    frame_data = out.get("frame_data", []) if isinstance(out, dict) else []
    features_by_frame = out.get("features_by_frame", []) if isinstance(out, dict) else []
    analysis_stats = out.get("analysis_stats", {}) if isinstance(out, dict) else {}
    valid_pose_frames = int(analysis_stats.get("valid_pose_frames", 0) or 0)
    logger.info(
        "Pipeline stats: total_frames=%s frames_processed=%s valid_pose=%s fallback_impact=%s quality_flags=%s",
        analysis_stats.get("frames_total", 0),
        len(frame_data),
        valid_pose_frames,
        analysis_stats.get("fallback_impact_used", False),
        analysis_stats.get("quality_flags", []),
    )

    if len(frame_data) == 0:
        message = "Insufficient data for analysis"
        st.warning("⚠️ Unable to process this video. Try a clearer recording.")
        return _fallback_analysis_result(
            "frame_validation",
            message,
            player_name,
            partial={"analysis_stats": analysis_stats},
        )

    try:
        shot_prediction = classify_shot_ml(out.get("key_features", features_by_frame), shot_model, window_size=15)
    except Exception as exc:
        logger.exception("Stage classification failed: %s", exc)
        shot_prediction = {
            "shot_type": "Uncertain shot",
            "confidence_score": 0.0,
            "probabilities": {},
            "insight": "Unable to classify shot from available frames.",
            "low_confidence_warning": "Low confidence due to poor visibility or unstable camera",
        }

    raw_shot = str(shot_prediction.get("shot_type", "Uncertain shot"))
    confidence_score = float(shot_prediction.get("confidence_score", 0.0))
    low_conf_warning = str(shot_prediction.get("low_confidence_warning", ""))

    reference_key_map = {
        "defensive": "defense",
        "drive": "cover_drive",
        "lofted": "straight_drive",
        "pull": "pull_shot",
        "cut": "pull_shot",
        "sweep": "defense",
    }
    reference_shot = reference_key_map.get(raw_shot.lower(), "defense")

    try:
        reference_profile = _load_reference_profile(reference_shot)
        reference_features = get_reference_means(reference_profile)
        comparator = PoseComparator()
    except Exception as exc:
        logger.warning("Stage reference_profile failed: %s", exc)
        reference_profile = {}
        reference_features = {}
        comparator = None

    frame_results: List[Dict[str, object]] = []
    biomechanics_series: List[Dict[str, float]] = []
    display_frames: List[np.ndarray] = []
    advanced_projection_frames: List[np.ndarray] = []

    for i, row in enumerate(frame_data, start=1):
        features = row.get("features", {}) if isinstance(row, dict) else {}
        if not isinstance(features, dict) or not features:
            continue

        comparison = {"similarity_score": 0.0, "joint_errors": {}}
        if comparator is not None and isinstance(reference_profile, dict):
            try:
                comparison = comparator.compare(features, reference_profile)
            except Exception:
                comparison = {"similarity_score": 0.0, "joint_errors": {}}

        frame_results.append(
            {
                "frame_index": int(row.get("frame_index", i)),
                "features": features,
                "similarity_score": float(comparison.get("similarity_score", 0.0)),
                "comparison": comparison,
            }
        )

        pose_metrics = row.get("pose_features", {}) if isinstance(row, dict) else {}
        if isinstance(pose_metrics, dict) and pose_metrics:
            biomechanics_series.append(
                {
                    "elbow_angle": float(pose_metrics.get("elbow_angle", 0.0)),
                    "knee_bend_angle": float(pose_metrics.get("knee_bend", 0.0)),
                    "spine_tilt": float(pose_metrics.get("torso_tilt", 0.0)),
                    "head_position_relative_ball_line": float(pose_metrics.get("head_alignment", 0.0)),
                }
            )

        if len(display_frames) < 8 and i % max(1, len(frame_data) // 8) == 0:
            try:
                frame_bgr = row.get("frame_bgr", None)
                if frame_bgr is None:
                    continue
                overlay = draw_pose_overlay(
                    frame_bgr=frame_bgr.copy(),
                    keypoints=row.get("keypoints", {}),
                    joint_errors=comparison.get("joint_errors", {}),
                    angles=row.get("pose_features", {}),
                )
                det = row.get("detection", {})
                for label, color in (("bat_box", (0, 255, 255)), ("ball_box", (0, 165, 255)), ("player_box", (255, 128, 0))):
                    box = det.get(label) if isinstance(det, dict) else None
                    if box is not None:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, f"Shot: {raw_shot.replace('_', ' ').title()}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(overlay, f"Confidence: {confidence_score:.1f}%", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (76, 201, 240), 2, cv2.LINE_AA)
                display_frames.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                advanced_projection_frames.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            except Exception:
                continue

    bat_path = out.get("bat_path", []) if isinstance(out, dict) else []
    ball_path = out.get("ball_path", []) if isinstance(out, dict) else []
    bat_tracking = _build_bat_tracking_from_path(bat_path if isinstance(bat_path, list) else [])
    ball_tracking = _build_ball_tracking_from_path(ball_path if isinstance(ball_path, list) else [], bat_tracking.get("smoothed_bat_path", []))

    biomechanics_data = {
        "bat": {
            **bat_tracking,
            "swing_speed_per_frame": [float(f.get("bat_velocity", 0.0)) for f in features_by_frame if isinstance(f, dict)],
        },
        "ball": ball_tracking,
        "pose3d": {
            "torso_twist_series": [float(f.get("torso_tilt", 0.0)) for f in features_by_frame if isinstance(f, dict)],
            "bat_swing_plane_series": [float(f.get("bat_angle", 0.0)) for f in features_by_frame if isinstance(f, dict)],
        },
    }

    metrics = compute_performance_metrics(frame_results, biomechanics_data=biomechanics_data)
    bio_scores = summarize_biomechanics(biomechanics_series)
    similarity_series = [float(fr.get("similarity_score", 0.0)) for fr in frame_results]
    avg_features = _aggregate_features(frame_results)
    final_feedback = {
        "summary": generate_contextual_feedback(shot_prediction, bio_scores, features_by_frame),
        "tips": [
            "Keep head alignment stable over the ball line.",
            "Maintain knee flexion and avoid early body opening.",
            "Control bat follow-through to improve repeatability.",
        ],
    }

    try:
        features_df, similarity_df, deviation_df, frame_table_df = build_analysis_frames(frame_results, reference_features)
        mistake_summary = summarize_mistakes_and_suggestions(deviation_df)
    except Exception as exc:
        logger.warning("Stage dataframe_build failed: %s", exc)
        features_df = pd.DataFrame()
        similarity_df = pd.DataFrame()
        deviation_df = pd.DataFrame()
        frame_table_df = pd.DataFrame()
        mistake_summary = {"top_mistakes": [], "suggestions": ["Insufficient data for detailed mistake summary."]}

    score_card = {
        "technique": float(bio_scores.get("technique_score", 0.0)),
        "balance": float(bio_scores.get("balance_score", 0.0)),
        "consistency": float(bio_scores.get("consistency_score", 0.0)),
        "overall": round(
            0.4 * float(bio_scores.get("technique_score", 0.0))
            + 0.3 * float(bio_scores.get("balance_score", 0.0))
            + 0.3 * float(bio_scores.get("consistency_score", 0.0)),
            2,
        ),
    }

    try:
        save_analysis_session(
            user_id=user_id,
            video_name=getattr(uploaded_video, "name", "uploaded_video.mp4"),
            shot_type=raw_shot,
            confidence=confidence_score,
            technique_score=score_card["technique"],
            balance_score=score_card["balance"],
            consistency_score=score_card["consistency"],
        )
    except Exception:
        pass

    session_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player": player_name,
        "shot": raw_shot,
        "confidence": round(confidence_score, 2),
        "overall_score": score_card["overall"],
    }
    st.session_state.session_history.append(session_entry)

    status = "ok"
    message = "Analysis complete"
    if len(frame_results) < 4 or metrics.get("status") != "ok":
        status = "partial"
        message = "Insufficient data for analysis"

    result = {
        "status": status,
        "message": message,
        "stage": "complete",
        "player_name": player_name,
        "fps": float(out.get("fps", 0.0)) if isinstance(out, dict) else 0.0,
        "detected_shot": raw_shot,
        "confidence_score": confidence_score,
        "shot_prediction": shot_prediction,
        "low_confidence_warning": low_conf_warning,
        "metrics": metrics,
        "score_card": score_card,
        "similarity_series": similarity_series,
        "avg_features": avg_features,
        "reference_features": reference_features,
        "final_feedback": final_feedback,
        "detected_mistakes": [],
        "display_frames": display_frames,
        "features_df": features_df,
        "similarity_df": similarity_df,
        "deviation_df": deviation_df,
        "frame_table_df": frame_table_df,
        "mistake_summary": mistake_summary,
        "biomechanics": biomechanics_data,
        "advanced_projection_frames": advanced_projection_frames,
        "last_frame_rgb": display_frames[-1] if display_frames else None,
        "analysis_stats": analysis_stats,
    }

    if low_conf_warning or status == "partial":
        st.warning("⚠️ Analysis completed with limited data. Results may be less accurate.")

    return result


def _render_score_card(score_card: Dict[str, float]) -> None:
    st.markdown("### Session Score Card")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Technique Score", f"{score_card['technique']}%")
    c2.metric("Balance Score", f"{score_card['balance']}%")
    c3.metric("Consistency Score", f"{score_card['consistency']}%")
    c4.metric("Overall Score", f"{score_card['overall']}%")


def _render_coaching_view(analysis: Dict[str, object]) -> None:
    top1, top2 = st.columns(2)
    top1.metric("Detected Shot Type", str(analysis["detected_shot"]).replace("_", " ").title())
    top2.metric("Confidence Score", f"{float(analysis['confidence_score']):.2f}%")

    st.subheader("Shot Classification Probabilities")
    fig_prob = _plot_shot_probabilities(analysis["shot_prediction"]["probabilities"])
    st.pyplot(fig_prob, use_container_width=True)
    plt.close(fig_prob)

    _render_score_card(analysis["score_card"])

    st.subheader("Coaching Feedback")
    final_feedback = analysis["final_feedback"]
    st.write(final_feedback["summary"])
    for tip in final_feedback["tips"]:
        st.markdown(f"- {tip}")

    st.subheader("Detected Technique Issues")
    detected_mistakes = analysis["detected_mistakes"]
    if detected_mistakes:
        for issue in detected_mistakes:
            st.markdown(f"- {issue['display_name']} – Confidence {issue['confidence']:.0f}%")
    else:
        st.markdown("- No major technique issues detected.")

    st.subheader("Player vs Ideal Pose")
    compare_left, compare_right = st.columns(2)
    with compare_left:
        st.markdown("**Player Pose Snapshots**")
        display_frames = analysis["display_frames"]
        if display_frames:
            st.image(display_frames[0], caption="Detected Player Pose", use_container_width=True)
    with compare_right:
        fig_compare = _plot_player_vs_ideal(analysis["avg_features"], analysis["reference_features"])
        st.pyplot(fig_compare, use_container_width=True)
        plt.close(fig_compare)

    st.subheader("Pose Skeleton Snapshots")
    display_frames = analysis["display_frames"]
    if display_frames:
        img_cols = st.columns(4)
        for i, img in enumerate(display_frames):
            with img_cols[i % 4]:
                st.image(img, caption=f"Frame sample {i + 1}", use_container_width=True)

    st.caption(f"Processed at sampled frame-rate. Original FPS: {float(analysis['fps']):.2f}")


def _render_report_view(analysis: Dict[str, object]) -> None:
    fig_angles = plot_joint_angle_trends(analysis["features_df"])
    fig_similarity = plot_similarity_timeline(analysis["similarity_df"])
    fig_heatmap = plot_posture_heatmap(analysis["deviation_df"])
    fig_radar = plot_joint_deviation_radar(analysis["deviation_df"])

    csv_bytes = export_report_csv(analysis["frame_table_df"])
    pdf_bytes = export_report_pdf(
        overall_posture_score=float(analysis["metrics"]["posture_accuracy_score"]),
        detected_shot=str(analysis["detected_shot"]),
        confidence_score=float(analysis["confidence_score"]),
        top_mistakes=analysis["mistake_summary"]["top_mistakes"],
        suggestions=analysis["mistake_summary"]["suggestions"],
        figures=[fig_angles, fig_similarity, fig_heatmap, fig_radar],
    )

    st.subheader("Professional Performance Report")
    r1, r2, r3 = st.columns(3)
    r1.metric("Overall Posture Score", f"{analysis['metrics']['posture_accuracy_score']}%")
    r2.metric("Detected Shot", str(analysis["detected_shot"]).replace("_", " ").title())
    r3.metric("Model Confidence", f"{float(analysis['confidence_score']):.2f}%")

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Swing Efficiency", f"{analysis['metrics'].get('swing_efficiency', 0.0):.2f}%")
    a2.metric("Bat Plane Consistency", f"{analysis['metrics'].get('bat_plane_consistency', 0.0):.2f}%")
    a3.metric("Torso Rotation Power", f"{analysis['metrics'].get('torso_rotation_power', 0.0):.2f}%")
    a4.metric("Impact Alignment", f"{analysis['metrics'].get('impact_alignment_score', 0.0):.2f}%")
    a5.metric("Advanced Score", f"{analysis['metrics'].get('advanced_performance_score', 0.0):.2f}%")

    st.markdown("**Top Mistakes**")
    if analysis["mistake_summary"]["top_mistakes"]:
        for joint, dev in analysis["mistake_summary"]["top_mistakes"][:5]:
            st.markdown(f"- {joint}: {dev:.2f}° avg deviation")
    else:
        st.markdown("- No major mistakes detected.")

    st.markdown("**Improvement Suggestions**")
    for suggestion in analysis["mistake_summary"]["suggestions"]:
        st.markdown(f"- {suggestion}")

    st.subheader("Charts")
    st.pyplot(fig_angles, use_container_width=True)
    st.pyplot(fig_similarity, use_container_width=True)
    st.pyplot(fig_heatmap, use_container_width=True)
    st.pyplot(fig_radar, use_container_width=True)

    st.subheader("Frame-by-Frame Analysis")
    st.dataframe(analysis["frame_table_df"], use_container_width=True, hide_index=True)

    ex1, ex2 = st.columns(2)
    with ex1:
        st.download_button(
            label="Export CSV Report",
            data=csv_bytes,
            file_name="smartcoach_performance_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            label="Export PDF Report",
            data=pdf_bytes,
            file_name="smartcoach_performance_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    plt.close(fig_angles)
    plt.close(fig_similarity)
    plt.close(fig_heatmap)
    plt.close(fig_radar)


def _render_advanced_biomechanics_view(analysis: Dict[str, object]) -> None:
    st.subheader("Advanced Biomechanics")

    biomechanics = analysis.get("biomechanics", {})
    bat_data = biomechanics.get("bat", {})
    ball_data = biomechanics.get("ball", {})
    pose3d_data = biomechanics.get("pose3d", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Swing Efficiency", f"{analysis['metrics'].get('swing_efficiency', 0.0):.2f}%")
    m2.metric("Bat Plane Consistency", f"{analysis['metrics'].get('bat_plane_consistency', 0.0):.2f}%")
    m3.metric("Torso Rotation Power", f"{analysis['metrics'].get('torso_rotation_power', 0.0):.2f}%")
    m4.metric("Impact Alignment", f"{analysis['metrics'].get('impact_alignment_score', 0.0):.2f}%")
    m5.metric("Advanced Score", f"{analysis['metrics'].get('advanced_performance_score', 0.0):.2f}%")

    base_rgb = analysis.get("last_frame_rgb")
    if base_rgb is not None:
        frame_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = draw_bat_trajectory(frame_bgr, bat_data.get("smoothed_bat_path", []))
        frame_bgr = draw_ball_trajectory(
            frame_bgr,
            ball_data.get("ball_trajectory", []),
            impact_point=ball_data.get("impact_point_estimate"),
        )
        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Bat + Ball Trajectory with Impact Zone", use_container_width=True)

    st.markdown(
        f"**Impact Alignment Score:** {float(ball_data.get('bat_ball_alignment_score', 0.0)):.2f}%  |  "
        f"**Swing Arc Angle:** {float(bat_data.get('swing_arc_angle', 0.0)):.2f}°"
    )

    st.subheader("3D Skeleton Projection")
    projection_frames = analysis.get("advanced_projection_frames", [])
    if projection_frames:
        cols = st.columns(4)
        for i, img in enumerate(projection_frames[:8]):
            with cols[i % 4]:
                st.image(img, caption=f"3D Projection {i + 1}", use_container_width=True)
    else:
        st.info("3D projection frames not available for this session.")

    st.subheader("Biomechanics Graphs")
    fig_speed = plot_swing_speed_vs_frame(bat_data.get("swing_speed_per_frame", []))
    fig_arc = plot_bat_arc_angle(float(bat_data.get("swing_arc_angle", 0.0)))
    fig_torso = plot_torso_rotation([float(v) for v in pose3d_data.get("torso_twist_series", [])])

    st.pyplot(fig_speed, use_container_width=True)
    st.pyplot(fig_arc, use_container_width=True)
    st.pyplot(fig_torso, use_container_width=True)

    plt.close(fig_speed)
    plt.close(fig_arc)
    plt.close(fig_torso)


def main() -> None:
    st.set_page_config(page_title="SmartCoach AI", layout="wide")
    _init_state()

    st.markdown(
        """
        <style>
            :root {
                --primary: #4CC9F0;
                --secondary: #4361EE;
                --surface: rgba(255,255,255,0.05);
                --bg: #0b1320;
            }

            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            .stApp {
                background: linear-gradient(120deg, #0b1320, #14213d, #2b2d6e, #1a2b4c);
                background-size: 240% 240%;
                animation: gradientShift 24s ease infinite;
                color: #E5E7EB;
                font-family: "Inter", "Segoe UI", Arial, sans-serif;
            }
            .block-container {padding-top: 1.0rem; padding-bottom: 1.2rem; max-width: 1320px;}
            section[data-testid="stSidebar"] {
                background: rgba(7, 12, 23, 0.92);
                border-right: 1px solid rgba(255,255,255,0.08);
                padding-top: 1rem;
            }
            section[data-testid="stSidebar"] hr {
                border-color: rgba(255,255,255,0.12);
            }

            .stMetric {
                background: var(--surface);
                backdrop-filter: blur(12px);
                border-radius: 12px;
                padding: 10px;
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 10px 30px rgba(0,0,0,0.35);
            }
            [data-testid="stMetricLabel"] {
                justify-content: center;
                font-size: 0.88rem;
                color: #cdd6e3;
                text-transform: uppercase;
                letter-spacing: 0.03em;
            }
            [data-testid="stMetricValue"] {
                justify-content: center;
                font-size: 1.7rem;
                font-weight: 700;
                color: var(--primary);
            }

            .action-card {
                background: var(--surface);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 14px;
                padding: 18px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.35);
                min-height: 145px;
                transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            }
            .action-card:hover {
                transform: translateY(-4px);
                border-color: rgba(76, 201, 240, 0.5);
                box-shadow: 0 14px 34px rgba(67, 97, 238, 0.22);
            }
            .action-card h4 {margin: 0 0 8px 0; font-size: 1.1rem; color: #F3F4F6;}
            .action-card p {margin: 0; color: #CBD5E1; font-size: 0.93rem; line-height: 1.4;}

            .hero-wrap {
                padding: 0.3rem 0 0.5rem 0;
            }
            .hero-title {
                margin: 0;
                font-size: 2.35rem;
                font-weight: 700;
                color: #F8FAFC;
            }
            .hero-subtitle {
                margin: 0.2rem 0 0.25rem 0;
                font-size: 1.08rem;
                color: #D6DEEA;
                font-weight: 500;
            }
            .hero-caption {
                margin: 0.15rem 0 0 0;
                font-size: 0.95rem;
                color: #BAC7DA;
            }
            .hero-line {
                margin-top: 0.9rem;
                width: 100%;
                height: 1px;
                background: linear-gradient(90deg, rgba(76,201,240,0.65), rgba(67,97,238,0.25), rgba(255,255,255,0.03));
            }

            .stButton > button {
                border-radius: 10px;
                border: 1px solid rgba(76, 201, 240, 0.42);
                background: linear-gradient(90deg, rgba(67,97,238,0.30), rgba(76,201,240,0.28));
                color: #EAF3FF;
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            }
            .stButton > button:hover {
                transform: scale(1.015);
                border-color: var(--primary);
                box-shadow: 0 0 0 1px rgba(76,201,240,0.2), 0 8px 18px rgba(76,201,240,0.16);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-wrap">
            <h1 class="hero-title">SmartCoach AI</h1>
            <div class="hero-subtitle">Cricket Performance Analytics Platform</div>
            <div class="hero-caption">Computer vision powered cricket technique analysis and biomechanics insights.</div>
            <div class="hero-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    analysis = st.session_state.last_analysis
    top_score = analysis["score_card"] if analysis else {"technique": 0.0, "balance": 0.0, "consistency": 0.0}
    st.subheader("Session Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Technique Score", f"{float(top_score['technique']):.2f}%")
    s2.metric("Balance Score", f"{float(top_score['balance']):.2f}%")
    s3.metric("Consistency Score", f"{float(top_score['consistency']):.2f}%")
    st.divider()

    with st.sidebar:
        player_name = st.text_input("Player Name", value=st.session_state.player_name)
        st.session_state.player_name = player_name

        options = ["Home", "Video Analysis", "Live Coaching", "Performance Reports", "Advanced Biomechanics"]
        current = st.session_state.current_section if st.session_state.current_section in options else "Home"
        section = st.radio(
            "Navigation",
            options=options,
            index=options.index(current),
        )
        st.session_state.current_section = section

        st.divider()
        st.markdown("### Session Statistics")
        history = st.session_state.session_history
        total_sessions = len(history)
        avg_overall = float(np.mean([float(h.get("overall_score", 0.0)) for h in history])) if history else 0.0
        st.metric("Sessions", str(total_sessions))
        st.metric("Avg Overall", f"{avg_overall:.1f}%")

    if section == "Home":
        st.markdown(
            "SmartCoach AI is a computer vision–based cricket analytics platform that evaluates "
            "player technique using pose estimation and biomechanics analysis."
        )
        st.divider()

        row1 = st.columns(2)
        with row1[0]:
            st.markdown(
                '<div class="action-card"><h4>Video Analysis</h4><p>Analyze uploaded clips with pose tracking, shot classification, mistake detection, and coaching feedback.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Open Video Analysis", use_container_width=True):
                st.session_state.current_section = "Video Analysis"
                st.rerun()
        with row1[1]:
            st.markdown(
                '<div class="action-card"><h4>Live Coaching</h4><p>Run real-time webcam analysis with on-frame technique guidance and immediate posture correction cues.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Open Live Coaching", use_container_width=True):
                st.session_state.current_section = "Live Coaching"
                st.rerun()

        row2 = st.columns(2)
        with row2[0]:
            st.markdown(
                '<div class="action-card"><h4>Performance Reports</h4><p>Review full analytical breakdowns, charts, and downloadable session reports for coaching workflows.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Open Performance Reports", use_container_width=True):
                st.session_state.current_section = "Performance Reports"
                st.rerun()
        with row2[1]:
            st.markdown(
                '<div class="action-card"><h4>Advanced Biomechanics</h4><p>Inspect bat/ball trajectories, 3D projection, swing mechanics, and impact alignment metrics.</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Open Advanced Biomechanics", use_container_width=True):
                st.session_state.current_section = "Advanced Biomechanics"
                st.rerun()

        if st.session_state.last_analysis:
            st.divider()
            _render_score_card(st.session_state.last_analysis["score_card"])

    elif section == "Video Analysis":
        st.subheader("Video Analysis")
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_video = st.file_uploader("Upload cricket video", type=["mp4", "avi", "mov", "mkv"])
        with col2:
            sample_rate = st.slider("Frame sampling", min_value=1, max_value=6, value=2, step=1)

        process_clicked = st.button("Process Video", type="primary", use_container_width=True)

        if process_clicked and uploaded_video is not None:
            try:
                analysis = _run_video_analysis(uploaded_video, sample_rate, 1, player_name)
            except Exception as exc:
                logger.exception("Unhandled analysis error: %s", exc)
                st.warning("⚠️ Unable to process this video. Try a clearer recording.")
                analysis = _fallback_analysis_result(
                    stage="unhandled",
                    message="Unable to analyze video properly",
                    player_name=player_name,
                )

            if analysis is not None:
                status = str(analysis.get("status", "ok"))
                if status in {"ok", "partial"}:
                    st.session_state.last_analysis = analysis
                if status == "error":
                    st.warning("⚠️ Unable to process this video. Try a clearer recording.")
                    st.info(str(analysis.get("message", "Unable to analyze video properly")))

        if st.session_state.last_analysis:
            _render_coaching_view(st.session_state.last_analysis)
        else:
            st.info("Upload a video and click Process Video.")

    elif section == "Live Coaching":
        st.subheader("Live Coaching")
        live_col1, live_col2 = st.columns([2, 1])
        with live_col1:
            live_reference_shot = st.selectbox(
                "Live reference shot",
                options=["cover_drive", "straight_drive", "pull_shot", "defense"],
                index=0,
                help="Reference used for instant coaching in webcam mode.",
            )
        with live_col2:
            launch_live = st.button("Launch Live Coaching", use_container_width=True)

        if launch_live:
            try:
                live_reference = _load_reference_profile(live_reference_shot)

                def _start_live() -> None:
                    try:
                        run_realtime_coaching(reference_features=live_reference)
                    except Exception:
                        # Error is surfaced in Streamlit via pre-launch checks.
                        pass

                threading.Thread(target=_start_live, daemon=True).start()
                st.success("Realtime coaching started in external window. Press Q to quit.")
            except Exception as exc:
                st.error(f"Live coaching failed: {exc}")

    elif section == "Performance Reports":
        st.subheader("Performance Reports")
        analysis = st.session_state.last_analysis
        if not analysis:
            st.info("No analyzed session found. Run Video Analysis first.")
            return
        _render_report_view(analysis)

    elif section == "Advanced Biomechanics":
        st.subheader("Advanced Biomechanics")
        analysis = st.session_state.last_analysis
        if not analysis:
            st.info("No analyzed session found. Run Video Analysis first.")
            return
        _render_advanced_biomechanics_view(analysis)

    else:
        st.info("Select a section from the sidebar.")


if __name__ == "__main__":
    main()
