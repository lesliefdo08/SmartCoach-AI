from __future__ import annotations

from datetime import datetime
import json
import tempfile
import threading
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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
from modules.session_manager import fetch_user_sessions, save_analysis_session
from modules.shot_classifier import classify_shot_temporal, generate_contextual_feedback
from ui.dashboard import render_user_dashboard
from ui.login import render_login
from ui.signup import render_signup
from utils.visualization import (
    draw_3d_skeleton_projection,
    draw_ball_trajectory,
    draw_bat_trajectory,
    draw_pose_overlay,
)


ROOT_DIR = Path(__file__).parent
REFERENCE_DIR = ROOT_DIR / "reference_data"
MODEL_PATH = ROOT_DIR / "assets" / "shot_classifier.pkl"
MISTAKE_MODEL_PATH = ROOT_DIR / "assets" / "mistake_detector.pkl"


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
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None


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
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)

    reference_profiles = load_reference_profiles(REFERENCE_DIR)
    sequences, labels = generate_synthetic_training_data(
        reference_profiles=reference_profiles,
        samples_per_class=220,
        min_seq_len=18,
        max_seq_len=40,
        random_state=42,
    )
    bundle = train_model(
        sequences=sequences,
        labels=labels,
        model_type="random_forest",
        random_state=42,
    )
    save_model(bundle, MODEL_PATH)
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


def _run_video_analysis(uploaded_video, sample_rate: int, user_id: int, player_name: str) -> Dict[str, object] | None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_video.read())
            temp_path = temp_video.name
    except Exception as exc:
        st.error(f"Invalid or unreadable upload: {exc}")
        return None

    try:
        with st.spinner("Loading video..."):
            cap = load_video(temp_path)
            frames, fps = extract_frames(cap, sample_rate=sample_rate)
            cap.release()
    except Exception as exc:
        st.error(f"Video loading failed: {exc}")
        return None

    if not frames:
        st.error("No frames could be extracted from the uploaded video.")
        return None

    pipeline = FramePipeline(target_size=(854, 480))
    comparator = PoseComparator()

    pose_entries: List[Dict[str, object]] = []
    keypoint_series: List[Dict[str, tuple[float, float, float]]] = []
    motion_series: List[Dict[str, float]] = []
    biomechanics_series: List[Dict[str, float]] = []
    display_candidates: List[Dict[str, object]] = []
    resized_frames: List[np.ndarray] = []
    torso_rotation_series: List[float] = []
    bat_plane_series: List[float] = []
    advanced_projection_frames: List[np.ndarray] = []
    bat_tips: List[tuple[int, int]] = []
    ball_centers: List[tuple[int, int]] = []
    previous_pose: Dict[str, object] = {}

    progress = st.progress(0, text="Analyzing frames...")
    total = len(frames)

    for idx, frame in enumerate(frames, start=1):
        run_pose_this_frame = (idx % 2 == 0)
        shared = pipeline.process_frame(frame, previous_pose=previous_pose, run_pose=run_pose_this_frame)

        frame_bgr_small = shared["frame_bgr"]
        resized_frames.append(frame_bgr_small)
        keypoints = shared["pose_landmarks"]
        features = shared["pose_features"]
        pose3d = shared.get("pose3d", {})

        previous_pose = {
            "pose_landmarks": keypoints,
            "pose3d": pose3d,
            "confidence": shared.get("pose_confidence", 0.0),
        }

        bat_det = shared.get("bat_detection", {})
        if bat_det.get("tip") is not None:
            bat_tips.append(bat_det["tip"])

        ball_det = shared.get("ball_detection", {})
        if ball_det.get("center") is not None:
            ball_centers.append(ball_det["center"])

        bio3d = shared.get("biomechanics", {})
        torso_rotation_series.append(float(bio3d.get("torso_twist", 0.0)))
        bat_plane_series.append(float(bio3d.get("bat_swing_plane_angle", 0.0)))
        if pose3d and len(advanced_projection_frames) < 8 and idx % max(1, total // 8) == 0:
            advanced_projection_frames.append(draw_3d_skeleton_projection(frame_bgr_small, pose3d))

        if not keypoints:
            progress.progress(int((idx / total) * 100), text=f"Analyzing frames... {idx}/{total}")
            continue

        keypoint_series.append(keypoints)
        if len(keypoint_series) >= 2:
            prev = keypoint_series[-2]
            curr = keypoint_series[-1]

            lw_prev = np.array(prev["left_wrist"][:2], dtype=np.float32)
            rw_prev = np.array(prev["right_wrist"][:2], dtype=np.float32)
            lw_curr = np.array(curr["left_wrist"][:2], dtype=np.float32)
            rw_curr = np.array(curr["right_wrist"][:2], dtype=np.float32)
            sh_prev = (np.array(prev["left_shoulder"][:2]) + np.array(prev["right_shoulder"][:2])) / 2.0
            sh_curr = (np.array(curr["left_shoulder"][:2]) + np.array(curr["right_shoulder"][:2])) / 2.0
            wr_prev = (lw_prev + rw_prev) / 2.0
            wr_curr = (lw_curr + rw_curr) / 2.0

            vec_prev = wr_prev - sh_prev
            vec_curr = wr_curr - sh_curr
            bat_angle_prev = float(np.degrees(np.arctan2(-vec_prev[1], vec_prev[0])))
            bat_angle_curr = float(np.degrees(np.arctan2(-vec_curr[1], vec_curr[0])))
            follow_dx = float(wr_curr[0] - wr_prev[0])
            follow_dy = float(wr_curr[1] - wr_prev[1])
            body_lean = float(
                np.degrees(
                    np.arctan2(
                        abs(curr["nose"][0] - ((curr["left_hip"][0] + curr["right_hip"][0]) / 2.0)),
                        abs(curr["nose"][1] - ((curr["left_hip"][1] + curr["right_hip"][1]) / 2.0)) + 1e-6,
                    )
                )
            )
            shoulder_vec = np.array(curr["right_shoulder"][:2]) - np.array(curr["left_shoulder"][:2])
            motion_series.append(
                {
                    "bat_swing_arc": abs(bat_angle_curr - bat_angle_prev),
                    "wrist_velocity": float((np.linalg.norm(lw_curr - lw_prev) + np.linalg.norm(rw_curr - rw_prev)) / 2.0),
                    "body_lean": body_lean,
                    "follow_dx": follow_dx,
                    "follow_dy": follow_dy,
                    "horizontal_ratio": abs(follow_dx) / (abs(follow_dy) + 1e-6),
                    "upward_ratio": max(0.0, -follow_dy) / (abs(follow_dx) + 1e-6),
                    "wrist_height_norm": float((sh_curr[1] - wr_curr[1]) / (abs(sh_curr[1]) + 1e-6)),
                    "bat_angle": bat_angle_curr,
                    "shoulder_rotation": float(abs(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))),
                }
            )

        ball_line_x = float(ball_det["center"][0]) if ball_det.get("center") is not None else float(frame_bgr_small.shape[1]) / 2.0
        biomechanics_series.append(compute_biomechanics_frame(keypoints=keypoints, ball_line_x=ball_line_x))

        pose_entries.append(
            {
                "frame_index": idx,
                "features": features,
            }
        )

        if len(display_candidates) < 8 and idx % max(1, total // 8) == 0:
            display_candidates.append(
                {
                    "frame_rgb": cv2.cvtColor(frame_bgr_small, cv2.COLOR_BGR2RGB),
                    "keypoints": keypoints,
                    "features": features,
                }
            )

        progress.progress(int((idx / total) * 100), text=f"Analyzing frames... {idx}/{total}")

    pipeline.close()
    progress.empty()

    if not pose_entries:
        st.warning("No valid poses detected. Try a clearer side-view cricket clip.")
        return None

    bat_tracking = _build_bat_tracking_from_path(bat_tips)
    ball_tracking = _build_ball_tracking_from_path(ball_centers, bat_tracking.get("smoothed_bat_path", []))

    shot_prediction = classify_shot_temporal(motion_series, window_size=15)
    raw_shot = str(shot_prediction["shot_type"])
    detected_shot = raw_shot.lower().replace(" ", "_")
    confidence_score = float(shot_prediction["confidence_score"])
    reference_key_map = {
        "defense": "defense",
        "drive": "cover_drive",
        "lofted_shot": "straight_drive",
        "cut_pull": "pull_shot",
    }
    reference_shot = reference_key_map.get(detected_shot, "defense")
    try:
        reference_profile = _load_reference_profile(reference_shot)
    except Exception as exc:
        st.error(f"Reference profile loading failed: {exc}")
        return None
    reference_features = get_reference_means(reference_profile)

    frame_results: List[Dict[str, object]] = []
    for entry in pose_entries:
        comparison = comparator.compare(entry["features"], reference_profile)
        frame_results.append(
            {
                "frame_index": entry["frame_index"],
                "features": entry["features"],
                "similarity_score": comparison["similarity_score"],
                "comparison": comparison,
            }
        )

    display_frames: List[np.ndarray] = []
    for sample in display_candidates:
        comparison = comparator.compare(sample["features"], reference_profile)
        frame_bgr = cv2.cvtColor(sample["frame_rgb"], cv2.COLOR_RGB2BGR)
        overlay = draw_pose_overlay(
            frame_bgr=frame_bgr,
            keypoints=sample["keypoints"],
            joint_errors=comparison["joint_errors"],
            angles=sample["features"],
        )
        display_frames.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    biomechanics_data = {
        "bat": bat_tracking,
        "ball": ball_tracking,
        "pose3d": {
            "torso_twist_series": torso_rotation_series,
            "bat_swing_plane_series": bat_plane_series,
        },
    }
    metrics = compute_performance_metrics(frame_results, biomechanics_data=biomechanics_data)
    bio_scores = summarize_biomechanics(biomechanics_series)
    similarity_series = [fr["similarity_score"] for fr in frame_results]
    avg_features = _aggregate_features(frame_results)
    avg_comparison = comparator.compare(avg_features, reference_profile)
    detected_mistakes: List[Dict[str, object]] = []
    contextual_summary = generate_contextual_feedback(shot_prediction, bio_scores, motion_series)
    final_feedback = {
        "summary": contextual_summary,
        "tips": [
            "Track head stability through impact for improved balance.",
            "Maintain a repeatable bat path in the follow-through phase.",
            "Use front-knee control to improve shot consistency.",
        ],
    }

    features_df, similarity_df, deviation_df, frame_table_df = build_analysis_frames(frame_results, reference_features)
    mistake_summary = summarize_mistakes_and_suggestions(deviation_df)
    score_card = {
        "technique": bio_scores["technique_score"],
        "balance": bio_scores["balance_score"],
        "consistency": bio_scores["consistency_score"],
        "overall": round(
            0.4 * bio_scores["technique_score"]
            + 0.3 * bio_scores["balance_score"]
            + 0.3 * bio_scores["consistency_score"],
            2,
        ),
    }

    save_analysis_session(
        user_id=user_id,
        video_name=getattr(uploaded_video, "name", "uploaded_video.mp4"),
        shot_type=str(shot_prediction["shot_type"]),
        confidence=confidence_score,
        technique_score=score_card["technique"],
        balance_score=score_card["balance"],
        consistency_score=score_card["consistency"],
    )

    session_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player": player_name,
        "shot": raw_shot,
        "confidence": round(confidence_score, 2),
        "overall_score": score_card["overall"],
    }
    st.session_state.session_history.append(session_entry)

    return {
        "player_name": player_name,
        "fps": fps,
        "detected_shot": raw_shot,
        "confidence_score": confidence_score,
        "shot_prediction": shot_prediction,
        "metrics": metrics,
        "score_card": score_card,
        "similarity_series": similarity_series,
        "avg_features": avg_features,
        "reference_features": reference_features,
        "final_feedback": final_feedback,
        "detected_mistakes": detected_mistakes,
        "display_frames": display_frames,
        "features_df": features_df,
        "similarity_df": similarity_df,
        "deviation_df": deviation_df,
        "frame_table_df": frame_table_df,
        "mistake_summary": mistake_summary,
        "biomechanics": biomechanics_data,
        "advanced_projection_frames": [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in advanced_projection_frames],
        "last_frame_rgb": cv2.cvtColor(resized_frames[-1], cv2.COLOR_BGR2RGB) if resized_frames else None,
    }


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

    if not st.session_state.authenticated:
        st.markdown("### Login or Sign Up")
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
        with login_tab:
            render_login()
        with signup_tab:
            render_signup()
        return

    user = st.session_state.user
    if not user:
        st.session_state.authenticated = False
        st.rerun()

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
        st.markdown(f"### User: {user['username']}")
        player_name = st.text_input("Player Name", value=st.session_state.player_name)
        st.session_state.player_name = player_name

        options = ["Home", "Video Analysis", "Live Coaching", "Performance Reports", "Advanced Biomechanics", "My Dashboard"]
        current = st.session_state.current_section if st.session_state.current_section in options else "Home"
        section = st.radio(
            "Navigation",
            options=options,
            index=options.index(current),
        )
        st.session_state.current_section = section

        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.last_analysis = None
            st.rerun()

        st.divider()
        st.markdown("### Session Statistics")
        history = fetch_user_sessions(int(user["id"]), limit=200)
        total_sessions = len(history)
        avg_overall = (
            float(np.mean([(float(h["technique_score"]) + float(h["balance_score"]) + float(h["consistency_score"])) / 3.0 for h in history]))
            if history
            else 0.0
        )
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
            analysis = _run_video_analysis(uploaded_video, sample_rate, int(user["id"]), player_name)
            if analysis is not None:
                st.session_state.last_analysis = analysis

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
        render_user_dashboard(int(user["id"]))


if __name__ == "__main__":
    main()
