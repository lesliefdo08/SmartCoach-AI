"""Real-time webcam coaching mode for SmartCoach AI."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import streamlit as st

from core.feedback_engine import generate_feedback
from core.frame_pipeline import FramePipeline
from core.pose_comparator import PoseComparator, get_reference_means
from utils.visualization import draw_ball_trajectory, draw_bat_trajectory, draw_pose_overlay


def run_realtime_coaching(
    reference_features: Dict[str, object],
    camera_index: int = 0,
    target_size: tuple[int, int] = (854, 480),
    save_dir: str | Path = "assets/realtime_captures",
    max_frames: int = 600,
    save_interval_frames: int = 120,
) -> None:
    """Run live coaching on webcam frames.

    Frames are rendered using Streamlit images for cloud compatibility.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    pipeline = FramePipeline(target_size=target_size)
    comparator = PoseComparator()
    reference_means = get_reference_means(reference_features)

    prev_time = time.time()
    fps_buffer: List[float] = []

    latest_overlay: np.ndarray | None = None
    latest_payload: Dict[str, object] = {}
    bat_path: List[tuple[int, int]] = []
    ball_path: List[tuple[int, int]] = []
    previous_pose: Dict[str, object] = {}
    frame_counter = 0
    placeholder = st.empty()

    try:
        while frame_counter < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                continue

            frame_counter += 1
            # CPU optimization: run heavy pose estimation every other frame.
            run_pose_this_frame = (frame_counter % 2 == 0)
            shared = pipeline.process_frame(frame_bgr, previous_pose=previous_pose, run_pose=run_pose_this_frame)
            frame_bgr = shared["frame_bgr"]
            keypoints = shared["pose_landmarks"]
            features = shared["pose_features"]
            bio3d = shared["biomechanics"]
            previous_pose = {
                "pose_landmarks": keypoints,
                "pose3d": shared.get("pose3d", {}),
                "confidence": shared.get("pose_confidence", 0.0),
            }

            similarity_score = 0.0
            tips: List[str] = ["Align your setup and hold still for detection."]
            joint_errors: List[Dict[str, float]] = []

            if keypoints:
                comparison = comparator.compare(features, reference_features)
                feedback = generate_feedback(features, reference_means, comparison["joint_errors"])
                similarity_score = float(comparison["similarity_score"])
                tips = feedback.get("tips", [])[:3]
                joint_errors = comparison["joint_errors"]

            bat_det = shared.get("bat_detection", {})
            if bat_det.get("tip") is not None:
                bat_path.append(bat_det["tip"])
                if len(bat_path) > 80:
                    bat_path.pop(0)

            ball_det = shared.get("ball_detection", {})
            if ball_det.get("center") is not None:
                ball_path.append(ball_det["center"])
                if len(ball_path) > 80:
                    ball_path.pop(0)

            overlay = draw_pose_overlay(
                frame_bgr=frame_bgr,
                keypoints=keypoints,
                joint_errors=joint_errors,
                angles=features,
            )
            overlay = draw_bat_trajectory(overlay, bat_path)
            overlay = draw_ball_trajectory(overlay, ball_path, impact_point=ball_path[-1] if ball_path else None)

            now = time.time()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            fps_buffer.append(inst_fps)
            if len(fps_buffer) > 20:
                fps_buffer.pop(0)
            smooth_fps = float(np.mean(fps_buffer)) if fps_buffer else 0.0

            cv2.putText(
                overlay,
                f"Similarity: {similarity_score:.1f}%",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"FPS: {smooth_fps:.1f}",
                (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 255, 80) if smooth_fps >= 15.0 else (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Torso Twist: {float(bio3d.get('torso_twist', 0.0)):.1f} deg",
                (12, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 215, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Bat Plane: {float(bio3d.get('bat_swing_plane_angle', 0.0)):.1f} deg",
                (12, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 215, 0),
                1,
                cv2.LINE_AA,
            )

            y = 122
            for tip in tips:
                cv2.putText(
                    overlay,
                    f"- {tip}",
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y += 24

            cv2.putText(
                overlay,
                "Live stream mode (Streamlit)",
                (12, target_size[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            latest_overlay = overlay.copy()
            latest_payload = {
                "timestamp": datetime.now().isoformat(),
                "similarity_score": similarity_score,
                "fps": round(smooth_fps, 2),
                "features": features,
                "biomechanics": bio3d,
                "tips": tips,
            }

            placeholder.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            if save_interval_frames > 0 and frame_counter % save_interval_frames == 0:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = save_path / f"analysis_{stamp}.jpg"
                json_path = save_path / f"analysis_{stamp}.json"

                if latest_overlay is not None:
                    cv2.imwrite(str(image_path), latest_overlay)
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(latest_payload, f, indent=2)

    finally:
        pipeline.close()
        cap.release()
        st.info("Live coaching stream ended.")
