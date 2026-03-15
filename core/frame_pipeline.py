"""Shared per-frame processing pipeline for pose and object tracking signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from core.ball_tracker import detect_ball
from core.bat_tracker import detect_bat
from core.feature_extractor import extract_pose_features
from core.pose_3d_estimator import compute_biomechanical_metrics
from core.pose_detector import PoseDetector
from core.video_processor import preprocess_frame


@dataclass
class FramePipeline:
    """Single-pass frame processor used by batch and realtime flows."""

    target_size: tuple[int, int] = (854, 480)
    pose_detector: Optional[PoseDetector] = None

    def __post_init__(self) -> None:
        if self.pose_detector is None:
            self.pose_detector = PoseDetector(model_complexity=0)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        previous_pose: Optional[Dict[str, object]] = None,
        run_pose: bool = True,
    ) -> Dict[str, object]:
        """Process one frame and return shared outputs for downstream modules."""
        resized_bgr = cv2.resize(frame_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        processed = preprocess_frame(resized_bgr, size=self.target_size)

        pose_keypoints: Dict[str, tuple[float, float, float]] = {}
        pose3d: Dict[str, tuple[float, float, float]] = {}
        pose_conf = 0.0

        if run_pose:
            meta = self.pose_detector.detect_with_meta(processed)
            pose_keypoints = meta.get("keypoints", {})
            pose3d = meta.get("pose3d", {})
            pose_conf = float(meta.get("confidence", 0.0))

        if (not pose_keypoints) and previous_pose and float(previous_pose.get("confidence", 0.0)) >= 0.55:
            pose_keypoints = previous_pose.get("pose_landmarks", {}) or {}
            pose3d = previous_pose.get("pose3d", {}) or {}
            pose_conf = float(previous_pose.get("confidence", 0.0))

        pose_features = extract_pose_features(pose_keypoints) if pose_keypoints else {}
        wrists = _wrists_from_keypoints(pose_keypoints)

        bat_detection = detect_bat(resized_bgr, wrists)
        ball_detection = detect_ball(resized_bgr)

        biomechanics = compute_biomechanical_metrics(pose3d) if pose3d else {
            "shoulder_rotation": 0.0,
            "torso_twist": 0.0,
            "bat_swing_plane_angle": 0.0,
            "center_of_gravity_estimate": [0.0, 0.0, 0.0],
        }

        return {
            "frame_bgr": resized_bgr,
            "pose_landmarks": pose_keypoints,
            "pose_features": pose_features,
            "pose3d": pose3d,
            "pose_confidence": pose_conf,
            "biomechanics": biomechanics,
            "bat_detection": bat_detection,
            "ball_detection": ball_detection,
        }

    def close(self) -> None:
        if self.pose_detector:
            self.pose_detector.close()


def _wrists_from_keypoints(keypoints: Dict[str, tuple[float, float, float]]) -> list[tuple[int, int]]:
    if not keypoints:
        return []
    out: list[tuple[int, int]] = []
    for name in ("left_wrist", "right_wrist"):
        if name in keypoints:
            x, y, vis = keypoints[name]
            if vis >= 0.2:
                out.append((int(x), int(y)))
    return out
