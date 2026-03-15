"""MediaPipe pose detection module for SmartCoach AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from utils.mediapipe_compat import get_pose_model


Keypoint = Tuple[float, float, float]


@dataclass
class PoseDetector:
    """Wrapper around MediaPipe Pose for landmark extraction."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1

    def __post_init__(self) -> None:
        self._pose = get_pose_model(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=self.model_complexity,
        )

    def detect(self, frame_rgb_normalized: np.ndarray) -> Dict[str, Keypoint]:
        """Detect required keypoints from a normalized RGB frame."""
        image = (np.clip(frame_rgb_normalized, 0.0, 1.0) * 255).astype(np.uint8)
        result = self._pose.detect_pose(image)
        return result.keypoints_2d

    def detect_with_meta(self, frame_rgb_normalized: np.ndarray) -> Dict[str, object]:
        """Detect pose with confidence and 3D landmark metadata."""
        image = (np.clip(frame_rgb_normalized, 0.0, 1.0) * 255).astype(np.uint8)
        result = self._pose.detect_pose(image)
        return {
            "keypoints": result.keypoints_2d,
            "pose3d": result.keypoints_3d,
            "confidence": result.confidence,
        }

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self._pose.close()
        except Exception:
            pass


def get_default_connections() -> Tuple[Tuple[str, str], ...]:
    """Return a compact set of limb connections for rendering."""
    return (
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
    )
