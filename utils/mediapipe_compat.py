"""MediaPipe compatibility layer supporting legacy `solutions` and newer `tasks` APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import urllib.request

import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None


Keypoint2D = Tuple[float, float, float]
Keypoint3D = Tuple[float, float, float]

# BlazePose index map shared by both APIs
LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

POSE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


@dataclass
class PoseDetectionResult:
    keypoints_2d: Dict[str, Keypoint2D]
    keypoints_3d: Dict[str, Keypoint3D]
    confidence: float


class _PoseBackendBase:
    def detect_pose(self, image_rgb: np.ndarray) -> PoseDetectionResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _NoopPoseBackend(_PoseBackendBase):
    def detect_pose(self, image_rgb: np.ndarray) -> PoseDetectionResult:
        return PoseDetectionResult({}, {}, 0.0)


class _SolutionsPoseBackend(_PoseBackendBase):
    def __init__(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_complexity: int,
    ) -> None:
        pose_mod = mp.solutions.pose
        self._pose = pose_mod.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_pose(self, image_rgb: np.ndarray) -> PoseDetectionResult:
        h, w = image_rgb.shape[:2]
        res = self._pose.process(image_rgb)

        if not res.pose_landmarks:
            return PoseDetectionResult({}, {}, 0.0)

        lm2d = res.pose_landmarks.landmark
        lm3d = res.pose_world_landmarks.landmark if res.pose_world_landmarks else lm2d

        keypoints_2d: Dict[str, Keypoint2D] = {}
        keypoints_3d: Dict[str, Keypoint3D] = {}
        vis_vals = []

        for name, idx in LANDMARK_INDEX.items():
            p2 = lm2d[idx]
            p3 = lm3d[idx]
            vis = float(getattr(p2, "visibility", 1.0))
            keypoints_2d[name] = (float(p2.x * w), float(p2.y * h), vis)
            keypoints_3d[name] = (float(p3.x), float(p3.y), float(p3.z))
            vis_vals.append(vis)

        conf = float(np.mean(vis_vals)) if vis_vals else 0.0
        return PoseDetectionResult(keypoints_2d, keypoints_3d, conf)

    def close(self) -> None:
        self._pose.close()


class _TasksPoseBackend(_PoseBackendBase):
    def __init__(self, model_asset_path: Optional[str] = None) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(model_asset_path) if model_asset_path else _ensure_pose_task_model()

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect_pose(self, image_rgb: np.ndarray) -> PoseDetectionResult:
        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        res = self._landmarker.detect(mp_image)

        if not res.pose_landmarks:
            return PoseDetectionResult({}, {}, 0.0)

        lm2d = res.pose_landmarks[0]
        lm3d = res.pose_world_landmarks[0] if res.pose_world_landmarks else lm2d

        keypoints_2d: Dict[str, Keypoint2D] = {}
        keypoints_3d: Dict[str, Keypoint3D] = {}
        vis_vals = []

        for name, idx in LANDMARK_INDEX.items():
            p2 = lm2d[idx]
            p3 = lm3d[idx]
            vis = float(getattr(p2, "visibility", getattr(p2, "presence", 1.0)))
            keypoints_2d[name] = (float(p2.x * w), float(p2.y * h), vis)
            keypoints_3d[name] = (float(p3.x), float(p3.y), float(p3.z))
            vis_vals.append(vis)

        conf = float(np.mean(vis_vals)) if vis_vals else 0.0
        return PoseDetectionResult(keypoints_2d, keypoints_3d, conf)

    def close(self) -> None:
        self._landmarker.close()


def _ensure_pose_task_model() -> Path:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "assets" / "pose_landmarker_lite.task"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        return model_path

    try:
        urllib.request.urlretrieve(POSE_TASK_URL, str(model_path))
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe Tasks backend selected but pose model was not found and auto-download failed. "
            "Place `pose_landmarker_lite.task` under assets/."
        ) from exc

    return model_path


def get_pose_model(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    model_asset_path: Optional[str] = None,
) -> _PoseBackendBase:
    """Get a pose model backend compatible across MediaPipe versions."""
    if mp is None:
        return _NoopPoseBackend()

    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return _SolutionsPoseBackend(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity,
            )
    except Exception:
        pass

    # Fallback for modern mediapipe.tasks API
    return _TasksPoseBackend(model_asset_path=model_asset_path)
