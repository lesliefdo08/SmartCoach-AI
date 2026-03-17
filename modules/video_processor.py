from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.video_processor import extract_frames, load_video
from modules.bat_detector import YOLOBatBallDetector, movement_features, track_paths
from modules.feature_extractor import merge_features, sliding_window_average
from modules.pose_detector import CricketPoseTracker, compute_pose_biomechanics


@dataclass
class CricketAnalyticsPipeline:
    sample_rate: int = 2
    target_size: tuple[int, int] = (854, 480)

    def __post_init__(self) -> None:
        self.detector = YOLOBatBallDetector()
        self.pose_tracker = CricketPoseTracker(min_confidence=0.45)

    def process_video(self, video_path: str | Path) -> Dict[str, object]:
        return self.process_video_filtered(video_path=video_path, strict_filter=False)

    def process_video_filtered(
        self,
        video_path: str | Path,
        strict_filter: bool = False,
        min_player_area_ratio: float = 0.25,
    ) -> Dict[str, object]:
        cap = load_video(str(video_path))
        frames, fps = extract_frames(cap, sample_rate=max(1, self.sample_rate))
        cap.release()

        frame_data: List[Dict[str, object]] = []
        features_by_frame: List[Dict[str, float]] = []
        motion_scores: List[float] = []
        paths: Dict[str, List[tuple[int, int]]] = {"bat": [], "ball": []}
        prev_wrist_center: Optional[np.ndarray] = None
        pose_cache: Dict[Tuple[int, int], Dict[str, object]] = {}

        for idx, frame in enumerate(frames, start=1):
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            rgb_norm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            det = self.detector.detect(frame)
            player_box = det.get("player_box")
            bat_box = det.get("bat_box")
            paths = track_paths(paths, det)
            obj_features = movement_features(paths, frame_shape=frame.shape)

            frame_signature = self._frame_signature(frame)
            cached_pose = pose_cache.get(frame_signature)
            if cached_pose is None:
                keypoints = self.pose_tracker.track_landmarks(rgb_norm)
                cached_pose = {"keypoints": keypoints}
                pose_cache[frame_signature] = cached_pose
            else:
                keypoints = cached_pose.get("keypoints", {})

            if strict_filter:
                if not keypoints or bat_box is None or player_box is None:
                    continue
                x1, y1, x2, y2 = player_box
                player_area = max(0, x2 - x1) * max(0, y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                if frame_area <= 0 or (player_area / frame_area) < min_player_area_ratio:
                    continue

            bat_center = None
            if bat_box is not None:
                x1, y1, x2, y2 = bat_box
                bat_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            pose_metrics = compute_pose_biomechanics(keypoints, bat_center=bat_center)

            wrist_center = self._wrist_center(keypoints)
            wrist_trajectory = 0.0
            if wrist_center is not None and prev_wrist_center is not None:
                wrist_trajectory = float(np.linalg.norm(wrist_center - prev_wrist_center))
            if wrist_center is not None:
                prev_wrist_center = wrist_center

            pose_metrics["wrist_trajectory"] = round(wrist_trajectory, 3)
            movement_score = float(abs(obj_features.get("bat_velocity", 0.0)) + wrist_trajectory)
            motion_scores.append(movement_score)

            combined = merge_features(obj_features, pose_metrics)
            combined["motion_score"] = round(movement_score, 3)
            features_by_frame.append(combined)

            frame_data.append(
                {
                    "frame_index": idx,
                    "frame_bgr": frame,
                    "detection": det,
                    "keypoints": keypoints,
                    "object_features": obj_features,
                    "pose_features": pose_metrics,
                    "features": combined,
                }
            )

        if features_by_frame:
            move_arr = np.array(motion_scores, dtype=np.float32)
            start_threshold = float(max(1.5, np.percentile(move_arr, 40)))
            peak_threshold = float(max(start_threshold * 1.8, np.percentile(move_arr, 85)))

            for f in features_by_frame:
                score = float(f.get("motion_score", 0.0))
                if score < start_threshold:
                    phase = 0.0
                elif score >= peak_threshold:
                    phase = 2.0
                else:
                    phase = 1.0
                f["motion_phase"] = phase

        key_features = [f for f in features_by_frame if float(f.get("motion_phase", 0.0)) > 0.0]
        if not key_features:
            key_features = features_by_frame

        agg = sliding_window_average(key_features, window_size=15)

        return {
            "fps": fps,
            "frame_data": frame_data,
            "features_by_frame": features_by_frame,
            "key_features": key_features,
            "window_features": agg,
            "bat_path": paths.get("bat", []),
            "ball_path": paths.get("ball", []),
        }

    def close(self) -> None:
        self.pose_tracker.close()

    @staticmethod
    def _wrist_center(keypoints: Dict[str, Tuple[float, float, float]]) -> Optional[np.ndarray]:
        if not keypoints or "left_wrist" not in keypoints or "right_wrist" not in keypoints:
            return None
        l_wr = np.array(keypoints["left_wrist"][:2], dtype=np.float32)
        r_wr = np.array(keypoints["right_wrist"][:2], dtype=np.float32)
        return (l_wr + r_wr) / 2.0

    @staticmethod
    def _frame_signature(frame_bgr: np.ndarray) -> Tuple[int, int]:
        # Lightweight cache key to avoid repeated pose inference on near-identical frames.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return int(np.mean(gray)), int(np.std(gray))
