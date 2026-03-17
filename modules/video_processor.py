from __future__ import annotations

from dataclasses import dataclass
import json
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
        self.motion_start_floor = 1.5
        self.motion_start_percentile = 40.0
        self.motion_peak_percentile = 85.0
        self.motion_peak_multiplier = 1.8
        self._load_calibration_settings()

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
            start_threshold = float(max(self.motion_start_floor, np.percentile(move_arr, self.motion_start_percentile)))
            peak_threshold = float(max(start_threshold * self.motion_peak_multiplier, np.percentile(move_arr, self.motion_peak_percentile)))

            for f in features_by_frame:
                score = float(f.get("motion_score", 0.0))
                if score < start_threshold:
                    phase = 0.0
                elif score >= peak_threshold:
                    phase = 2.0
                else:
                    phase = 1.0
                f["motion_phase"] = phase

            bat_vel = np.array([float(f.get("bat_velocity", 0.0)) for f in features_by_frame], dtype=np.float32)
            bat_angle = np.array([float(f.get("bat_angle", 0.0)) for f in features_by_frame], dtype=np.float32)
            direction_change = np.zeros_like(bat_angle)
            if len(bat_angle) >= 2:
                direction_change[1:] = np.abs(np.diff(bat_angle))

            impact_score = bat_vel + 0.75 * direction_change
            impact_idx = int(np.argmax(impact_score)) if len(impact_score) else 0

            pre_start = max(0, impact_idx - 2)
            pre_end = max(0, impact_idx - 1)
            post_start = min(len(features_by_frame) - 1, impact_idx + 1)
            post_end = min(len(features_by_frame) - 1, impact_idx + 2)
            key_indices = set(range(pre_start, post_end + 1))

            bat_angle_impact = float(features_by_frame[impact_idx].get("bat_angle", 0.0))

            post_follow_vals = [
                float(features_by_frame[i].get("follow_through_height", 0.0))
                for i in range(impact_idx, len(features_by_frame))
            ]
            follow_through_height_max = float(np.max(post_follow_vals)) if post_follow_vals else 0.0

            swing_direction_score = 0.0
            swing_trend = 0.0
            if len(paths.get("bat", [])) >= 3:
                bat_path = paths.get("bat", [])
                p0 = bat_path[max(0, len(bat_path) - 3)]
                p1 = bat_path[-1]
                dx = float(p1[0] - p0[0])
                dy = float(p1[1] - p0[1])
                swing_direction_score = float(np.clip((-dy) / (abs(dx) + abs(dy) + 1e-6), -1.0, 1.0))
                swing_trend = float(np.sign(-dy))

            wrist_x, wrist_y = self._wrist_position_relative(frame_data[impact_idx].get("keypoints", {}))
            impact_labels = {}
            for i in range(len(features_by_frame)):
                if i < impact_idx:
                    impact_labels[i] = "pre-impact"
                elif i == impact_idx:
                    impact_labels[i] = "impact"
                else:
                    impact_labels[i] = "post-impact"

            for i, f in enumerate(features_by_frame):
                f["bat_angle_impact"] = bat_angle_impact
                f["follow_through_height_max"] = follow_through_height_max
                f["swing_direction_score"] = swing_direction_score
                f["swing_trend"] = swing_trend
                f["wrist_position_impact_x"] = wrist_x
                f["wrist_position_impact_y"] = wrist_y
                f["player_body_lean"] = float(f.get("body_lean", f.get("torso_tilt", 0.0)))
                f["impact_phase"] = impact_labels.get(i, "post-impact")
                if i == impact_idx:
                    f["frame_weight"] = 2.4
                elif i in key_indices:
                    f["frame_weight"] = 1.6
                else:
                    f["frame_weight"] = 0.8

        key_features = [
            f
            for f in features_by_frame
            if str(f.get("impact_phase", "")).lower() in {"pre-impact", "impact", "post-impact"}
            and float(f.get("frame_weight", 0.0)) >= 1.6
        ]
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

    def _load_calibration_settings(self) -> None:
        calibration_path = Path(__file__).resolve().parents[1] / "models" / "calibration.json"
        if not calibration_path.exists():
            return
        try:
            with calibration_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return

        self.motion_start_floor = float(payload.get("motion_start_floor", self.motion_start_floor))
        self.motion_start_percentile = float(payload.get("motion_start_percentile", self.motion_start_percentile))
        self.motion_peak_percentile = float(payload.get("motion_peak_percentile", self.motion_peak_percentile))
        self.motion_peak_multiplier = float(payload.get("motion_peak_multiplier", self.motion_peak_multiplier))

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

    @staticmethod
    def _wrist_position_relative(keypoints: Dict[str, Tuple[float, float, float]]) -> Tuple[float, float]:
        if not keypoints:
            return 0.0, 0.0
        required = {"left_wrist", "right_wrist", "left_hip", "right_hip", "left_shoulder", "right_shoulder"}
        if any(k not in keypoints for k in required):
            return 0.0, 0.0

        l_wr = np.array(keypoints["left_wrist"][:2], dtype=np.float32)
        r_wr = np.array(keypoints["right_wrist"][:2], dtype=np.float32)
        wr_center = (l_wr + r_wr) / 2.0

        l_hip = np.array(keypoints["left_hip"][:2], dtype=np.float32)
        r_hip = np.array(keypoints["right_hip"][:2], dtype=np.float32)
        body_center = (l_hip + r_hip) / 2.0

        l_sh = np.array(keypoints["left_shoulder"][:2], dtype=np.float32)
        r_sh = np.array(keypoints["right_shoulder"][:2], dtype=np.float32)
        shoulder_span = float(np.linalg.norm(r_sh - l_sh)) + 1e-6

        rel_x = float((wr_center[0] - body_center[0]) / shoulder_span)
        rel_y = float((body_center[1] - wr_center[1]) / shoulder_span)
        return rel_x, rel_y
