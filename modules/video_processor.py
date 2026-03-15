from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
        cap = load_video(str(video_path))
        frames, fps = extract_frames(cap, sample_rate=max(1, self.sample_rate))
        cap.release()

        frame_data: List[Dict[str, object]] = []
        features_by_frame: List[Dict[str, float]] = []
        paths: Dict[str, List[tuple[int, int]]] = {"bat": [], "ball": []}

        for idx, frame in enumerate(frames, start=1):
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            rgb_norm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            det = self.detector.detect(frame)
            paths = track_paths(paths, det)
            obj_features = movement_features(paths)

            keypoints = self.pose_tracker.track_landmarks(rgb_norm)
            bat_center = None
            if det.get("bat_box") is not None:
                x1, y1, x2, y2 = det["bat_box"]
                bat_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            pose_metrics = compute_pose_biomechanics(keypoints, bat_center=bat_center)

            combined = merge_features(obj_features, pose_metrics)
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

        agg = sliding_window_average(features_by_frame, window_size=15)

        return {
            "fps": fps,
            "frame_data": frame_data,
            "features_by_frame": features_by_frame,
            "window_features": agg,
            "bat_path": paths.get("bat", []),
            "ball_path": paths.get("ball", []),
        }

    def close(self) -> None:
        self.pose_tracker.close()
