from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from core.pose_detector import PoseDetector


Keypoint = Tuple[float, float, float]


@dataclass
class CricketPoseTracker:
    min_confidence: float = 0.45

    def __post_init__(self) -> None:
        self.detector = PoseDetector(min_detection_confidence=self.min_confidence, min_tracking_confidence=self.min_confidence)

    def track_landmarks(self, frame_rgb_normalized: np.ndarray) -> Dict[str, Keypoint]:
        points = self.detector.detect(frame_rgb_normalized)
        if not points:
            return {}

        required = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "nose",
        ]
        if any(name not in points for name in required):
            return {}

        left_hip = points["left_hip"]
        right_hip = points["right_hip"]
        hip_center = (
            (left_hip[0] + right_hip[0]) / 2.0,
            (left_hip[1] + right_hip[1]) / 2.0,
            (left_hip[2] + right_hip[2]) / 2.0,
        )

        facing_right = points["right_shoulder"][0] > points["left_shoulder"][0]
        front_knee = points["left_knee"] if facing_right else points["right_knee"]
        back_knee = points["right_knee"] if facing_right else points["left_knee"]

        tracked = {name: points[name] for name in required}
        tracked["hip_center"] = hip_center
        tracked["front_knee"] = front_knee
        tracked["back_knee"] = back_knee
        return tracked

    def close(self) -> None:
        self.detector.close()


def compute_motion_series(landmark_series: List[Dict[str, Keypoint]]) -> List[Dict[str, float]]:
    if len(landmark_series) < 2:
        return []

    metrics: List[Dict[str, float]] = []
    for i in range(1, len(landmark_series)):
        prev = landmark_series[i - 1]
        curr = landmark_series[i]

        left_wrist_prev = np.array(prev["left_wrist"][:2], dtype=np.float32)
        right_wrist_prev = np.array(prev["right_wrist"][:2], dtype=np.float32)
        left_wrist_curr = np.array(curr["left_wrist"][:2], dtype=np.float32)
        right_wrist_curr = np.array(curr["right_wrist"][:2], dtype=np.float32)

        wrist_velocity = float((np.linalg.norm(left_wrist_curr - left_wrist_prev) + np.linalg.norm(right_wrist_curr - right_wrist_prev)) / 2.0)

        sh_prev = (np.array(prev["left_shoulder"][:2]) + np.array(prev["right_shoulder"][:2])) / 2.0
        sh_curr = (np.array(curr["left_shoulder"][:2]) + np.array(curr["right_shoulder"][:2])) / 2.0
        wr_prev = (left_wrist_prev + right_wrist_prev) / 2.0
        wr_curr = (left_wrist_curr + right_wrist_curr) / 2.0

        def _arm_angle(origin: np.ndarray, wrist: np.ndarray) -> float:
            vec = wrist - origin
            return float(np.degrees(np.arctan2(-vec[1], vec[0])))

        bat_angle_prev = _arm_angle(sh_prev, wr_prev)
        bat_angle_curr = _arm_angle(sh_curr, wr_curr)
        bat_swing_arc = float(abs(bat_angle_curr - bat_angle_prev))

        hip_curr = np.array(curr["hip_center"][:2], dtype=np.float32)
        nose_curr = np.array(curr["nose"][:2], dtype=np.float32)
        body_lean = float(np.degrees(np.arctan2(abs(nose_curr[0] - hip_curr[0]), abs(nose_curr[1] - hip_curr[1]) + 1e-6)))

        shoulder_vec = np.array(curr["right_shoulder"][:2], dtype=np.float32) - np.array(curr["left_shoulder"][:2], dtype=np.float32)
        shoulder_rotation = float(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))

        follow_dx = float(wr_curr[0] - wr_prev[0])
        follow_dy = float(wr_curr[1] - wr_prev[1])
        horizontal_ratio = abs(follow_dx) / (abs(follow_dy) + 1e-6)
        upward_ratio = max(0.0, -follow_dy) / (abs(follow_dx) + 1e-6)

        metrics.append(
            {
                "bat_swing_arc": round(bat_swing_arc, 3),
                "wrist_velocity": round(wrist_velocity, 3),
                "body_lean": round(body_lean, 3),
                "shoulder_rotation": round(abs(shoulder_rotation), 3),
                "follow_dx": round(follow_dx, 3),
                "follow_dy": round(follow_dy, 3),
                "horizontal_ratio": round(horizontal_ratio, 3),
                "upward_ratio": round(upward_ratio, 3),
                "wrist_height_norm": round(float((sh_curr[1] - wr_curr[1]) / (abs(sh_curr[1]) + 1e-6)), 3),
                "bat_angle": round(bat_angle_curr, 3),
            }
        )

    return metrics


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    denom = float(np.linalg.norm(ab) * np.linalg.norm(cb))
    if denom < 1e-8:
        return 0.0
    cos_v = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_v)))


def compute_pose_biomechanics(keypoints: Dict[str, Keypoint], bat_center: tuple[int, int] | None = None) -> Dict[str, float]:
    if not keypoints:
        return {
            "elbow_angle": 0.0,
            "knee_bend": 0.0,
            "torso_tilt": 0.0,
            "head_alignment": 0.0,
            "bat_arm_alignment": 0.0,
            "follow_through_height": 0.0,
            "body_rotation": 0.0,
            "head_position": 0.0,
        }

    l_sh = np.array(keypoints["left_shoulder"][:2], dtype=np.float32)
    r_sh = np.array(keypoints["right_shoulder"][:2], dtype=np.float32)
    l_el = np.array(keypoints["left_elbow"][:2], dtype=np.float32)
    r_el = np.array(keypoints["right_elbow"][:2], dtype=np.float32)
    l_wr = np.array(keypoints["left_wrist"][:2], dtype=np.float32)
    r_wr = np.array(keypoints["right_wrist"][:2], dtype=np.float32)
    l_hip = np.array(keypoints["left_hip"][:2], dtype=np.float32)
    r_hip = np.array(keypoints["right_hip"][:2], dtype=np.float32)
    l_kn = np.array(keypoints["left_knee"][:2], dtype=np.float32)
    r_kn = np.array(keypoints["right_knee"][:2], dtype=np.float32)
    l_an = np.array(keypoints["left_ankle"][:2], dtype=np.float32)
    r_an = np.array(keypoints["right_ankle"][:2], dtype=np.float32)
    head = np.array(keypoints["nose"][:2], dtype=np.float32)

    left_elbow = _angle(l_sh, l_el, l_wr)
    right_elbow = _angle(r_sh, r_el, r_wr)
    elbow_angle = float((left_elbow + right_elbow) / 2.0)

    left_knee = _angle(l_hip, l_kn, l_an)
    right_knee = _angle(r_hip, r_kn, r_an)
    knee_bend = float((left_knee + right_knee) / 2.0)

    hip_center = (l_hip + r_hip) / 2.0
    dx = float(abs(head[0] - hip_center[0]))
    dy = float(abs(head[1] - hip_center[1])) + 1e-6
    torso_tilt = float(np.degrees(np.arctan2(dx, dy)))

    shoulder_vec = r_sh - l_sh
    body_rotation = float(abs(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0]))))

    head_alignment = float(abs(head[0] - hip_center[0]))
    head_position = float(head_alignment)

    wr_center = (l_wr + r_wr) / 2.0
    sh_center = (l_sh + r_sh) / 2.0
    arm_vec = wr_center - sh_center
    arm_angle = float(np.degrees(np.arctan2(-arm_vec[1], arm_vec[0])))
    bat_arm_alignment = 0.0
    if bat_center is not None:
        bat_vec = np.array([float(bat_center[0]) - sh_center[0], float(bat_center[1]) - sh_center[1]], dtype=np.float32)
        bat_angle = float(np.degrees(np.arctan2(-bat_vec[1], bat_vec[0])))
        bat_arm_alignment = float(180.0 - abs(bat_angle - arm_angle))
    else:
        bat_arm_alignment = float(180.0 - abs(arm_angle))

    follow_through_height = float((sh_center[1] - wr_center[1]) / (abs(sh_center[1]) + 1e-6))

    return {
        "elbow_angle": round(elbow_angle, 3),
        "knee_bend": round(knee_bend, 3),
        "torso_tilt": round(torso_tilt, 3),
        "head_alignment": round(head_alignment, 3),
        "bat_arm_alignment": round(float(np.clip(bat_arm_alignment, 0.0, 180.0)), 3),
        "follow_through_height": round(follow_through_height, 3),
        "body_rotation": round(body_rotation, 3),
        "head_position": round(head_position, 3),
    }
