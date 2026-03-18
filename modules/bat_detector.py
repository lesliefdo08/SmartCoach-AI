from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class YOLOBatBallDetector:
    model_name: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    _model: object | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
        except Exception:
            self._model = None

    def detect(self, frame_bgr: np.ndarray) -> Dict[str, object]:
        if self._model is None:
            return {
                "bat_box": None,
                "ball_box": None,
                "player_box": None,
                "player_count": 0,
            }

        results = self._model.predict(source=frame_bgr, conf=self.conf_threshold, verbose=False)
        if not results:
            return {"bat_box": None, "ball_box": None, "player_box": None, "player_count": 0}

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return {"bat_box": None, "ball_box": None, "player_box": None, "player_count": 0}

        names = r.names if hasattr(r, "names") else {}
        bat_box: BBox | None = None
        ball_box: BBox | None = None
        player_box: BBox | None = None
        bat_conf = -1.0
        ball_conf = -1.0
        person_conf = -1.0
        person_count = 0

        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            label = str(names.get(cls_id, ""))

            if label in {"baseball bat", "bat"} and conf > bat_conf:
                bat_conf = conf
                bat_box = (x1, y1, x2, y2)
            elif label in {"sports ball", "ball"} and conf > ball_conf:
                ball_conf = conf
                ball_box = (x1, y1, x2, y2)
            elif label == "person" and conf > person_conf:
                person_count += 1
                person_conf = conf
                player_box = (x1, y1, x2, y2)
            elif label == "person":
                person_count += 1

        return {
            "bat_box": bat_box,
            "ball_box": ball_box,
            "player_box": player_box,
            "player_count": int(person_count),
        }


def box_center(box: BBox | None) -> Optional[Point]:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def track_paths(paths: Dict[str, List[Point]], detection: Dict[str, object], max_len: int = 120) -> Dict[str, List[Point]]:
    bat_center = box_center(detection.get("bat_box"))
    ball_center = box_center(detection.get("ball_box"))

    if bat_center is not None:
        paths.setdefault("bat", []).append(bat_center)
        if len(paths["bat"]) > max_len:
            paths["bat"].pop(0)

    if ball_center is not None:
        paths.setdefault("ball", []).append(ball_center)
        if len(paths["ball"]) > max_len:
            paths["ball"].pop(0)

    return paths


def movement_features(paths: Dict[str, List[Point]], frame_shape: tuple[int, int, int] | None = None) -> Dict[str, float]:
    bat_path = paths.get("bat", [])
    ball_path = paths.get("ball", [])

    bat_swing_arc = 0.0
    bat_velocity = 0.0
    bat_angle = 0.0
    bat_follow_through_height = 0.0
    ball_direction = 0.0

    if len(bat_path) >= 3:
        arr = np.array(bat_path, dtype=np.float32)
        vectors = arr[1:] - arr[:-1]
        bat_velocity = float(np.mean(np.linalg.norm(vectors, axis=1)))
        angles = np.degrees(np.arctan2(-vectors[:, 1], vectors[:, 0]))
        bat_swing_arc = float(np.max(angles) - np.min(angles))
        bat_angle = float(angles[-1])
        y_start = float(arr[0, 1])
        y_end = float(arr[-1, 1])
        if frame_shape is not None and frame_shape[0] > 0:
            bat_follow_through_height = float((y_start - y_end) / frame_shape[0])
        else:
            bat_follow_through_height = float(y_start - y_end)

    if len(ball_path) >= 2:
        p0 = np.array(ball_path[-2], dtype=np.float32)
        p1 = np.array(ball_path[-1], dtype=np.float32)
        v = p1 - p0
        ball_direction = float(np.degrees(np.arctan2(-v[1], v[0])))

    return {
        "bat_swing_arc": round(bat_swing_arc, 3),
        "bat_velocity": round(bat_velocity, 3),
        "bat_angle": round(bat_angle, 3),
        "bat_follow_through_height": round(bat_follow_through_height, 3),
        "ball_direction": round(ball_direction, 3),
    }
