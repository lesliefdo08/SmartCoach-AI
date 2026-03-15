"""Video loading and frame preprocessing utilities for SmartCoach AI."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def load_video(path: str) -> cv2.VideoCapture:
    """Load a video file and return an OpenCV capture object."""
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    return video


def extract_frames(
    video: cv2.VideoCapture,
    sample_rate: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], float]:
    """Extract frames from a video capture object.

    Args:
        video: OpenCV VideoCapture object.
        sample_rate: Keep every Nth frame.
        max_frames: Optional maximum number of returned frames.

    Returns:
        A tuple of (frames, fps).
    """
    fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break

        if frame_idx % max(sample_rate, 1) == 0:
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_idx += 1

    return frames, fps


def preprocess_frame(
    frame: np.ndarray,
    size: Tuple[int, int] = (640, 360),
) -> np.ndarray:
    """Resize and normalize a frame for pose processing.

    - Resizes to `size` (width, height)
    - Converts BGR to RGB
    - Normalizes pixel values to [0, 1]
    """
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return normalized
