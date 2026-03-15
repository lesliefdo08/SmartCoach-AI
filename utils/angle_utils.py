"""Geometric utility functions for angle and alignment calculations."""

from __future__ import annotations

from typing import Tuple

import numpy as np


Point2D = Tuple[float, float]


def midpoint(a: Point2D, b: Point2D) -> Point2D:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def angle_between_three_points(a: Point2D, b: Point2D, c: Point2D) -> float:
    """Return angle ABC in degrees using 3-point geometry."""
    ba = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0.0 or norm_bc == 0.0:
        return float("nan")

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def line_angle_degrees(a: Point2D, b: Point2D) -> float:
    """Return absolute line angle to horizontal in degrees."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return float(abs(np.degrees(np.arctan2(dy, dx))))
