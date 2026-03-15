"""Visualization charts for advanced biomechanics tab."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def plot_swing_speed_vs_frame(swing_speed_per_frame: List[float]):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if not swing_speed_per_frame:
        ax.set_title("Swing Speed vs Frame")
        ax.text(0.5, 0.5, "No bat trajectory data", ha="center", va="center")
        return fig

    x = list(range(1, len(swing_speed_per_frame) + 1))
    ax.plot(x, swing_speed_per_frame, marker="o", color="#38BDF8")
    ax.set_title("Swing Speed vs Frame")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed (px/frame)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_bat_arc_angle(arc_angle: float):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(["Bat Arc Angle"], [arc_angle], color="#22C55E")
    ax.set_ylim(0, 180)
    ax.set_ylabel("Degrees")
    ax.set_title("Bat Swing Arc")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_torso_rotation(torso_rotation_series: List[float]):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if not torso_rotation_series:
        ax.set_title("Torso Rotation")
        ax.text(0.5, 0.5, "No 3D pose data", ha="center", va="center")
        return fig

    df = pd.DataFrame({"frame": range(1, len(torso_rotation_series) + 1), "torso_twist": torso_rotation_series})
    ax.plot(df["frame"], df["torso_twist"], color="#F59E0B", linewidth=2)
    ax.set_title("Torso Rotation (Twist) vs Frame")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Degrees")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig
