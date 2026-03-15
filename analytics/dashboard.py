"""Professional analytics dashboard utilities for SmartCoach AI."""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns


def _collect_joint_names(frame_results: List[Dict[str, object]]) -> List[str]:
    joints = set()
    for fr in frame_results:
        joints.update(fr.get("features", {}).keys())
    return sorted(joints)


def build_analysis_frames(
    frame_results: List[Dict[str, object]],
    reference_features: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build dataframes for trends, similarity, deviations, and frame summary."""
    frame_indices = [int(fr["frame_index"]) for fr in frame_results]
    joint_names = _collect_joint_names(frame_results)

    features_rows: List[Dict[str, float]] = []
    deviation_rows: List[Dict[str, float]] = []
    frame_summary_rows: List[Dict[str, float]] = []

    for fr in frame_results:
        idx = int(fr["frame_index"])
        features = fr.get("features", {})
        similarity = float(fr.get("similarity_score", 0.0))

        frow: Dict[str, float] = {"frame": idx}
        drow: Dict[str, float] = {"frame": idx}

        abs_dev_values: List[float] = []
        for joint in joint_names:
            detected = float(features.get(joint, np.nan))
            reference = float(reference_features.get(joint, np.nan))
            dev = abs(detected - reference) if np.isfinite(detected) and np.isfinite(reference) else np.nan
            frow[joint] = detected
            drow[joint] = dev
            if np.isfinite(dev):
                abs_dev_values.append(float(dev))

        features_rows.append(frow)
        deviation_rows.append(drow)

        frame_summary_rows.append(
            {
                "Frame": idx,
                "Similarity": similarity,
                "Mean Abs Joint Deviation": round(float(np.mean(abs_dev_values)) if abs_dev_values else 0.0, 2),
                "Worst Joint": _worst_joint_name(drow),
                "Worst Joint Deviation": round(float(np.nanmax(list(drow.values())[1:])) if len(drow) > 1 else 0.0, 2),
            }
        )

    features_df = pd.DataFrame(features_rows).sort_values("frame") if features_rows else pd.DataFrame()
    similarity_df = pd.DataFrame({"frame": frame_indices, "similarity": [fr["similarity_score"] for fr in frame_results]}).sort_values("frame")
    deviation_df = pd.DataFrame(deviation_rows).sort_values("frame") if deviation_rows else pd.DataFrame()
    frame_table_df = pd.DataFrame(frame_summary_rows).sort_values("Frame") if frame_summary_rows else pd.DataFrame()

    return features_df, similarity_df, deviation_df, frame_table_df


def _worst_joint_name(deviation_row: Dict[str, float]) -> str:
    pairs = [(k, v) for k, v in deviation_row.items() if k != "frame" and np.isfinite(v)]
    if not pairs:
        return "n/a"
    worst = max(pairs, key=lambda p: p[1])[0]
    return worst


def plot_joint_angle_trends(features_df: pd.DataFrame):
    """Angle vs frame chart for all joint features."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    if features_df.empty:
        ax.set_title("Joint Angle Trends")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    value_cols = [c for c in features_df.columns if c != "frame"]
    long_df = features_df.melt(id_vars=["frame"], value_vars=value_cols, var_name="joint", value_name="angle")

    sns.lineplot(data=long_df, x="frame", y="angle", hue="joint", ax=ax, linewidth=1.8)
    ax.set_title("Joint Angle Trends Across Frames")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (degrees)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)
    fig.tight_layout()
    return fig


def plot_similarity_timeline(similarity_df: pd.DataFrame):
    """Pose similarity graph across frames."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    if similarity_df.empty:
        ax.set_title("Pose Similarity Timeline")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    sns.lineplot(data=similarity_df, x="frame", y="similarity", marker="o", color="#00C2FF", ax=ax)
    ax.fill_between(similarity_df["frame"], similarity_df["similarity"], color="#00C2FF", alpha=0.12)
    ax.set_ylim(0, 100)
    ax.set_title("Pose Similarity Timeline")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Similarity (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def plot_posture_heatmap(deviation_df: pd.DataFrame):
    """Posture heatmap using absolute joint deviations per frame."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if deviation_df.empty:
        ax.set_title("Posture Heatmap")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    heat = deviation_df.set_index("frame").transpose()
    sns.heatmap(heat, cmap="YlOrRd", linewidths=0.1, cbar_kws={"label": "Abs Deviation (deg)"}, ax=ax)
    ax.set_title("Posture Heatmap (Joint Deviation by Frame)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Joint Feature")
    fig.tight_layout()
    return fig


def plot_joint_deviation_radar(deviation_df: pd.DataFrame):
    """Radar chart showing average deviation by joint."""
    if deviation_df.empty:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        ax.set_title("Joint Deviation Radar")
        ax.text(0.0, 0.0, "No data")
        return fig

    avg_dev = deviation_df.drop(columns=["frame"], errors="ignore").mean(axis=0, numeric_only=True)
    labels = list(avg_dev.index)
    values = avg_dev.values.astype(float)

    if len(labels) == 0:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        ax.set_title("Joint Deviation Radar")
        ax.text(0.0, 0.0, "No data")
        return fig

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    values_closed = np.concatenate([values, [values[0]]])
    angles_closed = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.plot(angles_closed, values_closed, color="#F97316", linewidth=2)
    ax.fill(angles_closed, values_closed, color="#FDBA74", alpha=0.35)
    ax.set_xticks(angles)
    ax.set_xticklabels([lbl.replace("_", " ") for lbl in labels], fontsize=8)
    ax.set_title("Joint Deviation Radar Chart")
    ax.grid(alpha=0.35)
    fig.tight_layout()
    return fig


def summarize_mistakes_and_suggestions(
    deviation_df: pd.DataFrame,
    threshold: float = 12.0,
) -> Dict[str, object]:
    """Return top mistakes and improvement suggestions from deviation data."""
    if deviation_df.empty:
        return {"top_mistakes": [], "suggestions": ["Collect more valid pose frames for analysis."]}

    avg_dev = deviation_df.drop(columns=["frame"], errors="ignore").mean(axis=0, numeric_only=True).sort_values(ascending=False)
    top_mistakes = [(joint, float(dev)) for joint, dev in avg_dev.head(5).items() if np.isfinite(dev)]

    suggestions: List[str] = []
    for joint, dev in top_mistakes:
        if dev < threshold:
            continue
        if "elbow" in joint:
            suggestions.append("Focus on elbow control: keep elbows higher and aligned through impact.")
        elif "knee" in joint:
            suggestions.append("Improve base stability: maintain active knee flexion during setup and transfer.")
        elif "shoulder" in joint:
            suggestions.append("Stabilize shoulder rotation and avoid early opening of the upper body.")
        elif "hip" in joint:
            suggestions.append("Maintain hip level and controlled rotation for balance and power transfer.")
        elif "spine" in joint:
            suggestions.append("Keep spine angle steady and head balanced over the base.")

    if not suggestions:
        suggestions.append("Good consistency. Keep practicing with tempo control and repeatable setup.")

    return {
        "top_mistakes": top_mistakes,
        "suggestions": suggestions,
    }


def export_report_csv(frame_table_df: pd.DataFrame) -> bytes:
    """Export frame-by-frame analysis table as CSV bytes."""
    return frame_table_df.to_csv(index=False).encode("utf-8")


def export_report_pdf(
    overall_posture_score: float,
    detected_shot: str,
    confidence_score: float,
    top_mistakes: List[Tuple[str, float]],
    suggestions: List[str],
    figures: List,
) -> bytes:
    """Export dashboard summary and charts as a PDF report."""
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        fig_cover, ax_cover = plt.subplots(figsize=(8.27, 11.69))
        ax_cover.axis("off")
        lines = [
            "SmartCoach AI - Performance Report",
            "",
            f"Detected Shot: {detected_shot.replace('_', ' ').title()}",
            f"Shot Confidence: {confidence_score:.2f}%",
            f"Overall Posture Score: {overall_posture_score:.2f}%",
            "",
            "Top Mistakes:",
        ]
        for joint, dev in top_mistakes[:5]:
            lines.append(f"- {joint}: {dev:.2f} deg")
        lines.append("")
        lines.append("Improvement Suggestions:")
        for s in suggestions[:8]:
            lines.append(f"- {s}")

        ax_cover.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", fontsize=11)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        for fig in figures:
            pdf.savefig(fig)

    buffer.seek(0)
    return buffer.read()
