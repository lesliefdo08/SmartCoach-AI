from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np


def _window_features(window: List[Dict[str, float]]) -> Dict[str, float]:
    keys = window[0].keys()
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.array([float(item.get(key, 0.0)) for item in window], dtype=np.float32)
        out[key] = float(np.mean(vals))
    out["arc_total"] = float(np.sum([w["bat_swing_arc"] for w in window]))
    return out


def _classify_window(w: Dict[str, float]) -> tuple[str, float, str]:
    # Defensive shot
    if (
        w["arc_total"] < 95.0
        and abs(w["bat_angle"]) > 55.0
        and w["body_lean"] < 11.0
        and abs(w["follow_dx"]) < 10.0
        and abs(w["follow_dy"]) < 10.0
    ):
        return "defense", 82.0, "Compact vertical bat path and short follow-through indicate a defensive intent."

    # Drive
    if (
        95.0 <= w["arc_total"] <= 220.0
        and w["follow_dx"] > 3.0
        and 8.0 <= w["body_lean"] <= 22.0
        and w["horizontal_ratio"] < 1.8
    ):
        return "drive", 78.0, "Forward follow-through with controlled arc indicates a drive pattern."

    # Lofted shot
    if (
        w["arc_total"] > 210.0
        and w["upward_ratio"] > 1.1
        and w["wrist_height_norm"] > 0.08
        and w["wrist_velocity"] > 8.0
    ):
        return "lofted_shot", 86.0, "High wrist extension and upward follow-through suggest an attempted lofted shot."

    # Cut / pull
    if (
        w["horizontal_ratio"] >= 1.8
        and w["shoulder_rotation"] > 12.0
        and abs(w["follow_dx"]) > 8.0
    ):
        return "cut_pull", 80.0, "Horizontal bat travel with shoulder rotation indicates a cut/pull profile."

    return "uncertain", 42.0, "Shot pattern is mixed across this temporal segment."


def classify_shot_temporal(
    motion_series: List[Dict[str, float]],
    window_size: int = 15,
) -> Dict[str, object]:
    if not motion_series:
        return {
            "shot_type": "Uncertain Shot – Needs Review",
            "confidence_score": 0.0,
            "probabilities": {},
            "insight": "Insufficient motion frames for temporal classification.",
        }

    win = int(np.clip(window_size, 10, 20))
    if len(motion_series) < win:
        win = max(3, len(motion_series))

    window_votes: List[str] = []
    confidences: List[float] = []
    insights: List[str] = []

    for start in range(0, len(motion_series) - win + 1):
        window = motion_series[start : start + win]
        summary = _window_features(window)
        label, conf, insight = _classify_window(summary)
        window_votes.append(label)
        confidences.append(conf)
        insights.append(insight)

    if not window_votes:
        summary = _window_features(motion_series)
        label, conf, insight = _classify_window(summary)
        window_votes = [label]
        confidences = [conf]
        insights = [insight]

    counts = Counter(window_votes)
    total = len(window_votes)
    winner, vote_count = counts.most_common(1)[0]
    vote_ratio = vote_count / max(total, 1)

    label_conf = float(np.mean([c for v, c in zip(window_votes, confidences) if v == winner]))
    margin = 0.0
    if len(counts) > 1:
        margin = (vote_count - counts.most_common(2)[1][1]) / max(total, 1)
    final_conf = float(np.clip((label_conf * 0.65) + (vote_ratio * 100.0 * 0.25) + (margin * 100.0 * 0.10), 0.0, 100.0))

    probability_map = {
        k: round((v / total) * 100.0, 2)
        for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    }

    if final_conf < 50.0 or winner == "uncertain":
        return {
            "shot_type": "Uncertain Shot – Needs Review",
            "confidence_score": round(final_conf, 2),
            "probabilities": probability_map,
            "insight": "Temporal evidence is inconsistent. Review footwork, bat plane, and follow-through continuity.",
        }

    insight_pool = [i for v, i in zip(window_votes, insights) if v == winner]
    selected_insight = insight_pool[0] if insight_pool else "Shot dynamics were stable over multiple frame windows."

    return {
        "shot_type": winner,
        "confidence_score": round(final_conf, 2),
        "probabilities": probability_map,
        "insight": selected_insight,
    }


def generate_contextual_feedback(
    classification: Dict[str, object],
    biomech_scores: Dict[str, float],
    motion_series: List[Dict[str, float]],
) -> str:
    shot = str(classification.get("shot_type", "Uncertain Shot – Needs Review"))
    confidence = float(classification.get("confidence_score", 0.0))
    insight = str(classification.get("insight", ""))

    if not motion_series:
        return "Pose continuity was low. Re-record from a side-on view with full body visibility."

    avg_body_lean = float(np.mean([m["body_lean"] for m in motion_series]))
    avg_upward = float(np.mean([m["upward_ratio"] for m in motion_series]))

    balance = float(biomech_scores.get("balance_score", 0.0))
    technique = float(biomech_scores.get("technique_score", 0.0))

    if "lofted" in shot and balance < 65.0:
        return f"{insight} Follow-through height indicates an attempted lofted shot, but balance was lost near impact. Keep head stable and reduce excessive spine drift."
    if "drive" in shot and avg_body_lean > 20.0:
        return f"{insight} Drive intent is clear, but forward lean is high. Transfer weight through the front leg without collapsing the torso."
    if "defense" in shot and technique < 70.0:
        return f"{insight} Defensive setup is present; tighten elbow structure and keep bat closer to the pad line."
    if "cut_pull" in shot and avg_upward > 1.0:
        return f"{insight} Side-on shot mechanics are present; avoid lifting too early and keep the bat path flatter through contact."

    return f"{insight} Shot recognized at {confidence:.1f}% confidence with stable biomechanics."
