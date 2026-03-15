"""Rule-based coaching feedback engine."""

from __future__ import annotations

from typing import Dict, List


def generate_feedback(
    detected_features: Dict[str, float],
    reference_features: Dict[str, float],
    joint_errors: List[Dict[str, float]],
    predicted_mistakes: List[Dict[str, float]] | None = None,
) -> Dict[str, object]:
    """Generate structured feedback messages for the athlete.

    Priority source: AI predicted mistakes.
    Fallback source: angle/range rule heuristics.
    """
    tips: List[str] = []

    def ref(name: str, default: float = 0.0) -> float:
        return float(reference_features.get(name, default))

    # AI-driven feedback
    if predicted_mistakes:
        for item in predicted_mistakes:
            label = str(item.get("label", ""))
            tips.append(_mistake_tip(label))
        tips = _dedupe_preserve_order(tips)

    # Fallback heuristic guidance when AI labels are absent
    if not tips:
        if detected_features.get("left_elbow_angle", ref("left_elbow_angle")) < ref("left_elbow_angle") - 15:
            tips.append("Raise the front elbow higher.")
        if detected_features.get("right_elbow_angle", ref("right_elbow_angle")) < ref("right_elbow_angle") - 15:
            tips.append("Keep your back-arm elbow more elevated through the swing.")

        if abs(detected_features.get("spine_tilt", ref("spine_tilt")) - ref("spine_tilt")) > 8:
            tips.append("Keep your head aligned with the ball.")

        if detected_features.get("left_knee_angle", ref("left_knee_angle")) > ref("left_knee_angle") + 15:
            tips.append("Bend the front knee for better balance.")
        if detected_features.get("right_knee_angle", ref("right_knee_angle")) > ref("right_knee_angle") + 15:
            tips.append("Soften the back knee to improve stability.")

        if abs(detected_features.get("hip_alignment", ref("hip_alignment")) - ref("hip_alignment")) > 8:
            tips.append("Level your hips to maintain a stable base.")

    if not tips:
        tips.append("Great posture. Maintain this shape through impact.")

    top_joint_issues = [
        {
            "joint": issue["joint"],
            "error": round(float(issue["error"]), 2),
            "message": _joint_message(issue["joint"], float(issue["error"])),
        }
        for issue in joint_errors[:3]
        if issue.get("abs_error", 0.0) >= 10.0
    ]

    summary = (
        "Technique is stable with minor refinements suggested."
        if len(top_joint_issues) <= 1 and len(tips) <= 2
        else "Multiple technique deviations detected. Focus on the priority corrections below."
    )

    return {
        "summary": summary,
        "tips": tips,
        "joint_feedback": top_joint_issues,
        "detected_mistakes": predicted_mistakes or [],
    }


def _joint_message(joint: str, error: float) -> str:
    direction = "increase" if error < 0 else "reduce"
    joint_name = joint.replace("_", " ").replace(" angle", "")
    return f"{joint_name.title()}: {direction} by about {abs(error):.1f}°"


def _mistake_tip(label: str) -> str:
    mapping = {
        "early_bat_swing": "Delay your bat downswing slightly; let the front foot and head position settle first.",
        "low_elbow": "Raise your front elbow to keep a stronger bat path through the line.",
        "poor_head_alignment": "Keep your head steady and aligned over the ball path.",
        "straight_front_leg": "Add controlled bend in the front knee for better balance and transfer.",
        "imbalanced_stance": "Widen and stabilize your base; keep hips and shoulders balanced.",
    }
    return mapping.get(label, f"Work on correcting: {label.replace('_', ' ')}.")


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
