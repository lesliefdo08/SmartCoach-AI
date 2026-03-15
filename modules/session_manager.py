from __future__ import annotations

from typing import Dict, List

from database.database import get_dashboard_aggregates, get_user_sessions, insert_session


def save_analysis_session(
    user_id: int,
    video_name: str,
    shot_type: str,
    confidence: float,
    technique_score: float,
    balance_score: float,
    consistency_score: float,
) -> None:
    insert_session(
        user_id=user_id,
        video_name=video_name,
        shot_type=shot_type,
        confidence=confidence,
        technique_score=technique_score,
        balance_score=balance_score,
        consistency_score=consistency_score,
    )


def fetch_user_sessions(user_id: int, limit: int = 200) -> List[Dict[str, object]]:
    return get_user_sessions(user_id=user_id, limit=limit)


def fetch_user_dashboard(user_id: int) -> Dict[str, object]:
    return get_dashboard_aggregates(user_id=user_id)
