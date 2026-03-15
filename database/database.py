from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DB_DIR = ROOT_DIR / "database"
DB_PATH = DB_DIR / "smartcoach.db"


def _get_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                video_name TEXT NOT NULL,
                shot_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                technique_score REAL NOT NULL,
                balance_score REAL NOT NULL,
                consistency_score REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()


def create_user(username: str, email: str, password_hash: str) -> tuple[bool, str]:
    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username.strip(), email.strip().lower(), password_hash, datetime.utcnow().isoformat()),
            )
            conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def get_user_by_username_or_email(identifier: str) -> Optional[Dict[str, object]]:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, email, password_hash, created_at FROM users WHERE username = ? OR email = ?",
            (identifier.strip(), identifier.strip().lower()),
        ).fetchone()
    return dict(row) if row else None


def insert_session(
    user_id: int,
    video_name: str,
    shot_type: str,
    confidence: float,
    technique_score: float,
    balance_score: float,
    consistency_score: float,
) -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                user_id, video_name, shot_type, confidence,
                technique_score, balance_score, consistency_score, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                video_name,
                shot_type,
                float(confidence),
                float(technique_score),
                float(balance_score),
                float(consistency_score),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


def get_user_sessions(user_id: int, limit: int = 200) -> List[Dict[str, object]]:
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, video_name, shot_type, confidence, technique_score,
                   balance_score, consistency_score, timestamp
            FROM sessions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_dashboard_aggregates(user_id: int) -> Dict[str, object]:
    sessions = get_user_sessions(user_id=user_id, limit=500)
    if not sessions:
        return {
            "total_sessions": 0,
            "avg_technique": 0.0,
            "avg_balance": 0.0,
            "avg_consistency": 0.0,
            "shot_distribution": {},
            "recent": [],
        }

    total = len(sessions)
    avg_technique = sum(float(s["technique_score"]) for s in sessions) / total
    avg_balance = sum(float(s["balance_score"]) for s in sessions) / total
    avg_consistency = sum(float(s["consistency_score"]) for s in sessions) / total

    shot_distribution: Dict[str, int] = {}
    for s in sessions:
        shot = str(s["shot_type"])
        shot_distribution[shot] = shot_distribution.get(shot, 0) + 1

    return {
        "total_sessions": total,
        "avg_technique": round(avg_technique, 2),
        "avg_balance": round(avg_balance, 2),
        "avg_consistency": round(avg_consistency, 2),
        "shot_distribution": shot_distribution,
        "recent": sessions[:15],
    }
