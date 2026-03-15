from __future__ import annotations

import bcrypt

from database.database import create_user, get_user_by_username_or_email


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def signup_user(username: str, email: str, password: str) -> tuple[bool, str]:
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if "@" not in email or "." not in email:
        return False, "Enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    return create_user(username=username, email=email, password_hash=hash_password(password))


def login_user(identifier: str, password: str) -> tuple[bool, str, dict | None]:
    user = get_user_by_username_or_email(identifier)
    if not user:
        return False, "Account not found.", None
    if not verify_password(password, str(user["password_hash"])):
        return False, "Invalid password.", None
    return True, "Login successful.", user
