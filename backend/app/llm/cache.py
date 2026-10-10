"""SQLite-backed response cache and in-memory rate limiter for LLM calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any

from app.core.config import settings
from app.llm.schemas import LLMRateLimitError

_LOCK = threading.Lock()


def _get_sqlite_path() -> str:
    """Extract SQLite database filepath from settings.DATABASE_URL."""
    url = settings.DATABASE_URL
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "")
    return "sme_credit.db"


def init_cache_db() -> None:
    """Ensure llm_cache table exists in SQLite database."""
    db_path = _get_sqlite_path()
    with _LOCK, sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0.0,
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()


def compute_cache_key(
    task: str,
    prompt_version: str,
    system_prompt: str,
    rendered_prompt: str,
    facts: dict[str, Any] | None = None,
) -> str:
    """Generate deterministic 64-char hex SHA-256 cache key."""
    facts_str = json.dumps(facts or {}, sort_keys=True)
    payload = f"{task}:{prompt_version}:{system_prompt}:{rendered_prompt}:{facts_str}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached_response(key: str) -> dict[str, Any] | None:
    """Retrieve cached LLM result if available."""
    init_cache_db()
    db_path = _get_sqlite_path()
    try:
        with _LOCK, sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT response_json, prompt_tokens, completion_tokens, cost_usd "
                "FROM llm_cache WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "response_json": row[0],
                    "prompt_tokens": row[1],
                    "completion_tokens": row[2],
                    "cost_usd": row[3],
                }
    except Exception:
        pass
    return None


def set_cached_response(
    key: str,
    response_json: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Store LLM result in SQLite cache."""
    init_cache_db()
    db_path = _get_sqlite_path()
    try:
        with _LOCK, sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                (key, response_json, prompt_tokens, completion_tokens, cost_usd, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, response_json, prompt_tokens, completion_tokens, cost_usd, time.time()),
            )
            conn.commit()
    except Exception:
        pass


def count_cached() -> int:
    """Count number of entries in llm_cache."""
    init_cache_db()
    db_path = _get_sqlite_path()
    try:
        with _LOCK, sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM llm_cache")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def clear_cache() -> None:
    """Clear all entries in llm_cache."""
    init_cache_db()
    db_path = _get_sqlite_path()
    try:
        with _LOCK, sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM llm_cache")
            conn.commit()
    except Exception:
        pass


class RateLimiter:
    """Sliding-window in-memory rate limiter."""

    def __init__(self, max_rpm: int | None = None):
        self.max_rpm = max_rpm or settings.LLM_RATE_LIMIT_RPM
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Check if request is within limit; raises LLMRateLimitError if exceeded."""
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            # Purge timestamps older than 60 seconds
            self._timestamps = [t for t in self._timestamps if t > window_start]
            if len(self._timestamps) >= self.max_rpm:
                raise LLMRateLimitError(
                    f"LLM Rate limit exceeded: {len(self._timestamps)} requests "
                    f"in last 60 seconds (max {self.max_rpm} RPM)"
                )
            self._timestamps.append(now)

    def reset(self) -> None:
        """Reset internal rate limit counter."""
        with self._lock:
            self._timestamps.clear()


rate_limiter = RateLimiter()
