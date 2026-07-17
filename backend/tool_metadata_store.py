"""Persistent metadata for generated and legacy ProgramAT tools."""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DB_PATH = Path(
    os.environ.get(
        "TOOL_METADATA_DB_PATH",
        Path(__file__).resolve().parent / "tool_metadata.db",
    )
)
_DB_LOCK = threading.RLock()


def _connect(path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_difficulty_metadata (
            tool_id TEXT PRIMARY KEY,
            prompt_sha256 TEXT NOT NULL,
            difficulty_start TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return connection


def get_difficulty_start(
    tool_id: str,
    prompt_sha256: str,
    *,
    path: Optional[Path] = None,
) -> Optional[str]:
    """Return a cached value only when it matches the current prompt hash."""
    with _DB_LOCK:
        connection = _connect(path)
        try:
            row = connection.execute(
                """
                SELECT difficulty_start
                FROM tool_difficulty_metadata
                WHERE tool_id = ? AND prompt_sha256 = ?
                """,
                (tool_id, prompt_sha256),
            ).fetchone()
        finally:
            connection.close()
    return str(row["difficulty_start"]) if row else None


def put_difficulty_start(
    tool_id: str,
    prompt_sha256: str,
    difficulty_start: str,
    *,
    path: Optional[Path] = None,
) -> None:
    """Persist one current prediction, replacing stale prompt metadata."""
    with _DB_LOCK:
        connection = _connect(path)
        try:
            connection.execute(
                """
                INSERT INTO tool_difficulty_metadata (
                    tool_id, prompt_sha256, difficulty_start, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(tool_id) DO UPDATE SET
                    prompt_sha256 = excluded.prompt_sha256,
                    difficulty_start = excluded.difficulty_start,
                    updated_at = excluded.updated_at
                """,
                (
                    tool_id,
                    prompt_sha256,
                    difficulty_start,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()
        finally:
            connection.close()


__all__ = ["DB_PATH", "get_difficulty_start", "put_difficulty_start"]
