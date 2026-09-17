"""
db.py
Data-access layer. All persistence goes through this module so the rest of
the app never writes raw SQL elsewhere (maintainability / separation of
concerns). Uses SQLite so the whole project runs with zero external DB
server setup.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Tuple

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    record_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    date        TEXT NOT NULL,
    time        TEXT NOT NULL,
    confidence  REAL NOT NULL,
    emotion     TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    UNIQUE (user_id, date)  -- one attendance mark per user per day
);
"""


class Database:
    """Thin wrapper around sqlite3 with context-managed connections."""

    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # ---- Users -------------------------------------------------------
    def add_user(self, name: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (name, created_at) VALUES (?, ?)",
                (name, datetime.now().isoformat(timespec="seconds")),
            )
            return cur.lastrowid

    def get_user_by_name(self, name: str) -> Optional[Tuple[int, str]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, name FROM users WHERE name = ?", (name,)
            ).fetchone()
            return row

    def get_user_by_id(self, user_id: int) -> Optional[Tuple[int, str]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, name FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return row

    def list_users(self) -> List[Tuple[int, str]]:
        with self._connect() as conn:
            return conn.execute("SELECT user_id, name FROM users ORDER BY name").fetchall()

    # ---- Attendance ----------------------------------------------------
    def mark_attendance(
        self, user_id: int, date: str, time: str, confidence: float, emotion: Optional[str]
    ) -> bool:
        """Returns True if a new record was inserted, False if already marked today."""
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO attendance (user_id, date, time, confidence, emotion) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, date, time, confidence, emotion),
                )
            return True
        except sqlite3.IntegrityError:
            return False  # already marked for that date

    def get_attendance(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Tuple]:
        query = (
            "SELECT a.record_id, u.name, a.date, a.time, a.confidence, a.emotion "
            "FROM attendance a JOIN users u ON a.user_id = u.user_id"
        )
        params: List[str] = []
        conditions = []
        if start_date:
            conditions.append("a.date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("a.date <= ?")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY a.date, a.time"
        with self._connect() as conn:
            return conn.execute(query, params).fetchall()
