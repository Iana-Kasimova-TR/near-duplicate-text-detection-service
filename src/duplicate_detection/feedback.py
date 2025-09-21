import sqlite3
from pathlib import Path
from typing import Optional


def _resolve_sqlite_path(database_url: str) -> Path:
    """Convert a SQLite URL or filesystem path into a Path instance."""
    if database_url.startswith("sqlite:///"):
        return Path(database_url[len("sqlite:///") :])
    return Path(database_url)


def save_feedback(
    database_url: str,
    document_id: str,
    duplicate_id: str,
    verdict: str,
    notes: Optional[str] = None,
) -> None:
    """Persist duplicate-review feedback.

    Args:
        database_url: Location of the SQLite database (e.g. "feedback.db" or
            "sqlite:///var/data/feedback.db").
        document_id: ID of the document under review.
        duplicate_id: ID of the document flagged as duplicate.
        verdict: Reviewer decision (e.g. "confirm", "reject", "needs_edit").
        notes: Optional free-form comments from the reviewer.
    """

    db_path = _resolve_sqlite_path(database_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS duplicate_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                duplicate_id TEXT NOT NULL,
                verdict TEXT NOT NULL,
                notes TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO duplicate_feedback (document_id, duplicate_id, verdict, notes)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, duplicate_id, verdict, notes),
        )
        conn.commit()
    finally:
        conn.close()
