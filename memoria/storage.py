"""
Layer 1: Raw conversation storage.

Append-only, never mutated. The ground truth that all higher layers
point back to. You never search this layer directly — it exists
as the source of truth for provenance.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


class ConversationStore:
    """Manages raw conversation logs."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def append(
        self,
        content: str,
        role: str = "user",
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Append a conversation turn. Returns the conversation ID."""
        conv_id = str(uuid.uuid4())
        self.db.execute(
            """INSERT INTO conversations (id, timestamp, role, content, session_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                conv_id,
                time.time(),
                role,
                content,
                session_id,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self.db.commit()
        return conv_id

    def get(self, conv_id: str) -> dict | None:
        row = self.db.execute(
            "SELECT id, timestamp, role, content, session_id, metadata FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "timestamp": row[1],
            "role": row[2],
            "content": row[3],
            "session_id": row[4],
            "metadata": json.loads(row[5]) if row[5] else None,
        }

    def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over conversation content."""
        rows = self.db.execute(
            """SELECT c.id, c.timestamp, c.role, c.content, c.session_id
               FROM conversations_fts f
               JOIN conversations c ON c.rowid = f.rowid
               WHERE conversations_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "role": r[2], "content": r[3], "session_id": r[4]}
            for r in rows
        ]

    def get_session(self, session_id: str) -> list[dict]:
        """Get all turns in a session, ordered by time."""
        rows = self.db.execute(
            """SELECT id, timestamp, role, content, session_id, metadata
               FROM conversations WHERE session_id = ?
               ORDER BY timestamp""",
            (session_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "timestamp": r[1],
                "role": r[2],
                "content": r[3],
                "session_id": r[4],
                "metadata": json.loads(r[5]) if r[5] else None,
            }
            for r in rows
        ]

    def recent(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            """SELECT id, timestamp, role, content, session_id
               FROM conversations ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "role": r[2], "content": r[3], "session_id": r[4]}
            for r in rows
        ]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]


def export_session_markdown(store: ConversationStore, session_id: str) -> str:
    """Export a conversation session as markdown for human review."""
    turns = store.get_session(session_id)
    lines = [f"# Session {session_id}\n"]
    for t in turns:
        role = t["role"].upper()
        lines.append(f"**{role}** ({time.strftime('%Y-%m-%d %H:%M', time.localtime(t['timestamp']))})")
        lines.append(t["content"])
        lines.append("")
    return "\n".join(lines)
