"""
Layer 1: Raw conversation storage.

Append-first raw provenance for higher memory layers. Explicit maintenance can
archive old rows before deleting them locally; automatic ingestion never
deletes or expires user data.
"""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
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

    def retention_candidates(
        self,
        *,
        older_than: float | None = None,
        keep_latest: int | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return oldest raw rows eligible for explicit retention maintenance."""
        if older_than is None and keep_latest is None:
            raise ValueError("provide older_than or keep_latest")
        if keep_latest is not None and keep_latest < 0:
            raise ValueError("keep_latest must be non-negative")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")

        clauses = []
        parameters: list[float | int] = []
        if older_than is not None:
            clauses.append("timestamp < ?")
            parameters.append(float(older_than))
        if keep_latest is not None:
            clauses.append(
                "rowid NOT IN (SELECT rowid FROM conversations "
                "ORDER BY timestamp DESC, rowid DESC LIMIT ?)"
            )
            parameters.append(int(keep_latest))

        sql = (
            "SELECT id, timestamp, role, content, session_id, metadata "
            "FROM conversations WHERE " + " AND ".join(clauses)
            + " ORDER BY timestamp, rowid"
        )
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(int(limit))
        rows = self.db.execute(sql, parameters).fetchall()
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "role": row[2],
                "content": row[3],
                "session_id": row[4],
                "metadata": json.loads(row[5]) if row[5] else None,
            }
            for row in rows
        ]

    def archive_and_delete(
        self,
        rows: list[dict],
        archive_path: str | Path,
    ) -> dict:
        """Durably append rows to JSONL, then delete them in one DB transaction."""
        if not rows:
            return {"archived": 0, "deleted": 0, "archive_path": str(archive_path)}

        target = Path(archive_path).expanduser()
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"archive target must be a regular file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        with target.open("a", encoding="utf-8") as stream:
            for row in rows:
                record = dict(row)
                record["archive_format"] = "memoria.raw.v1"
                record["archived_at"] = datetime.now(timezone.utc).isoformat()
                canonical = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                record["checksum_sha256"] = hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest()
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not existed:
            try:
                target.chmod(0o600)
            except OSError:
                pass

        ids = [str(row["id"]) for row in rows]
        with self.db:
            cursor = self.db.executemany(
                "DELETE FROM conversations WHERE id = ?",
                [(value,) for value in ids],
            )
        return {
            "archived": len(rows),
            "deleted": max(cursor.rowcount, 0),
            "archive_path": str(target),
        }

    def read_archive(
        self,
        archive_path: str | Path,
        *,
        limit: int | None = None,
    ) -> list[dict]:
        """Read and validate a Memoria raw-provenance JSONL archive."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        source = Path(archive_path).expanduser()
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"archive source must be a regular file: {source}")
        rows = []
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid archive JSON on line {line_number}: {exc}"
                    ) from exc
                required = {"id", "timestamp", "role", "content"}
                if not isinstance(record, dict) or not required.issubset(record):
                    raise ValueError(f"invalid archive record on line {line_number}")
                checksum = record.get("checksum_sha256")
                if checksum:
                    canonical_record = dict(record)
                    canonical_record.pop("checksum_sha256", None)
                    canonical = json.dumps(
                        canonical_record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    if actual != checksum:
                        raise ValueError(
                            f"archive checksum mismatch on line {line_number}"
                        )
                rows.append(record)
                if limit is not None and len(rows) >= limit:
                    break
        return rows

    def restore_archive(self, rows: list[dict]) -> dict:
        """Restore archived raw rows, leaving already-present IDs unchanged."""
        restored = 0
        skipped = 0
        with self.db:
            for row in rows:
                cursor = self.db.execute(
                    """INSERT OR IGNORE INTO conversations
                       (id, timestamp, role, content, session_id, metadata)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        str(row["id"]),
                        float(row["timestamp"]),
                        str(row["role"]),
                        str(row["content"]),
                        row.get("session_id"),
                        json.dumps(row.get("metadata"))
                        if row.get("metadata") is not None
                        else None,
                    ),
                )
                if cursor.rowcount:
                    restored += 1
                else:
                    skipped += 1
        return {"restored": restored, "skipped_existing": skipped}


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
