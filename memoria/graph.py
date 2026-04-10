"""
Layer 2: Knowledge graph — the primary retrieval target.

Entity-Attribute-Value triples with temporal validity windows,
causal/supersedes/depends edges, and confidence scores.
Contradiction detection on the write path.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Entity:
    id: str
    name: str
    entity_type: str | None = None
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float | None = None
    embedding: np.ndarray | None = None


@dataclass
class Triple:
    id: str
    subject_id: str
    predicate: str
    object_id: str | None = None
    object_value: str | None = None
    relation_type: str = "fact"
    confidence: float = 1.0
    valid_from: float | None = None
    valid_until: float | None = None
    source_ref: str | None = None
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float | None = None


class KnowledgeGraph:
    """SQLite-backed knowledge graph with contradiction detection."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    # --- Entity operations ---

    def add_entity(self, name: str, entity_type: str | None = None,
                   confidence: float = 1.0, embedding: np.ndarray | None = None) -> str:
        """Add or return existing entity. Deduplicates by (name, entity_type)."""
        existing = self.db.execute(
            "SELECT id FROM entities WHERE name = ? AND entity_type IS ?",
            (name, entity_type),
        ).fetchone()
        if existing:
            return existing[0]

        eid = str(uuid.uuid4())
        emb_bytes = embedding.tobytes() if embedding is not None else None
        self.db.execute(
            """INSERT INTO entities (id, name, entity_type, created_at, confidence, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (eid, name, entity_type, time.time(), confidence, emb_bytes),
        )
        self.db.commit()
        return eid

    def get_entity(self, entity_id: str) -> Entity | None:
        row = self.db.execute(
            """SELECT id, name, entity_type, created_at, confidence,
                      access_count, last_accessed, embedding
               FROM entities WHERE id = ?""",
            (entity_id,),
        ).fetchone()
        if not row:
            return None
        emb = np.frombuffer(row[7], dtype=np.float64) if row[7] else None
        return Entity(
            id=row[0], name=row[1], entity_type=row[2], created_at=row[3],
            confidence=row[4], access_count=row[5], last_accessed=row[6],
            embedding=emb,
        )

    def find_entities(self, name: str) -> list[Entity]:
        """Find entities by name (case-insensitive prefix match)."""
        rows = self.db.execute(
            """SELECT id, name, entity_type, created_at, confidence,
                      access_count, last_accessed
               FROM entities WHERE LOWER(name) LIKE LOWER(?) || '%'
               ORDER BY confidence DESC, access_count DESC
               LIMIT 20""",
            (name,),
        ).fetchall()
        return [
            Entity(id=r[0], name=r[1], entity_type=r[2], created_at=r[3],
                   confidence=r[4], access_count=r[5], last_accessed=r[6])
            for r in rows
        ]

    def entity_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    # --- Triple operations ---

    def add_triple(
        self,
        subject_id: str,
        predicate: str,
        object_id: str | None = None,
        object_value: str | None = None,
        relation_type: str = "fact",
        confidence: float = 1.0,
        valid_from: float | None = None,
        source_ref: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Add a triple, detecting contradictions.

        Returns (triple_id, contradictions) where contradictions is a list
        of existing triples that conflict with this one.
        """
        contradictions = []
        if relation_type == "fact":
            contradictions = self._detect_contradictions(
                subject_id, predicate, object_id, object_value
            )

        tid = str(uuid.uuid4())
        now = time.time()
        self.db.execute(
            """INSERT INTO triples
               (id, subject_id, predicate, object_id, object_value,
                relation_type, confidence, valid_from, valid_until,
                source_ref, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
            (tid, subject_id, predicate, object_id, object_value,
             relation_type, confidence, valid_from or now, source_ref, now),
        )

        # If contradictions found, create supersedes edges and close old facts
        for old in contradictions:
            # Close the old triple's validity window
            self.db.execute(
                "UPDATE triples SET valid_until = ? WHERE id = ?",
                (now, old["id"]),
            )
            # Create supersedes edge
            self.db.execute(
                """INSERT INTO triples
                   (id, subject_id, predicate, object_id, object_value,
                    relation_type, confidence, valid_from, source_ref, created_at)
                   VALUES (?, ?, 'supersedes', ?, NULL, 'supersedes', ?, ?, ?, ?)""",
                (str(uuid.uuid4()), tid, old["id"], confidence, now, source_ref, now),
            )

        self.db.commit()
        return (tid, contradictions)

    def _detect_contradictions(
        self, subject_id: str, predicate: str,
        object_id: str | None, object_value: str | None,
    ) -> list[dict]:
        """Find active triples about the same (subject, predicate) with different values."""
        rows = self.db.execute(
            """SELECT id, subject_id, predicate, object_id, object_value,
                      confidence, valid_from, source_ref
               FROM triples
               WHERE subject_id = ? AND predicate = ?
                 AND relation_type = 'fact'
                 AND valid_until IS NULL""",
            (subject_id, predicate),
        ).fetchall()

        contradictions = []
        for r in rows:
            old_obj = r[3]
            old_val = r[4]
            # Different object entity or different literal value
            if (object_id and old_obj and object_id != old_obj) or \
               (object_value and old_val and object_value != old_val):
                contradictions.append({
                    "id": r[0], "subject_id": r[1], "predicate": r[2],
                    "object_id": r[3], "object_value": r[4],
                    "confidence": r[5], "valid_from": r[6], "source_ref": r[7],
                })
        return contradictions

    def get_triples(
        self,
        subject_id: str | None = None,
        predicate: str | None = None,
        object_id: str | None = None,
        active_only: bool = True,
        relation_type: str | None = None,
        as_of: float | None = None,
    ) -> list[dict]:
        """Query triples with optional filters."""
        conditions = []
        params = []

        if subject_id:
            conditions.append("subject_id = ?")
            params.append(subject_id)
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if object_id:
            conditions.append("object_id = ?")
            params.append(object_id)
        if relation_type:
            conditions.append("relation_type = ?")
            params.append(relation_type)
        if active_only and as_of is None:
            conditions.append("valid_until IS NULL")
        if as_of is not None:
            conditions.append("valid_from <= ? AND (valid_until IS NULL OR valid_until > ?)")
            params.extend([as_of, as_of])

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self.db.execute(
            f"""SELECT id, subject_id, predicate, object_id, object_value,
                       relation_type, confidence, valid_from, valid_until,
                       source_ref, created_at, access_count
                FROM triples WHERE {where}
                ORDER BY confidence DESC, created_at DESC""",
            params,
        ).fetchall()

        return [
            {
                "id": r[0], "subject_id": r[1], "predicate": r[2],
                "object_id": r[3], "object_value": r[4], "relation_type": r[5],
                "confidence": r[6], "valid_from": r[7], "valid_until": r[8],
                "source_ref": r[9], "created_at": r[10], "access_count": r[11],
            }
            for r in rows
        ]

    def neighbors(self, entity_id: str, max_depth: int = 1, active_only: bool = True) -> dict:
        """BFS neighbor walk from entity. Returns {depth: [triples]}."""
        result = {}
        visited_entities = {entity_id}
        frontier = {entity_id}

        for depth in range(1, max_depth + 1):
            triples_at_depth = []
            next_frontier = set()

            for eid in frontier:
                # Outgoing
                for t in self.get_triples(subject_id=eid, active_only=active_only):
                    triples_at_depth.append(t)
                    if t["object_id"] and t["object_id"] not in visited_entities:
                        next_frontier.add(t["object_id"])
                # Incoming
                for t in self.get_triples(object_id=eid, active_only=active_only):
                    triples_at_depth.append(t)
                    if t["subject_id"] not in visited_entities:
                        next_frontier.add(t["subject_id"])

            if triples_at_depth:
                result[depth] = triples_at_depth

            visited_entities.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        return result

    def touch(self, triple_ids: list[str]):
        """Update access count and last_accessed for retrieved triples."""
        now = time.time()
        for tid in triple_ids:
            self.db.execute(
                "UPDATE triples SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, tid),
            )
        # Also touch the entities involved
        self.db.execute(
            f"""UPDATE entities SET access_count = access_count + 1, last_accessed = ?
                WHERE id IN (
                    SELECT DISTINCT subject_id FROM triples WHERE id IN ({','.join('?' * len(triple_ids))})
                    UNION
                    SELECT DISTINCT object_id FROM triples WHERE id IN ({','.join('?' * len(triple_ids))}) AND object_id IS NOT NULL
                )""",
            [now] + triple_ids + triple_ids,
        )
        self.db.commit()

    def all_triples_for_spectral(self) -> tuple[list[dict], dict[str, int]]:
        """Get all active triples and entity index for spectral analysis."""
        entities = self.db.execute(
            "SELECT id FROM entities ORDER BY id"
        ).fetchall()
        entity_index = {r[0]: i for i, r in enumerate(entities)}

        triples = self.get_triples(active_only=True)
        # Filter to triples with entity objects (not literal-value-only)
        edge_triples = [t for t in triples if t["object_id"] is not None]
        return edge_triples, entity_index

    def triple_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    def history(self, subject_id: str, predicate: str) -> list[dict]:
        """Get the full history of a (subject, predicate) pair, including superseded values."""
        return self.get_triples(
            subject_id=subject_id, predicate=predicate, active_only=False
        )
