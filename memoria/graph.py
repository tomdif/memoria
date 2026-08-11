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

from .embeddings import deserialize_embedding, serialize_embedding


# These predicates describe additive relationships. Other fact predicates are
# treated as stateful attributes unless the caller explicitly overrides the
# behavior with replace_existing=False.
MULTI_VALUED_PREDICATES = {
    "caused",
    "contains",
    "depends",
    "depends_on",
    "includes",
    "implements",
    "knows",
    "likes",
    "member_of",
    "related_to",
    "requires",
    "supports",
    "uses",
    "works_on",
    "works_with",
}


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
        """Add or return an entity, deduplicating names case-insensitively."""
        # Exact match on (name, type) first
        existing = self.db.execute(
            "SELECT id FROM entities WHERE LOWER(name) = LOWER(?) AND entity_type IS ?",
            (name, entity_type),
        ).fetchone()
        if existing:
            if embedding is not None:
                self.db.execute(
                    "UPDATE entities SET embedding = COALESCE(embedding, ?) WHERE id = ?",
                    (serialize_embedding(embedding), existing[0]),
                )
                self.db.commit()
            return existing[0]
        # Fall back to name-only match to prevent duplicates
        existing = self.db.execute(
            "SELECT id FROM entities WHERE LOWER(name) = LOWER(?) LIMIT 1",
            (name,),
        ).fetchone()
        if existing:
            # Fill missing metadata without overwriting an established type.
            self.db.execute(
                """UPDATE entities
                   SET entity_type = COALESCE(entity_type, ?),
                       embedding = COALESCE(embedding, ?)
                   WHERE id = ?""",
                (entity_type, serialize_embedding(embedding), existing[0]),
            )
            self.db.commit()
            return existing[0]

        eid = str(uuid.uuid4())
        emb_bytes = serialize_embedding(embedding)
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
        emb = deserialize_embedding(row[7])
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
        replace_existing: bool | None = None,
    ) -> tuple[str, list[dict]]:
        """Add a triple, detecting contradictions.

        Returns (triple_id, contradictions) where contradictions is a list
        of existing triples that conflict with this one.
        """
        contradictions = []
        if replace_existing is None:
            replace_existing = (
                relation_type == "fact"
                and predicate.lower() not in MULTI_VALUED_PREDICATES
            )

        if relation_type == "fact" and replace_existing:
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

        # If contradictions found, close old facts (valid_until marks them as superseded)
        for old in contradictions:
            self.db.execute(
                "UPDATE triples SET valid_until = ? WHERE id = ?",
                (now, old["id"]),
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

    def neighbors(
        self,
        entity_id: str,
        max_depth: int = 1,
        active_only: bool = True,
        as_of: float | None = None,
    ) -> dict:
        """BFS neighbor walk from entity. Returns {depth: [triples]}."""
        result = {}
        visited_entities = {entity_id}
        frontier = {entity_id}

        for depth in range(1, max_depth + 1):
            triples_at_depth = []
            next_frontier = set()

            for eid in frontier:
                # Outgoing
                for t in self.get_triples(
                    subject_id=eid, active_only=active_only, as_of=as_of
                ):
                    triples_at_depth.append(t)
                    if t["object_id"] and t["object_id"] not in visited_entities:
                        next_frontier.add(t["object_id"])
                # Incoming
                for t in self.get_triples(
                    object_id=eid, active_only=active_only, as_of=as_of
                ):
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

    def delete_entity(self, entity_id: str) -> int:
        """Delete an entity and all its triples (both as subject and object). Returns count of deleted triples."""
        # Delete triples where this entity is subject or object
        c1 = self.db.execute(
            "DELETE FROM triples WHERE subject_id = ?", (entity_id,)
        ).rowcount
        c2 = self.db.execute(
            "DELETE FROM triples WHERE object_id = ?", (entity_id,)
        ).rowcount
        # Delete the entity itself
        self.db.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        self.db.commit()
        return c1 + c2

    def delete_triple(self, triple_id: str) -> bool:
        """Hard-delete a specific triple. Returns True if found."""
        result = self.db.execute(
            "DELETE FROM triples WHERE id = ?", (triple_id,)
        )
        self.db.commit()
        return result.rowcount > 0

    def merge_entities(self, keep_id: str, merge_id: str) -> int:
        """Merge merge_id into keep_id: reassign all triples, then delete merge_id.
        Returns number of triples reassigned."""
        if keep_id == merge_id:
            return 0
        reassigned = 0
        # Reassign triples where merge_id is subject
        r1 = self.db.execute(
            "UPDATE triples SET subject_id = ? WHERE subject_id = ?",
            (keep_id, merge_id),
        )
        reassigned += r1.rowcount
        # Reassign triples where merge_id is object
        r2 = self.db.execute(
            "UPDATE triples SET object_id = ? WHERE object_id = ?",
            (keep_id, merge_id),
        )
        reassigned += r2.rowcount
        # Delete the merged entity
        self.db.execute("DELETE FROM entities WHERE id = ?", (merge_id,))
        self.db.commit()
        return reassigned

    def list_entities(self, limit: int = 100) -> list[dict]:
        """List all entities with their triple counts."""
        rows = self.db.execute(
            """SELECT e.id, e.name, e.entity_type, e.confidence, e.created_at,
                      e.access_count,
                      (SELECT COUNT(*) FROM triples t
                       WHERE (t.subject_id = e.id OR t.object_id = e.id)
                         AND t.valid_until IS NULL) as active_triples
               FROM entities e
               ORDER BY active_triples DESC, e.confidence DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "type": r[2], "confidence": round(r[3], 3),
             "created_at": r[4], "access_count": r[5], "active_triples": r[6]}
            for r in rows
        ]

    def list_triples_enriched(self, limit: int = 100) -> list[dict]:
        """List all active triples with resolved entity names."""
        rows = self.db.execute(
            """SELECT t.id, e1.name as subj, t.predicate,
                      COALESCE(e2.name, t.object_value) as obj,
                      t.confidence, t.created_at, t.relation_type
               FROM triples t
               JOIN entities e1 ON t.subject_id = e1.id
               LEFT JOIN entities e2 ON t.object_id = e2.id
               WHERE t.valid_until IS NULL
               ORDER BY t.confidence DESC, t.created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [
            {"id": r[0], "subject": r[1], "predicate": r[2], "object": r[3],
             "confidence": round(r[4], 3), "created_at": r[5], "type": r[6]}
            for r in rows
        ]

    def find_orphan_entities(self) -> list[dict]:
        """Find entities with no triples at all (active or historical)."""
        rows = self.db.execute(
            """SELECT e.id, e.name, e.entity_type, e.confidence
               FROM entities e
               WHERE NOT EXISTS (
                   SELECT 1 FROM triples t
                   WHERE t.subject_id = e.id OR t.object_id = e.id
               )
               ORDER BY e.name"""
        ).fetchall()
        return [{"id": r[0], "name": r[1], "type": r[2], "confidence": r[3]} for r in rows]

    def find_duplicate_entities(self) -> list[list[dict]]:
        """Find entities with similar names (case-insensitive, prefix overlap)."""
        rows = self.db.execute(
            """SELECT e1.id, e1.name, e2.id, e2.name
               FROM entities e1
               JOIN entities e2 ON e1.id < e2.id
               WHERE LOWER(e1.name) = LOWER(e2.name)
                  OR LOWER(e1.name) LIKE LOWER(e2.name) || '%'
                  OR LOWER(e2.name) LIKE LOWER(e1.name) || '%'
               ORDER BY e1.name"""
        ).fetchall()
        groups = []
        for r in rows:
            groups.append([
                {"id": r[0], "name": r[1]},
                {"id": r[2], "name": r[3]},
            ])
        return groups

    def purge_expired(self) -> int:
        """Hard-delete all soft-deleted triples (valid_until IS NOT NULL)."""
        result = self.db.execute(
            "DELETE FROM triples WHERE valid_until IS NOT NULL"
        )
        self.db.commit()
        return result.rowcount

    def triple_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    def history(self, subject_id: str, predicate: str) -> list[dict]:
        """Get the full history of a (subject, predicate) pair, including superseded values."""
        return self.get_triples(
            subject_id=subject_id, predicate=predicate, active_only=False
        )
