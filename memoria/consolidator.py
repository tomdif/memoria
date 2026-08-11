"""
Active consolidation — the part everyone skips.

Human memory works because of forgetting. Every existing memory system
is append-only with no consolidation, so retrieval degrades linearly
with time. This module fixes that.

Operations:
  - Merge: Facts appearing 3+ times get promoted to high confidence
  - Decay: Apply spectral consolidation operator to update confidences
  - Prune: Soft-delete triples below confidence threshold
  - Cluster: Recompute emergent clusters from graph structure
"""

from __future__ import annotations

import json
import time

import numpy as np

from .graph import KnowledgeGraph
from .spectral import (
    build_adjacency,
    consolidation_operator,
    find_clusters,
    spectral_gap,
)
from .embeddings import Embedder, serialize_embedding


class Consolidator:
    """Spectral-informed memory consolidation."""

    def __init__(self, kg: KnowledgeGraph, embedder: Embedder | None = None,
                 llm_call=None):
        self.kg = kg
        self.embedder = embedder
        self.llm_call = llm_call

    def consolidate(
        self,
        prune_threshold: float = 0.05,
        half_life_days: float = 30.0,
    ) -> dict:
        """Run full consolidation cycle. Returns stats."""
        stats = {
            "merged": 0,
            "decayed": 0,
            "pruned": 0,
            "clusters_found": 0,
            "self_refs_removed": 0,
            "duplicates_merged": 0,
            "orphans_removed": 0,
            "expired_purged": 0,
            "timestamp": time.time(),
        }

        # Phase 0: Promote repeated facts before deduplication removes copies.
        stats["merged"] = self._merge_repeated()

        # Phase 1: Self-cleanse structural garbage while preserving history.
        cleanse = self._self_cleanse()
        stats["self_refs_removed"] = cleanse["self_refs"]
        stats["duplicates_merged"] = cleanse["duplicates"]
        stats["orphans_removed"] = cleanse["orphans"]
        stats["expired_purged"] = cleanse["expired"]

        # Phase 2: Spectral decay
        stats["decayed"] = self._spectral_decay(half_life_days)

        # Phase 3: Prune low-confidence
        stats["pruned"] = self._prune(prune_threshold)

        # Phase 4: Recompute clusters
        stats["clusters_found"] = self._recluster()

        # Log consolidation
        self.kg.db.execute(
            "INSERT INTO consolidation_log (timestamp, action, details) VALUES (?, ?, ?)",
            (time.time(), "full_consolidation", json.dumps(stats)),
        )
        self.kg.db.commit()

        return stats

    def _self_cleanse(self) -> dict:
        """Automatic self-cleansing: fix structural issues in the graph.

        1. Merge exact-name duplicate entities
        2. Remove self-referencing triples (entity -> relation -> same entity)
        3. Deduplicate identical active triples, keeping highest confidence
        4. Remove true orphan entities (no active or historical triples)

        Expired triples are intentionally retained for temporal history. They
        can still be removed explicitly with cleanup(action="purge_expired").
        """
        result = {"self_refs": 0, "duplicates": 0, "orphans": 0, "expired": 0}

        # 1. Merge exact-name duplicate entities first; this can create new
        # self-references or duplicate triples that the following phases fix.
        name_dupes = self.kg.db.execute(
            """SELECT LOWER(name) as lname, GROUP_CONCAT(id), COUNT(*) as cnt
               FROM entities
               GROUP BY lname
               HAVING cnt > 1"""
        ).fetchall()
        for row in name_dupes:
            ids = row[1].split(",")
            keep_id = ids[0]
            for merge_id in ids[1:]:
                self.kg.merge_entities(keep_id, merge_id)
                result["duplicates"] += 1

        # 2. Remove self-referencing triples.
        r = self.kg.db.execute(
            """DELETE FROM triples
               WHERE object_id IS NOT NULL
                 AND subject_id = object_id
                 AND valid_until IS NULL"""
        )
        result["self_refs"] = r.rowcount

        # 3. Deduplicate identical active triples. Resolve the survivor with an
        # explicit ORDER BY; GROUP_CONCAT itself provides no ordering guarantee.
        dupes = self.kg.db.execute(
            """SELECT subject_id, predicate, object_id, object_value
               FROM triples
               WHERE valid_until IS NULL
               GROUP BY subject_id, predicate,
                        COALESCE(object_id, ''), COALESCE(object_value, '')
               HAVING COUNT(*) > 1"""
        ).fetchall()
        for subject_id, predicate, object_id, object_value in dupes:
            rows = self.kg.db.execute(
                """SELECT id FROM triples
                   WHERE subject_id = ? AND predicate = ?
                     AND object_id IS ? AND object_value IS ?
                     AND valid_until IS NULL
                   ORDER BY confidence DESC, created_at DESC, id""",
                (subject_id, predicate, object_id, object_value),
            ).fetchall()
            for (dup_id,) in rows[1:]:
                self.kg.db.execute("DELETE FROM triples WHERE id = ?", (dup_id,))
                result["duplicates"] += 1

        # 4. Remove entities that have no active or historical triples.
        orphans = self.kg.find_orphan_entities()
        for o in orphans:
            self.kg.delete_entity(o["id"])
        result["orphans"] = len(orphans)

        self.kg.db.commit()
        return result

    def _merge_repeated(self) -> int:
        """Promote facts that appear in 3+ sources to high confidence."""
        # Find exact (subject, predicate, object) facts with multiple sources.
        rows = self.kg.db.execute(
            """SELECT subject_id, predicate, object_id, object_value,
                      COUNT(DISTINCT source_ref) as src_count,
                      GROUP_CONCAT(id) as triple_ids
               FROM triples
               WHERE relation_type = 'fact' AND valid_until IS NULL
               GROUP BY subject_id, predicate,
                        COALESCE(object_id, ''), COALESCE(object_value, '')
               HAVING src_count >= 3"""
        ).fetchall()

        merged = 0
        for r in rows:
            triple_ids = r[5].split(",")
            # Boost confidence of all matching triples
            self.kg.db.execute(
                f"""UPDATE triples SET confidence = MIN(confidence * 1.5, 1.0)
                    WHERE id IN ({','.join('?' * len(triple_ids))})""",
                triple_ids,
            )
            merged += 1
        self.kg.db.commit()
        return merged

    def _spectral_decay(self, half_life_days: float) -> int:
        """Apply the spectral consolidation operator to update all confidences."""
        triples, entity_index = self.kg.all_triples_for_spectral()
        if len(entity_index) < 2:
            return 0

        A = build_adjacency(triples, entity_index)
        n = len(entity_index)
        index_to_id = {v: k for k, v in entity_index.items()}

        # Gather per-entity stats
        confidences = np.ones(n)
        access_counts = np.zeros(n)
        ages = np.zeros(n)
        now = time.time()

        for eid, idx in entity_index.items():
            entity = self.kg.get_entity(eid)
            if entity:
                confidences[idx] = entity.confidence
                access_counts[idx] = entity.access_count
                ages[idx] = (now - entity.created_at) / 86400  # days

        # Apply consolidation operator
        new_conf = consolidation_operator(
            A, confidences, access_counts, ages,
            half_life_days=half_life_days,
        )

        # Write back to entities
        updated = 0
        for idx in range(n):
            eid = index_to_id[idx]
            old = confidences[idx]
            new = float(new_conf[idx])
            if abs(old - new) > 0.01:
                self.kg.db.execute(
                    "UPDATE entities SET confidence = ? WHERE id = ?",
                    (new, eid),
                )
                # Also update triples involving this entity
                self.kg.db.execute(
                    """UPDATE triples SET confidence = confidence * ?
                       WHERE (subject_id = ? OR object_id = ?)
                         AND valid_until IS NULL""",
                    (new / max(old, 0.01), eid, eid),
                )
                updated += 1

        self.kg.db.commit()
        return updated

    def _prune(self, threshold: float) -> int:
        """Soft-delete triples below confidence threshold."""
        now = time.time()
        result = self.kg.db.execute(
            """UPDATE triples SET valid_until = ?
               WHERE confidence < ? AND valid_until IS NULL
                 AND relation_type = 'fact'""",
            (now, threshold),
        )
        self.kg.db.commit()
        return result.rowcount

    def _recluster(self) -> int:
        """Recompute emergent clusters from graph structure."""
        triples, entity_index = self.kg.all_triples_for_spectral()
        if len(entity_index) < 3:
            return 0

        A = build_adjacency(triples, entity_index)
        clusters = find_clusters(A)

        index_to_id = {v: k for k, v in entity_index.items()}

        # Clear old clusters
        self.kg.db.execute("DELETE FROM clusters")

        now = time.time()
        for i, cluster_indices in enumerate(clusters):
            entity_ids = [index_to_id[idx] for idx in cluster_indices if idx in index_to_id]
            if not entity_ids:
                continue

            cluster_id = f"cluster_{i}"

            # Compute centroid if embedder available
            centroid_bytes = None
            if self.embedder:
                embeddings = []
                for eid in entity_ids:
                    entity = self.kg.get_entity(eid)
                    if entity and entity.embedding is not None:
                        embeddings.append(entity.embedding)
                if embeddings:
                    centroid = np.mean(embeddings, axis=0)
                    centroid_bytes = serialize_embedding(centroid)

            # Generate summary if LLM available
            summary = None
            if self.llm_call:
                summary = self._summarize_cluster(entity_ids)

            self.kg.db.execute(
                """INSERT INTO clusters (id, entity_ids, summary, centroid, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cluster_id, json.dumps(entity_ids), summary, centroid_bytes, now, now),
            )

        self.kg.db.commit()
        return len(clusters)

    def _summarize_cluster(self, entity_ids: list[str]) -> str | None:
        """Generate a natural language summary of a cluster."""
        if not self.llm_call:
            return None

        # Gather entity names and key facts
        facts = []
        for eid in entity_ids[:20]:  # cap for prompt size
            entity = self.kg.get_entity(eid)
            if not entity:
                continue
            triples = self.kg.get_triples(subject_id=eid, active_only=True)
            for t in triples[:5]:
                obj = t.get("object_value", "")
                if not obj and t.get("object_id"):
                    obj_entity = self.kg.get_entity(t["object_id"])
                    obj = obj_entity.name if obj_entity else ""
                facts.append(f"{entity.name} {t['predicate']} {obj}")

        if not facts:
            return None

        prompt = (
            "Summarize these related facts into 2-3 sentences. "
            "Be specific and factual, no filler.\n\n"
            + "\n".join(facts[:30])
        )
        try:
            return self.llm_call(prompt)
        except Exception:
            return None

    def stats(self) -> dict:
        """Get consolidation statistics."""
        entity_count = self.kg.entity_count()
        triple_count = self.kg.triple_count()
        active_triples = self.kg.db.execute(
            "SELECT COUNT(*) FROM triples WHERE valid_until IS NULL"
        ).fetchone()[0]
        clusters = self.kg.db.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]

        # Spectral stats
        triples, entity_index = self.kg.all_triples_for_spectral()
        gap_val = 0.0
        eigenvalues = []
        if len(entity_index) >= 3:
            A = build_adjacency(triples, entity_index)
            gap_val, eigenvalues = spectral_gap(A)

        return {
            "entities": entity_count,
            "triples_total": triple_count,
            "triples_active": active_triples,
            "triples_superseded": triple_count - active_triples,
            "clusters": clusters,
            "spectral_gap": gap_val,
            "eigenvalues": eigenvalues.tolist() if len(eigenvalues) > 0 else [],
            "screening_radius": int(np.ceil(2.0 / max(gap_val, 0.01))),
        }
