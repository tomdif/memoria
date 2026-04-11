"""
Memoria: the orchestrator.

Ties together all layers:
  L1 — Raw conversation store (ground truth)
  L2 — Knowledge graph (primary retrieval target)
  L3 — Emergent cluster summaries

Provides the unified API that the MCP server and CLI consume.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .schema import SCHEMA_SQL
from .storage import ConversationStore
from .graph import KnowledgeGraph
from .extractor import extract_from_text, ingest_extraction
from .retriever import Retriever, RetrievalMode, RetrievalResponse, format_results
from .consolidator import Consolidator
from .compressor import compress, budget_report, CompressedMemory
from .embeddings import Embedder


class Memoria:
    """Main interface to the memoria system."""

    def __init__(
        self,
        db_path: str | Path = "~/.memoria/memoria.db",
        model_name: str = "all-MiniLM-L6-v2",
        llm_call=None,
    ):
        """Initialize memoria.

        Args:
            db_path: Path to SQLite database.
            model_name: Sentence transformer model for embeddings.
            llm_call: Optional callable(prompt: str) -> str for extraction/summarization.
                      If None, uses heuristic extraction (no LLM dependency).
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(str(self.db_path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA_SQL)

        self.store = ConversationStore(self.db)
        self.kg = KnowledgeGraph(self.db)
        self.embedder = Embedder(model_name)
        self.llm_call = llm_call
        self.retriever = Retriever(self.kg, self.embedder, llm_call)
        self.consolidator = Consolidator(self.kg, self.embedder, llm_call)

    def remember(
        self,
        text: str,
        role: str = "user",
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Ingest a conversation turn: store raw, extract knowledge, update graph.

        This is the primary write path. Returns extraction stats.
        """
        # L1: Store raw
        conv_id = self.store.append(text, role=role, session_id=session_id, metadata=metadata)

        # Extract entities and relations
        if self.llm_call:
            extraction = extract_from_text(text, self.llm_call)
        else:
            extraction = self._heuristic_extract(text)

        # L2: Write to knowledge graph
        stats = ingest_extraction(
            self.kg, extraction,
            source_ref=conv_id,
            embedder=self.embedder,
        )
        stats["conversation_id"] = conv_id
        return stats

    def recall(self, query: str, top_k: int = 20, as_of: float | None = None,
               mode: str = "balanced") -> RetrievalResponse:
        """Retrieve memories relevant to a query. Three-pass retrieval.

        Args:
            mode: "speed" (16 q/s, 94.0% R@5), "balanced" (12 q/s, 94.6% R@5),
                  or "quality" (5 q/s, 95.0% R@5).
        """
        retrieval_mode = RetrievalMode(mode)
        return self.retriever.retrieve(query, top_k=top_k, as_of=as_of, mode=retrieval_mode)

    def recall_formatted(self, query: str, top_k: int = 20, mode: str = "balanced") -> str:
        """Retrieve and format results as readable text."""
        response = self.recall(query, top_k=top_k, mode=mode)
        return format_results(response, self.kg)

    def consolidate(self, **kwargs) -> dict:
        """Run memory consolidation. Call periodically or before sessions."""
        return self.consolidator.consolidate(**kwargs)

    def compress(self, budget_tokens: int = 200) -> CompressedMemory:
        """Compress full memory state into a token budget.

        Uses spectral ranking: project graph onto top eigenvectors of the
        Laplacian, keep triples with highest spectral importance. The number
        of eigenvectors scales with the budget — more tokens, more detail.

        Tiers:  L0 (≤50) identity only
                L1 (≤200) key facts, compact
                L2 (≤2000) cluster summaries + facts
                L3 (>2000) full detail
        """
        return compress(self.kg, budget_tokens=budget_tokens, llm_call=self.llm_call)

    def budget_report(self) -> dict:
        """Show what you'd get at each compression tier."""
        return budget_report(self.kg)

    def history(self, entity_name: str, predicate: str | None = None) -> list[dict]:
        """Get the full history of an entity, including superseded values."""
        entities = self.kg.find_entities(entity_name)
        if not entities:
            return []

        results = []
        for entity in entities:
            if predicate:
                triples = self.kg.history(entity.id, predicate)
            else:
                triples = self.kg.get_triples(subject_id=entity.id, active_only=False)
            for t in triples:
                t["entity_name"] = entity.name
            results.extend(triples)

        return sorted(results, key=lambda t: t.get("created_at", 0))

    def graph_stats(self) -> dict:
        """Get full system statistics including spectral analysis."""
        base = self.consolidator.stats()
        base["conversations"] = self.store.count()
        base["db_path"] = str(self.db_path)
        return base

    def entity_context(self, entity_name: str) -> dict:
        """Get everything known about an entity — current facts, history, neighbors."""
        entities = self.kg.find_entities(entity_name)
        if not entities:
            return {"error": f"No entity found matching '{entity_name}'"}

        entity = entities[0]
        current = self.kg.get_triples(subject_id=entity.id, active_only=True)
        incoming = self.kg.get_triples(object_id=entity.id, active_only=True)
        historical = self.kg.get_triples(subject_id=entity.id, active_only=False)

        # Resolve entity names for readability
        def enrich(triples):
            enriched = []
            for t in triples:
                t = dict(t)
                if t.get("object_id"):
                    obj = self.kg.get_entity(t["object_id"])
                    t["object_name"] = obj.name if obj else None
                subj = self.kg.get_entity(t["subject_id"])
                t["subject_name"] = subj.name if subj else None
                enriched.append(t)
            return enriched

        return {
            "entity": {"id": entity.id, "name": entity.name, "type": entity.type if hasattr(entity, 'type') else entity.entity_type},
            "current_facts": enrich(current),
            "incoming_relations": enrich(incoming),
            "history": enrich(historical),
            "superseded_count": sum(1 for t in historical if t.get("valid_until")),
        }

    def _heuristic_extract(self, text: str) -> dict:
        """Fallback extraction without LLM — regex-based entity detection."""
        import re
        entities = []
        facts = []

        # Capitalized multi-word phrases (proper nouns)
        for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
            name = match.group(1)
            if len(name) > 2:
                entities.append({"name": name, "type": "concept"})

        # Quoted strings as entities
        for match in re.finditer(r'"([^"]+)"', text):
            entities.append({"name": match.group(1), "type": "concept"})

        # "X uses Y", "X is Y" patterns
        for match in re.finditer(
            r"\b(\w+(?:\s+\w+)?)\s+(uses?|is|was|has|prefers?|decided|switched to|moved to)\s+(\w+(?:\s+\w+)?)\b",
            text, re.IGNORECASE,
        ):
            facts.append({
                "subject": match.group(1),
                "predicate": match.group(2).lower().replace(" ", "_"),
                "object": match.group(3),
                "confidence": 0.5,
            })

        # Deduplicate entities
        seen = set()
        unique_entities = []
        for e in entities:
            if e["name"].lower() not in seen:
                seen.add(e["name"].lower())
                unique_entities.append(e)

        return {"entities": unique_entities, "facts": facts, "causal": [], "decisions": []}

    def cleanup(
        self,
        action: str,
        entity_name: str | None = None,
        entity_id: str | None = None,
        triple_id: str | None = None,
        merge_into: str | None = None,
    ) -> dict:
        """Clean up the knowledge graph.

        Actions:
          - "list_entities": Show all entities with triple counts
          - "list_triples": Show all active triples with resolved names
          - "delete_entity": Delete entity by name or ID and all its triples
          - "delete_triple": Delete a specific triple by ID
          - "merge_entities": Merge entity_name into merge_into (keeps merge_into)
          - "find_duplicates": Find entities with similar/identical names
          - "find_orphans": Find entities with no active triples
          - "purge_orphans": Delete all orphan entities
          - "purge_expired": Hard-delete all soft-deleted (expired) triples
        """
        if action == "list_entities":
            entities = self.kg.list_entities()
            return {"entities": entities, "count": len(entities)}

        elif action == "list_triples":
            triples = self.kg.list_triples_enriched()
            return {"triples": triples, "count": len(triples)}

        elif action == "delete_entity":
            if entity_id:
                deleted = self.kg.delete_entity(entity_id)
                return {"deleted_triples": deleted, "entity_id": entity_id}
            elif entity_name:
                entities = self.kg.find_entities(entity_name)
                if not entities:
                    return {"error": f"No entity found matching '{entity_name}'"}
                # Exact match only
                exact = [e for e in entities if e.name.lower() == entity_name.lower()]
                if not exact:
                    return {
                        "error": "No exact match. Candidates:",
                        "candidates": [{"id": e.id, "name": e.name} for e in entities],
                    }
                total = 0
                deleted_ids = []
                for e in exact:
                    total += self.kg.delete_entity(e.id)
                    deleted_ids.append(e.id)
                return {"deleted_triples": total, "deleted_entities": deleted_ids}
            else:
                return {"error": "Provide entity_name or entity_id"}

        elif action == "delete_triple":
            if not triple_id:
                return {"error": "Provide triple_id"}
            found = self.kg.delete_triple(triple_id)
            return {"deleted": found, "triple_id": triple_id}

        elif action == "merge_entities":
            if not entity_name or not merge_into:
                return {"error": "Provide entity_name (to remove) and merge_into (to keep)"}
            src = self.kg.find_entities(entity_name)
            dst = self.kg.find_entities(merge_into)
            if not src:
                return {"error": f"Source entity '{entity_name}' not found"}
            if not dst:
                return {"error": f"Target entity '{merge_into}' not found"}
            src_exact = [e for e in src if e.name.lower() == entity_name.lower()]
            dst_exact = [e for e in dst if e.name.lower() == merge_into.lower()]
            if not src_exact:
                return {"error": f"No exact match for '{entity_name}'",
                        "candidates": [{"id": e.id, "name": e.name} for e in src]}
            if not dst_exact:
                return {"error": f"No exact match for '{merge_into}'",
                        "candidates": [{"id": e.id, "name": e.name} for e in dst]}
            reassigned = self.kg.merge_entities(dst_exact[0].id, src_exact[0].id)
            return {"merged": entity_name, "into": merge_into, "triples_reassigned": reassigned}

        elif action == "find_duplicates":
            dupes = self.kg.find_duplicate_entities()
            return {"duplicates": dupes, "count": len(dupes)}

        elif action == "find_orphans":
            orphans = self.kg.find_orphan_entities()
            return {"orphans": orphans, "count": len(orphans)}

        elif action == "purge_orphans":
            orphans = self.kg.find_orphan_entities()
            for o in orphans:
                self.kg.delete_entity(o["id"])
            return {"purged": len(orphans)}

        elif action == "purge_expired":
            count = self.kg.purge_expired()
            return {"purged_triples": count}

        else:
            return {"error": f"Unknown action: {action}. "
                    "Valid: list_entities, list_triples, delete_entity, delete_triple, "
                    "merge_entities, find_duplicates, find_orphans, purge_orphans, purge_expired"}

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
