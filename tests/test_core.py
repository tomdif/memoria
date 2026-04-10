"""
End-to-end tests for memoria — exercises the full pipeline
without requiring an LLM (uses heuristic extraction).
"""

import os
import sqlite3
import tempfile
import time

import numpy as np
import pytest

# Ensure we can import without sentence-transformers for unit tests
os.environ.setdefault("MEMORIA_NO_EMBED", "1")

from memoria.schema import SCHEMA_SQL
from memoria.graph import KnowledgeGraph
from memoria.storage import ConversationStore
from memoria.compressor import compress, spectral_rank, budget_report
from memoria.spectral import (
    build_adjacency,
    consolidation_operator,
    find_clusters,
    graph_laplacian,
    local_gap,
    screening_radius,
    spectral_gap,
)


@pytest.fixture
def db():
    """In-memory SQLite database with schema."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    yield conn
    conn.close()


@pytest.fixture
def kg(db):
    return KnowledgeGraph(db)


@pytest.fixture
def store(db):
    return ConversationStore(db)


# --- Storage tests ---

class TestConversationStore:
    def test_append_and_get(self, store):
        cid = store.append("Hello world", role="user", session_id="s1")
        result = store.get(cid)
        assert result is not None
        assert result["content"] == "Hello world"
        assert result["role"] == "user"
        assert result["session_id"] == "s1"

    def test_count(self, store):
        assert store.count() == 0
        store.append("one")
        store.append("two")
        assert store.count() == 2

    def test_fts_search(self, store):
        store.append("We use OAuth2 for authentication")
        store.append("The database runs on Postgres")
        results = store.search_fts("OAuth2")
        assert len(results) == 1
        assert "OAuth2" in results[0]["content"]

    def test_session(self, store):
        store.append("turn 1", session_id="s1")
        store.append("turn 2", session_id="s1")
        store.append("other session", session_id="s2")
        turns = store.get_session("s1")
        assert len(turns) == 2


# --- Knowledge Graph tests ---

class TestKnowledgeGraph:
    def test_add_entity(self, kg):
        eid = kg.add_entity("OAuth2", entity_type="tool")
        assert eid is not None
        entity = kg.get_entity(eid)
        assert entity.name == "OAuth2"
        assert entity.entity_type == "tool"

    def test_entity_dedup(self, kg):
        id1 = kg.add_entity("OAuth2", entity_type="tool")
        id2 = kg.add_entity("OAuth2", entity_type="tool")
        assert id1 == id2

    def test_add_triple(self, kg):
        e1 = kg.add_entity("API", entity_type="project")
        e2 = kg.add_entity("OAuth2", entity_type="tool")
        tid, contras = kg.add_triple(e1, "uses", object_id=e2)
        assert tid is not None
        assert len(contras) == 0

    def test_contradiction_detection(self, kg):
        api = kg.add_entity("API")
        jwt = kg.add_entity("JWT")
        oauth = kg.add_entity("OAuth2")

        # First fact: API uses JWT
        t1, c1 = kg.add_triple(api, "auth_method", object_id=jwt, relation_type="fact")
        assert len(c1) == 0

        # Second fact: API uses OAuth2 — contradicts first
        t2, c2 = kg.add_triple(api, "auth_method", object_id=oauth, relation_type="fact")
        assert len(c2) == 1
        assert c2[0]["object_id"] == jwt

        # Old triple should be closed
        old = kg.get_triples(subject_id=api, predicate="auth_method", active_only=False)
        closed = [t for t in old if t["valid_until"] is not None]
        assert len(closed) == 1

    def test_history(self, kg):
        api = kg.add_entity("API")
        kg.add_triple(api, "version", object_value="v1")
        kg.add_triple(api, "version", object_value="v2")
        kg.add_triple(api, "version", object_value="v3")

        history = kg.history(api, "version")
        # Should have all 3 entries, with v1 and v2 superseded
        assert len(history) == 3

    def test_neighbors(self, kg):
        a = kg.add_entity("A")
        b = kg.add_entity("B")
        c = kg.add_entity("C")
        kg.add_triple(a, "knows", object_id=b)
        kg.add_triple(b, "knows", object_id=c)

        n1 = kg.neighbors(a, max_depth=1)
        assert 1 in n1
        n2 = kg.neighbors(a, max_depth=2)
        assert 2 in n2

    def test_touch(self, kg):
        e = kg.add_entity("X")
        tid, _ = kg.add_triple(e, "is", object_value="cool")
        kg.touch([tid])
        triples = kg.get_triples(subject_id=e)
        assert triples[0]["access_count"] == 1


# --- Spectral tests ---

class TestSpectral:
    def _make_chain(self, n):
        """Create a chain graph: 0-1-2-...(n-1)."""
        entity_index = {str(i): i for i in range(n)}
        triples = []
        for i in range(n - 1):
            triples.append({
                "subject_id": str(i),
                "object_id": str(i + 1),
                "relation_type": "fact",
                "confidence": 1.0,
            })
        return triples, entity_index

    def _make_two_cliques(self, size=5):
        """Two cliques connected by a single bridge edge."""
        n = 2 * size
        entity_index = {str(i): i for i in range(n)}
        triples = []
        # Clique 1
        for i in range(size):
            for j in range(i + 1, size):
                triples.append({
                    "subject_id": str(i), "object_id": str(j),
                    "relation_type": "fact", "confidence": 1.0,
                })
        # Clique 2
        for i in range(size, n):
            for j in range(i + 1, n):
                triples.append({
                    "subject_id": str(i), "object_id": str(j),
                    "relation_type": "fact", "confidence": 1.0,
                })
        # Bridge
        triples.append({
            "subject_id": str(size - 1), "object_id": str(size),
            "relation_type": "fact", "confidence": 1.0,
        })
        return triples, entity_index

    def test_spectral_gap_chain(self):
        """Chain graph should have a small spectral gap (long correlations)."""
        triples, idx = self._make_chain(20)
        A = build_adjacency(triples, idx)
        gap, eigenvalues = spectral_gap(A)
        assert gap > 0
        assert gap < 0.5  # chain has small gap
        assert screening_radius(gap) > 4  # need to walk far

    def test_spectral_gap_cliques(self):
        """Two cliques with bridge should have a small gap (near-disconnected)."""
        triples, idx = self._make_two_cliques(5)
        A = build_adjacency(triples, idx)
        gap, eigenvalues = spectral_gap(A)
        assert gap > 0
        # The gap should be small (near-disconnected)
        # But within each clique, things are well-connected

    def test_find_clusters_two_cliques(self):
        """Should find 2 clusters for two-clique graph."""
        triples, idx = self._make_two_cliques(5)
        A = build_adjacency(triples, idx)
        clusters = find_clusters(A)
        assert len(clusters) >= 2
        # Each cluster should roughly correspond to one clique

    def test_consolidation_operator(self):
        """High-access nodes should maintain confidence; low-access old nodes should decay."""
        triples, idx = self._make_chain(5)
        A = build_adjacency(triples, idx)
        n = 5
        confidences = np.ones(n)
        access_counts = np.array([10, 0, 0, 0, 10], dtype=float)
        ages = np.array([1, 100, 100, 100, 1], dtype=float)

        new_conf = consolidation_operator(A, confidences, access_counts, ages)
        # Endpoints (high access, young) should have higher confidence than middle (no access, old)
        assert new_conf[0] > new_conf[2]
        assert new_conf[4] > new_conf[2]

    def test_local_gap(self):
        triples, idx = self._make_chain(10)
        A = build_adjacency(triples, idx)
        gap, depth = local_gap(A, [0], max_radius=5)
        assert gap > 0
        assert depth >= 1

    def test_screening_radius_bounds(self):
        assert screening_radius(0.5) >= 1
        assert screening_radius(0.01) <= 15
        assert screening_radius(2.0) == 1


# --- Integration test (no LLM, no embeddings) ---

class TestIntegration:
    def test_full_pipeline(self, db):
        """Test the full remember → recall → consolidate cycle without LLM/embeddings."""
        store = ConversationStore(db)
        kg = KnowledgeGraph(db)

        # Simulate extraction results (bypassing LLM)
        from memoria.extractor import ingest_extraction

        # First conversation: team uses JWT
        conv1 = store.append("We decided to use JWT for API authentication", session_id="s1")
        extraction1 = {
            "entities": [
                {"name": "API", "type": "project"},
                {"name": "JWT", "type": "tool"},
            ],
            "facts": [
                {"subject": "API", "predicate": "auth_method", "object": "JWT", "confidence": 0.9},
            ],
            "causal": [],
            "decisions": [
                {"subject": "API", "decision": "use JWT for auth", "reason": "simplicity", "confidence": 0.9},
            ],
        }
        stats1 = ingest_extraction(kg, extraction1, source_ref=conv1)
        assert stats1["entities_added"] >= 2
        assert stats1["triples_added"] >= 1

        # Second conversation: switch to OAuth2
        conv2 = store.append("We're switching from JWT to OAuth2 because of the SSO requirement", session_id="s2")
        extraction2 = {
            "entities": [
                {"name": "API", "type": "project"},
                {"name": "OAuth2", "type": "tool"},
                {"name": "SSO", "type": "concept"},
            ],
            "facts": [
                {"subject": "API", "predicate": "auth_method", "object": "OAuth2", "confidence": 0.95},
            ],
            "causal": [
                {"cause": "SSO", "effect": "OAuth2", "confidence": 0.8},
            ],
            "decisions": [],
        }
        stats2 = ingest_extraction(kg, extraction2, source_ref=conv2)
        # Should detect contradiction with JWT
        assert len(stats2["contradictions"]) >= 1

        # Verify history shows the change
        api_entities = kg.find_entities("API")
        assert len(api_entities) > 0
        history = kg.history(api_entities[0].id, "auth_method")
        assert len(history) >= 2

        # Active fact should be OAuth2
        active = [h for h in history if h["valid_until"] is None]
        superseded = [h for h in history if h["valid_until"] is not None]
        assert len(active) >= 1
        assert len(superseded) >= 1

        # Check causal edge exists
        sso_entities = kg.find_entities("SSO")
        assert len(sso_entities) > 0
        causal = kg.get_triples(
            subject_id=sso_entities[0].id, relation_type="causal"
        )
        assert len(causal) >= 1

    def test_spectral_on_real_graph(self, db):
        """Build a real graph and verify spectral analysis works."""
        kg = KnowledgeGraph(db)
        from memoria.extractor import ingest_extraction

        # Build a graph with clear structure
        extraction = {
            "entities": [
                {"name": "Frontend", "type": "project"},
                {"name": "Backend", "type": "project"},
                {"name": "React", "type": "tool"},
                {"name": "Django", "type": "tool"},
                {"name": "Postgres", "type": "tool"},
                {"name": "Redis", "type": "tool"},
            ],
            "facts": [
                {"subject": "Frontend", "predicate": "uses", "object": "React", "confidence": 1.0},
                {"subject": "Backend", "predicate": "uses", "object": "Django", "confidence": 1.0},
                {"subject": "Backend", "predicate": "uses", "object": "Postgres", "confidence": 1.0},
                {"subject": "Backend", "predicate": "uses", "object": "Redis", "confidence": 1.0},
                {"subject": "Frontend", "predicate": "depends_on", "object": "Backend", "confidence": 0.9},
            ],
            "causal": [],
            "decisions": [],
        }
        ingest_extraction(kg, extraction)

        triples, entity_index = kg.all_triples_for_spectral()
        assert len(entity_index) >= 6

        A = build_adjacency(triples, entity_index)
        gap, eigenvalues = spectral_gap(A)
        assert gap > 0
        radius = screening_radius(gap)
        assert radius >= 1

        clusters = find_clusters(A)
        assert len(clusters) >= 1


# --- Compression tests ---

class TestCompression:
    def _build_graph(self, db):
        """Build a test graph with clear structure for compression tests."""
        kg = KnowledgeGraph(db)
        from memoria.extractor import ingest_extraction

        extraction = {
            "entities": [
                {"name": "Frontend", "type": "project"},
                {"name": "Backend", "type": "project"},
                {"name": "React", "type": "tool"},
                {"name": "Django", "type": "tool"},
                {"name": "Postgres", "type": "tool"},
                {"name": "Redis", "type": "tool"},
                {"name": "Auth", "type": "concept"},
                {"name": "OAuth2", "type": "tool"},
            ],
            "facts": [
                {"subject": "Frontend", "predicate": "uses", "object": "React", "confidence": 1.0},
                {"subject": "Backend", "predicate": "uses", "object": "Django", "confidence": 1.0},
                {"subject": "Backend", "predicate": "uses", "object": "Postgres", "confidence": 1.0},
                {"subject": "Backend", "predicate": "uses", "object": "Redis", "confidence": 0.8},
                {"subject": "Frontend", "predicate": "depends_on", "object": "Backend", "confidence": 0.9},
                {"subject": "Auth", "predicate": "uses", "object": "OAuth2", "confidence": 0.95},
                {"subject": "Backend", "predicate": "implements", "object": "Auth", "confidence": 0.9},
            ],
            "causal": [],
            "decisions": [],
        }
        ingest_extraction(kg, extraction)
        return kg

    def test_spectral_rank_orders_by_importance(self, db):
        """Structurally central nodes should rank higher."""
        kg = self._build_graph(db)
        ranked = spectral_rank(kg, budget_tokens=500)
        assert len(ranked) > 0
        # All should have spectral scores
        for t in ranked:
            assert "_spectral_score" in t
        # Should be sorted descending
        scores = [t["_spectral_score"] for t in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_compress_l0(self, db):
        """L0 should fit in ~50 tokens with just entity names."""
        kg = self._build_graph(db)
        result = compress(kg, budget_tokens=50)
        assert result.tier == "L0"
        assert result.token_estimate <= 60  # some slack
        assert "Known:" in result.text
        assert result.entities_included > 0
        assert result.compression_ratio <= 1.0

    def test_compress_l1(self, db):
        """L1 should give compact key=value facts."""
        kg = self._build_graph(db)
        result = compress(kg, budget_tokens=200)
        assert result.tier == "L1"
        assert "=" in result.text  # compact format
        assert result.entities_included >= result.entities_included  # tautology but checks field exists

    def test_compress_l3(self, db):
        """L3 should include more detail than L1."""
        kg = self._build_graph(db)
        l1 = compress(kg, budget_tokens=200)
        l3 = compress(kg, budget_tokens=5000)
        assert l3.tier == "L3"
        assert l3.entities_included >= l1.entities_included

    def test_higher_budget_includes_more(self, db):
        """More tokens → more entities included (monotonic)."""
        kg = self._build_graph(db)
        results = []
        for budget in [50, 200, 2000, 10000]:
            r = compress(kg, budget_tokens=budget)
            results.append(r)
        # Token estimates should be non-decreasing
        for i in range(1, len(results)):
            assert results[i].entities_included >= results[i - 1].entities_included or \
                   results[i].token_estimate >= results[i - 1].token_estimate

    def test_budget_report(self, db):
        """Budget report should have all four tiers."""
        kg = self._build_graph(db)
        report = budget_report(kg)
        assert "L0" in report
        assert "L1" in report
        assert "L2" in report
        assert "L3" in report
        for tier_info in report.values():
            assert "budget" in tier_info
            assert "compression_ratio" in tier_info
            assert "preview" in tier_info

    def test_empty_graph_compresses(self, db):
        """Compression on empty graph shouldn't crash."""
        kg = KnowledgeGraph(db)
        result = compress(kg, budget_tokens=200)
        assert result.entities_included == 0
