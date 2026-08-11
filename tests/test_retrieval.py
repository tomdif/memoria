"""Tests for the shared product and benchmark hybrid retrieval stack."""

import re
import sqlite3

import numpy as np
import pytest

from memoria.graph import KnowledgeGraph
from memoria.retriever import (
    Retriever,
    RetrievalMode,
    _descending_competition_ranks,
    _query_facets,
)
from memoria.schema import SCHEMA_SQL
from memoria.storage import ConversationStore


class FakeEmbedder:
    dimension = 3

    def embed_single(self, text):
        text = text.lower()
        if "postgres" in text or "database" in text:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "redis" in text or "cache" in text:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.stack([self.embed_single(text) for text in texts])


class RecordingReranker:
    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        scores = []
        for query, document in pairs:
            query_terms = set(re.findall(r"\w+", query.lower()))
            document_terms = set(re.findall(r"\w+", document.lower()))
            scores.append(float(len(query_terms & document_terms)))
        return np.asarray(scores, dtype=float)


@pytest.fixture
def retriever():
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA_SQL)
    kg = KnowledgeGraph(db)
    store = ConversationStore(db)
    embedder = FakeEmbedder()

    postgres = kg.add_entity("Postgres", entity_type="tool", embedding=embedder.embed_single("Postgres"))
    redis = kg.add_entity("Redis", entity_type="tool", embedding=embedder.embed_single("Redis"))
    backend = kg.add_entity("Backend", entity_type="project", embedding=embedder.embed_single("Backend"))
    postgres_source = store.append("The Backend database is Postgres")
    redis_source = store.append("Redis provides the Backend cache")
    kg.add_triple(backend, "uses", object_id=postgres, source_ref=postgres_source)
    kg.add_triple(backend, "uses", object_id=redis, source_ref=redis_source)

    value = Retriever(kg, embedder)
    value._reranker = RecordingReranker()
    yield value
    db.close()


def test_product_recall_uses_full_hybrid_pipeline(retriever):
    response = retriever.retrieve(
        "Which database does the backend use?",
        top_k=2,
        mode=RetrievalMode.BALANCED,
    )

    assert response.results
    assert response.mode == "balanced"
    assert {"vector", "bm25", "cross_encoder", "temporal"} <= set(
        response.passes_used
    )
    assert "vector" in response.results[0].source
    assert "cross_encoder" in response.results[0].source
    assert len(retriever._reranker.calls) == 2  # dual-pass mode


def test_modes_change_real_product_reranking(retriever):
    retriever.retrieve("database", mode=RetrievalMode.SPEED)
    speed_calls = len(retriever._reranker.calls)
    retriever._reranker.calls.clear()
    retriever.retrieve("database", mode=RetrievalMode.QUALITY)
    quality_calls = len(retriever._reranker.calls)

    assert speed_calls == 1
    assert quality_calls == 2


def test_float32_embeddings_drive_vector_retrieval(retriever):
    response = retriever.retrieve("database", top_k=2, mode=RetrievalMode.SPEED)

    assert response.results
    assert any("vector" in result.source for result in response.results)


def test_session_retrieval_uses_shared_hybrid_ranker(retriever, monkeypatch):
    called = {}

    def fake_hybrid_rank(**kwargs):
        called.update(kwargs)
        return [("s2", 1.0), ("s1", 0.5)]

    monkeypatch.setattr(retriever, "_hybrid_rank", fake_hybrid_rank)
    result = retriever.retrieve_sessions(
        "database",
        ["Redis cache", "Postgres database"],
        ["s1", "s2"],
        mode=RetrievalMode.QUALITY,
    )

    assert result[0][0] == "s2"
    assert called["mode"] is RetrievalMode.QUALITY
    assert called["document_ids"] == ["s1", "s2"]


def test_grouped_retrieval_returns_unique_parent_sessions(retriever):
    result = retriever.retrieve_grouped_sessions(
        "postgres database",
        [
            "A long opening about weather",
            "The late evidence says postgres database",
            "Redis cache configuration",
            "An unrelated project update",
        ],
        ["session_1", "session_1", "session_2", "session_3"],
        top_k=3,
    )

    assert result[0][0] == "session_1"
    assert len({group_id for group_id, _ in result}) == len(result)


def test_grouped_reranker_receives_complete_short_windows(retriever):
    late_marker = "late-marker"
    long_window = "x" * 500 + " " + late_marker
    retriever.retrieve_grouped_sessions(
        late_marker,
        [long_window, "other text"],
        ["session_1", "session_2"],
        top_k=2,
    )

    reranked_documents = [
        document
        for call in retriever._reranker.calls
        for _, document in call
    ]
    assert any(late_marker in document for document in reranked_documents)


def test_grouped_retrieval_validates_alignment(retriever):
    with pytest.raises(ValueError, match="equal length"):
        retriever.retrieve_grouped_sessions("query", ["one"], [])


def test_query_facets_are_limited_to_explicit_multi_entity_comparisons():
    comparison = "How did Ada and Bruno each unwind after work?"
    assert _query_facets(comparison) == [
        comparison,
        "Ada unwind after work",
        "Bruno unwind after work",
    ]

    ordinary = "When did Ada finish the migration?"
    assert _query_facets(ordinary) == [ordinary]


def test_grouped_comparison_rewards_complementary_parent_evidence(retriever):
    result = retriever.retrieve_grouped_sessions(
        "How did Ada and Bruno each unwind after work?",
        [
            "Ada unwinds after work by swimming",
            "Bruno unwinds after work by reading",
            "A quarterly finance review",
            "Ada unwinds after work by making pottery",
            "Bruno unwinds after work by tending a garden",
            "A database migration status meeting",
            "A weather forecast for next week",
            "A grocery list and meal plan",
            "A software release checklist",
        ],
        [
            "ada_only",
            "bruno_only",
            "noise_1",
            "complete",
            "complete",
            "noise_2",
            "noise_3",
            "noise_4",
            "noise_5",
        ],
        top_k=5,
    )

    assert result[0][0] == "complete"


def test_rank_fusion_does_not_break_ties_by_corpus_position():
    assert _descending_competition_ranks(np.array([0.5, 0.5, 0.1])).tolist() == [1, 1, 3]


def test_grouped_comparison_caps_entity_coverage_with_query_facets(retriever):
    query = "What do Ada, Bruno, Cora, Dinesh, and Elena each prefer?"
    result = retriever.retrieve_grouped_sessions(
        query,
        [
            "Ada prefers tea",
            "Bruno prefers coffee",
            "Cora prefers water",
            "Dinesh prefers juice",
            "Elena prefers cocoa",
        ],
        ["ada", "bruno", "cora", "dinesh", "elena"],
        top_k=5,
    )

    assert len(result) == 5
