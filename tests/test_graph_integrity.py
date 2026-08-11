"""Regression tests for graph data integrity and temporal behavior."""

import sqlite3

import numpy as np
import pytest

from memoria.consolidator import Consolidator
from memoria.embeddings import deserialize_embedding, serialize_embedding
from memoria.extractor import ingest_extraction
from memoria.graph import KnowledgeGraph
from memoria.schema import SCHEMA_SQL


@pytest.fixture
def kg():
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(SCHEMA_SQL)
    graph = KnowledgeGraph(db)
    yield graph
    db.close()


def test_embedding_round_trip_uses_float32(kg):
    original = np.linspace(0, 1, 384, dtype=np.float64)
    entity_id = kg.add_entity("Postgres", embedding=original)

    stored = kg.db.execute(
        "SELECT embedding FROM entities WHERE id = ?", (entity_id,)
    ).fetchone()[0]
    restored = kg.get_entity(entity_id).embedding

    assert len(stored) == 384 * np.dtype(np.float32).itemsize
    assert restored.dtype == np.float32
    assert restored.shape == (384,)
    np.testing.assert_allclose(restored, original, rtol=1e-6)


def test_legacy_float64_embedding_can_be_decoded():
    original = np.arange(8, dtype=np.float64)
    restored = deserialize_embedding(original.tobytes(), expected_dimension=8)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, original)


def test_entity_dedup_is_case_insensitive_and_backfills_embedding(kg):
    first = kg.add_entity("Postgres")
    second = kg.add_entity("postgres", entity_type="tool", embedding=np.ones(4))

    assert first == second
    entity = kg.get_entity(first)
    assert entity.entity_type == "tool"
    np.testing.assert_array_equal(entity.embedding, np.ones(4, dtype=np.float32))


def test_additive_relationships_do_not_supersede_each_other(kg):
    backend = kg.add_entity("Backend")
    tools = [kg.add_entity(name) for name in ("Django", "Postgres", "Redis")]

    for tool in tools:
        _, contradictions = kg.add_triple(backend, "uses", object_id=tool)
        assert contradictions == []

    active = kg.get_triples(subject_id=backend, predicate="uses")
    assert {triple["object_id"] for triple in active} == set(tools)


def test_stateful_relationships_retain_temporal_history(kg):
    api = kg.add_entity("API")
    for version in ("v1", "v2", "v3"):
        kg.add_triple(api, "version", object_value=version)

    history = kg.history(api, "version")
    assert len(history) == 3
    assert sum(triple["valid_until"] is None for triple in history) == 1


def test_explicit_cardinality_override_is_respected(kg):
    api = kg.add_entity("API")
    jwt = kg.add_entity("JWT")
    oauth = kg.add_entity("OAuth2")
    kg.add_triple(api, "uses", object_id=jwt, replace_existing=True)
    _, contradictions = kg.add_triple(
        api, "uses", object_id=oauth, replace_existing=True
    )

    assert len(contradictions) == 1
    assert len(kg.get_triples(subject_id=api, predicate="uses")) == 1


def test_malformed_extraction_does_not_resolve_empty_names(kg):
    existing = kg.add_entity("Existing")
    extraction = {
        "entities": [],
        "facts": [{"subject": "", "predicate": "uses", "object": "SQLite"}],
        "causal": [{"cause": "", "effect": "Existing"}],
        "decisions": [],
    }

    stats = ingest_extraction(kg, extraction)

    assert stats["triples_added"] == 0
    assert kg.get_triples(subject_id=existing) == []


def test_consolidation_preserves_history_and_manual_purge_removes_it(kg):
    api = kg.add_entity("API")
    for version in ("v1", "v2", "v3"):
        kg.add_triple(api, "version", object_value=version)

    stats = Consolidator(kg).consolidate()

    assert stats["expired_purged"] == 0
    assert len(kg.history(api, "version")) == 3
    assert kg.purge_expired() == 2
    assert len(kg.history(api, "version")) == 1


def test_consolidation_keeps_highest_confidence_duplicate(kg):
    api = kg.add_entity("API")
    for confidence, source in ((0.2, "a"), (0.9, "b"), (0.5, "c")):
        kg.add_triple(
            api,
            "status",
            object_value="stable",
            confidence=confidence,
            source_ref=source,
        )

    stats = Consolidator(kg).consolidate()
    active = kg.get_triples(subject_id=api, predicate="status")

    assert stats["merged"] == 1
    assert len(active) == 1
    assert active[0]["confidence"] == pytest.approx(1.0)
