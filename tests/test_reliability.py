"""Failure injection, concurrency, temporal correctness, and offline recall."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from memoria.core import Memoria
from memoria.database import transaction
from memoria.embeddings import Embedder
from memoria.extractor import ingest_extraction
from memoria.retriever import EmbeddingCache, Retriever


@pytest.fixture
def memory(tmp_path):
    with Memoria(tmp_path / 'memory.db', enable_embeddings=False) as value:
        yield value


def test_failed_write_rolls_back_raw_graph_and_supersession(memory, monkeypatch):
    entity = memory.kg.add_entity('App')
    original, _ = memory.kg.add_triple(entity, 'version', object_value='v1', valid_from=0)
    extraction = {'facts': [
        {'subject': 'App', 'predicate': 'version', 'object': 'v2'},
        {'subject': 'New entity', 'predicate': 'status', 'object': 'stable'},
    ]}
    memory.llm_call = lambda _: json.dumps(extraction)
    add = memory.kg.add_triple
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise sqlite3.OperationalError('injected disk failure')
        return add(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(memory.kg, 'add_triple', fail_second)
        with pytest.raises(sqlite3.OperationalError, match='injected'):
            memory.remember('App version changed')
    assert memory.store.count() == 0
    assert memory.store.search_fts('changed') == []
    assert memory.kg.entity_count() == 1
    assert [row['id'] for row in memory.kg.get_triples()] == [original]
    assert not memory.db.in_transaction
    assert memory.remember('App version changed')['triples_added'] == 2


@pytest.mark.parametrize('output', [[], None, {'facts': 'bad'}, {'entities': [None]},
    {'facts': [{'subject': 'App', 'predicate': 'uses', 'object': {}}]},
    {'facts': [{'subject': 'App', 'predicate': 'uses', 'object': 'DB', 'confidence': float('nan')}]},
    {'facts': [{'subject': 'App', 'predicate': 'uses', 'object': 'DB', 'replace_existing': 'false'}]},
])
def test_invalid_provider_output_never_partially_writes(memory, output):
    memory.llm_call = lambda _: json.dumps(output)
    with pytest.raises(ValueError):
        memory.remember('input')
    assert memory.store.count() == memory.kg.entity_count() == memory.kg.triple_count() == 0


def test_provider_failure_leaves_no_raw_memory(memory):
    def fail(_):
        raise RuntimeError('provider unavailable')
    memory.llm_call = fail
    with pytest.raises(RuntimeError, match='unavailable'):
        memory.remember('remember this')
    assert memory.store.count() == 0


def test_nested_operations_do_not_commit_the_callers_transaction(memory):
    memory.db.execute('BEGIN')
    memory.store.append('outer transaction')
    entity = memory.kg.add_entity('Outer')
    memory.kg.add_triple(entity, 'status', object_value='pending')
    memory.db.rollback()
    assert memory.store.count() == memory.kg.entity_count() == memory.kg.triple_count() == 0


def test_nested_failure_preserves_outer_work(memory):
    with transaction(memory.db):
        memory.store.append('keep this')
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(memory.db):
                doomed = memory.kg.add_entity('Doomed')
                memory.kg.add_triple(doomed, 'uses', object_id='missing')
        memory.store.append('also keep this')
    assert memory.store.count() == 2
    assert memory.kg.entity_count() == 0


def test_concurrent_writers_deduplicate_entities(tmp_path):
    path = tmp_path / 'concurrent.db'
    with Memoria(path, enable_embeddings=False):
        pass

    def write(index):
        with Memoria(path, enable_embeddings=False) as memory:
            entity = memory.kg.add_entity('Shared')
            memory.kg.add_triple(entity, 'uses', object_value=f'tool-{index}')
            return entity

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(write, range(18)))
    assert len(set(ids)) == 1
    with Memoria(path, enable_embeddings=False) as memory:
        assert memory.kg.triple_count() == 18
        assert memory.check_storage()['ok']


def test_entity_resolution_does_not_take_prefix_matches(memory):
    longer = memory.kg.add_entity('PostgresQL service')
    result = ingest_extraction(memory.kg, {
        'facts': [{'subject': 'Postgres', 'predicate': 'version', 'object': '17'}],
    })
    assert result['entities_added'] == 1
    fact = memory.kg.get_triples()[0]
    assert fact['subject_id'] != longer
    assert memory.kg.get_entity(fact['subject_id']).name == 'Postgres'


def test_entity_lookup_escapes_wildcards_and_prioritizes_exact_match(memory):
    exact = memory.kg.add_entity('API')
    memory.kg.add_entity('API Gateway')
    literal = memory.kg.add_entity('API_%')
    assert memory.kg.find_entities('API')[0].id == exact
    assert [entity.id for entity in memory.kg.find_entities('API_%')] == [literal]


def test_temporal_replacement_handles_object_types_empty_values_and_epoch(memory):
    app = memory.kg.add_entity('App')
    database = memory.kg.add_entity('SQLite')
    first, _ = memory.kg.add_triple(app, 'database', object_id=database, valid_from=0)
    second, conflicts = memory.kg.add_triple(app, 'database', object_value='Postgres', valid_from=10)
    assert conflicts[0]['id'] == first
    third, _ = memory.kg.add_triple(app, 'database', object_value='', valid_from=20)
    assert [row['id'] for row in memory.kg.get_triples(as_of=0)] == [first]
    assert [row['id'] for row in memory.kg.get_triples(as_of=10)] == [second]
    assert [row['id'] for row in memory.kg.get_triples(as_of=20)] == [third]
    with pytest.raises(ValueError, match='predate'):
        memory.kg.add_triple(app, 'database', object_value='obsolete', valid_from=15)
    assert memory.kg.triple_count() == 3


def test_unstructured_text_is_fully_searchable_without_models(memory, monkeypatch):
    def no_models(_):
        raise AssertionError('model loading is forbidden')
    monkeypatch.setattr(Embedder, 'model', property(no_models))
    monkeypatch.setattr(Retriever, 'reranker', property(no_models))
    content = 'lowercase unstructured note ' + 'filler ' * 400 + 'zephyrmarker'
    stored = memory.remember(content)
    response = memory.recall_json('zephyrmarker')
    assert stored['fallback_note'] is True
    assert response['count'] == 1
    assert response['results'][0]['fact']['object_value'] == content
    assert response['results'][0]['provenance']['id'] == stored['conversation_id']
    assert 'cross_encoder' not in response['passes_used']
    assert memory.recall('nomatchingterm').results == []
    assert memory.recall('zephyrmarker', top_k=0).results == []


@pytest.mark.parametrize('value', ['', '  ', None, [], 42])
def test_invalid_memory_input_is_rejected(memory, value):
    with pytest.raises(ValueError):
        memory.remember(value)
    assert memory.store.count() == 0


@pytest.mark.parametrize('value', [-1, True, 1.5, '10', 1001])
def test_invalid_result_limit_is_rejected(memory, value):
    with pytest.raises(ValueError):
        memory.recall('query', top_k=value)


def test_natural_language_fts_handles_punctuation_and_unicode(memory):
    source = memory.store.append('SQLite supports café users and C++ tools')
    for query in ['SQLite?', '"café', 'C++', 'SQLite: OR café']:
        assert memory.store.search_fts(query)[0]['id'] == source
    assert memory.store.search_fts('?!') == []
    assert memory.store.search_fts('SQLite', limit=0) == []
    assert memory.store.search_fts('"SQLite" AND "tools"', raw_query=True)[0]['id'] == source


class BatchEmbedder:
    dimension = 2

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.array([[len(text), 1] for text in texts], dtype=np.float32)

    def embed_single(self, text):
        return self.embed([text])[0]


def test_embeddings_are_batched_before_acquiring_writer_lock(memory):
    class CheckingEmbedder(BatchEmbedder):
        def embed(self, texts):
            # A separate writer must be able to write during slow model work.
            with sqlite3.connect(memory.db_path, timeout=0.1) as other:
                other.execute('BEGIN IMMEDIATE')
                other.rollback()
            return super().embed(texts)

    embedder = CheckingEmbedder()
    memory.embedder = embedder
    memory.llm_call = lambda _: json.dumps({
        'entities': [{'name': 'App'}, {'name': 'Database'}, {'name': 'App'}],
        'facts': [{'subject': 'App', 'predicate': 'uses', 'object': 'Database'}],
    })
    assert memory.remember('App uses Database')['entities_added'] == 2
    assert embedder.calls == [['App', 'Database']]


def test_embedding_cache_is_bounded_deduplicates_and_preserves_batch_order():
    embedder = BatchEmbedder()
    cache = EmbeddingCache(embedder, max_entries=2)
    result = cache.get_batch(['a', 'bb', 'a', 'ccc'])
    assert embedder.calls == [['a', 'bb', 'ccc']]
    np.testing.assert_array_equal(result[:, 0], [1, 2, 1, 3])
    assert cache.size == 2
    assert all(vector.base is None for vector in cache._cache.values())
    cache.get('bb')
    cache.get('dddd')
    cache.get('bb')
    assert embedder.calls[-1] == ['dddd']
    assert cache.size == 2


def test_consolidation_failure_rolls_back_every_phase(memory, monkeypatch):
    entity = memory.kg.add_entity('Repeated')
    for _ in range(3):
        memory.kg.add_triple(entity, 'status', object_value='stable')
    before = list(memory.db.iterdump())
    def fail():
        raise RuntimeError('cluster failure')
    monkeypatch.setattr(memory.consolidator, '_recluster', fail)
    with pytest.raises(RuntimeError, match='cluster failure'):
        memory.consolidate()
    assert list(memory.db.iterdump()) == before


def test_new_databases_are_private(memory):
    assert memory.db_path.stat().st_mode & 0o777 == 0o600


def test_degenerate_spectral_clustering_is_a_complete_partition():
    from memoria.spectral import _spectral_kmeans, find_clusters
    from scipy import sparse
    clusters = _spectral_kmeans(np.ones((7, 3)), 4)
    assert sorted(index for cluster in clusters for index in cluster) == list(range(7))
    clusters = find_clusters(sparse.csr_matrix([[0., 1.], [1., 0.]]), n_clusters=2)
    assert sorted(index for cluster in clusters for index in cluster) == [0, 1]


def test_recall_document_loading_uses_bounded_sql_queries(memory):
    # No extracted entities in the lowercase query: isolate document rendering
    # and source loading from graph traversal and model inference.
    entity = memory.kg.add_entity('Record')
    for index in range(30):
        source = memory.store.append(f'searchable source {index}')
        memory.kg.add_triple(entity, 'records', object_value=f'searchable {index}',
                             source_ref=source, replace_existing=False)
    statements = []
    memory.db.set_trace_callback(statements.append)
    assert memory.recall('searchable', top_k=1).results
    reads = [sql for sql in statements if sql.lstrip().upper().startswith('SELECT')]
    assert len(reads) <= 5


def test_empty_neural_database_does_not_load_models(tmp_path, monkeypatch):
    def forbidden(_):
        raise AssertionError('empty recall loaded a model')
    monkeypatch.setattr(Embedder, 'model', property(forbidden))
    monkeypatch.setattr(Retriever, 'reranker', property(forbidden))
    with Memoria(tmp_path / 'empty.db', enable_embeddings=True) as memory:
        assert memory.recall('query').results == []


def test_consolidation_preserves_distinct_relation_types(memory):
    cause = memory.kg.add_entity('Cause')
    effect = memory.kg.add_entity('Effect')
    for kind in ('fact', 'causal'):
        memory.kg.add_triple(cause, 'caused', object_id=effect, relation_type=kind)
    memory.consolidator._self_cleanse()
    assert {fact['relation_type'] for fact in memory.kg.get_triples()} == {'fact', 'causal'}
