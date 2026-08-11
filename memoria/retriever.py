"""
Hybrid retrieval engine.

Pass 1 — Graph walk: Extract entities from query, look up in KG,
         walk 1/γ hops along causal/depends edges.
Pass 2 — Hybrid candidate search: scoped vectors plus BM25 lexical matching.
Pass 3 — Cross-encoder rerank fused with temporal decay and confidence.

The screening radius from the spectral gap gives a provable completeness
guarantee — if you search within the radius, you haven't missed anything relevant.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .graph import KnowledgeGraph
from .spectral import (
    build_adjacency,
    local_gap,
    screening_radius as compute_screening_radius,
)
from .embeddings import Embedder, deserialize_embedding, top_k_similar


class RetrievalMode(Enum):
    """Retrieval speed/quality tradeoff.

    SPEED uses a top-20 single-pass rerank. BALANCED uses a top-15 dual-pass
    rerank. QUALITY uses a top-20 dual-pass rerank. Actual throughput depends
    on corpus length and hardware; see the dated benchmark report.
    """
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


# Rerank config per mode: (top_k_rerank, dual_pass)
_RERANK_CONFIG = {
    RetrievalMode.SPEED:    (20, False),
    RetrievalMode.BALANCED: (15, True),
    RetrievalMode.QUALITY:  (20, True),
}

# Grouped conversational retrieval reranks short windows rather than truncated
# session prefixes.  The larger pools preserve enough distinct parent sessions
# after multiple windows collapse to the same result.
_GROUP_RERANK_CONFIG = {
    RetrievalMode.SPEED: 20,
    RetrievalMode.BALANCED: 30,
    RetrievalMode.QUALITY: 50,
}
_RRF_CONSTANT = 60.0
_MAX_QUERY_FACETS = 4

# These are grammatical cues, not benchmark categories.  Faceting is limited
# to explicit comparisons so ordinary single-intent queries retain the exact
# retrieval path they used before.
_COMPARISON_CUES = frozenset({
    "both", "common", "compare", "difference", "different", "each",
    "either", "same", "similar", "respectively",
})
_QUESTION_WORDS = frozenset({
    "how", "what", "when", "where", "which", "who", "whom", "whose", "why",
})
_FACET_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "both",
    "by", "common", "compare", "did", "difference", "different", "do",
    "does", "each", "either", "for", "from", "had", "has", "have", "how",
    "in", "is", "it", "of", "on", "or", "respectively", "same", "similar",
    "that", "the", "their", "them", "they", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "with",
})


@dataclass
class RetrievalResult:
    """A single retrieved memory with provenance."""
    triple: dict
    score: float
    source: str  # contributing candidate sources, e.g. "graph+vector+bm25"
    depth: int = 0  # graph distance from query entities
    entity_name: str | None = None


@dataclass
class RetrievalResponse:
    """Full retrieval response with metadata."""
    results: list[RetrievalResult]
    query_entities: list[str]
    screening_depth: int
    local_spectral_gap: float
    passes_used: list[str]
    mode: str = "balanced"


class EmbeddingCache:
    """Cache session embeddings by content hash for fast repeated lookups."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._cache: dict[str, np.ndarray] = {}

    def get(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self._cache:
            self._cache[key] = self.embedder.embed_single(text)
        return self._cache[key]

    def get_batch(self, texts: list[str]) -> np.ndarray:
        result = np.empty((len(texts), self.embedder.dimension), dtype=np.float32)
        to_embed_idx = []
        to_embed_texts = []
        for i, text in enumerate(texts):
            key = hashlib.md5(text.encode()).hexdigest()
            if key in self._cache:
                result[i] = self._cache[key]
            else:
                to_embed_idx.append(i)
                to_embed_texts.append(text)
        if to_embed_texts:
            new_embs = self.embedder.embed(to_embed_texts)
            for j, i in enumerate(to_embed_idx):
                result[i] = new_embs[j]
                key = hashlib.md5(texts[i].encode()).hexdigest()
                self._cache[key] = new_embs[j]
        return result

    @property
    def size(self) -> int:
        return len(self._cache)


def _tokenize(t: str) -> list[str]:
    return re.findall(r"\w+", t.lower())


def _descending_competition_ranks(scores: np.ndarray) -> np.ndarray:
    """Return descending 1-based ranks while assigning ties the same rank."""
    values = np.nan_to_num(
        np.asarray(scores, dtype=float).reshape(-1), nan=-np.inf
    )
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    previous = None
    current_rank = 1
    for position, index in enumerate(order, start=1):
        value = float(values[int(index)])
        if previous is not None and value < previous:
            current_rank = position
        ranks[int(index)] = current_rank
        previous = value
    return ranks


def _comparison_entities(query: str) -> list[str]:
    """Return named entities only when a query explicitly compares them."""
    tokens = _tokenize(query)
    if not _COMPARISON_CUES.intersection(tokens):
        return []

    entities: list[str] = []
    seen_entities: set[str] = set()
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*)*", query
    ):
        parts = match.group(0).split()
        while parts and parts[0].lower() in _QUESTION_WORDS:
            parts.pop(0)
        if not parts:
            continue
        entity = " ".join(parts)
        key = entity.casefold()
        if key not in seen_entities:
            seen_entities.add(key)
            entities.append(entity)

    return entities if len(entities) >= 2 else []


def _query_facets(query: str) -> list[str]:
    """Decompose explicit multi-entity comparisons into neutral search probes.

    The original query is always first.  For a comparison such as ``How do Ada
    and Bruno each relax after work?``, the additional probes retain the
    question's content terms while conditioning once on each named entity.
    This lets a parent session earn credit for complementary evidence even when
    no single short window mentions every entity.
    """
    entities = _comparison_entities(query)
    if not entities:
        return [query]

    tokens = _tokenize(query)
    entity_tokens = {
        token
        for entity in entities
        for token in _tokenize(entity)
    }
    focus_terms = [
        token for token in tokens
        if token not in _FACET_STOPWORDS and token not in entity_tokens
    ]
    focus = " ".join(dict.fromkeys(focus_terms))

    facets = [query]
    for entity in entities[:_MAX_QUERY_FACETS - 1]:
        facet = f"{entity} {focus}".strip()
        if facet.casefold() not in {value.casefold() for value in facets}:
            facets.append(facet)
    return facets


class Retriever:
    """Three-pass retrieval engine with spectral-informed depth."""

    def __init__(self, kg: KnowledgeGraph, embedder: Embedder | None = None,
                 llm_call=None):
        self.kg = kg
        self.embedder = embedder
        self.llm_call = llm_call
        self._emb_cache = EmbeddingCache(embedder) if embedder else None
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        return self._reranker

    def retrieve(self, query: str, top_k: int = 20,
                 as_of: float | None = None,
                 mode: RetrievalMode = RetrievalMode.BALANCED) -> RetrievalResponse:
        """Retrieve graph facts with the same hybrid stack used by benchmarks."""

        # Extract entity names from the query
        query_entities = self._extract_query_entities(query)

        # Resolve entities to IDs
        entity_ids = []
        entity_names = {}
        for name in query_entities:
            found = self.kg.find_entities(name)
            if found:
                entity_ids.append(found[0].id)
                entity_names[found[0].id] = found[0].name

        # --- Pass 1: Graph walk ---
        pass1_results = []
        gap = 1.0
        depth = 2  # default

        if entity_ids:
            # Compute local spectral gap for adaptive walk depth
            triples, entity_index = self.kg.all_triples_for_spectral()
            if len(entity_index) >= 3:
                A = build_adjacency(triples, entity_index)
                center_indices = [entity_index[eid] for eid in entity_ids if eid in entity_index]
                if center_indices:
                    gap, depth = local_gap(A, center_indices)

            # Walk the graph to the computed depth
            for eid in entity_ids:
                neighbor_triples = self.kg.neighbors(
                    eid,
                    max_depth=depth,
                    active_only=(as_of is None),
                    as_of=as_of,
                )
                for d, trip_list in neighbor_triples.items():
                    for t in trip_list:
                        # Score: confidence × (1 / depth) so closer = higher
                        score = t["confidence"] * (1.0 / d)
                        pass1_results.append(RetrievalResult(
                            triple=t, score=score, source="graph",
                            depth=d, entity_name=entity_names.get(eid),
                        ))

        # --- Pass 2: Scoped vector search with adaptive expansion ---
        pass2_results = []
        if self.embedder is not None:
            query_emb = self.embedder.embed_single(query)

            # Get entity embeddings for scoping
            all_entities = self.kg.db.execute(
                "SELECT id, name, embedding FROM entities WHERE embedding IS NOT NULL"
            ).fetchall()

            if all_entities:
                emb_ids: list[str] = []
                emb_names: list[str] = []
                emb_matrix: list[np.ndarray] = []
                expected_dim = self.embedder.dimension
                for eid, ename, emb_bytes in all_entities:
                    embedding = deserialize_embedding(emb_bytes, expected_dim)
                    if embedding is not None:
                        emb_ids.append(eid)
                        emb_names.append(ename)
                        emb_matrix.append(embedding)

                if emb_matrix:
                    corpus = np.stack(emb_matrix)
                    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
                    corpus = corpus / np.where(norms > 0, norms, 1.0)

                    qnorm = np.linalg.norm(query_emb)
                    query_emb_norm = query_emb / qnorm if qnorm > 0 else query_emb
                    similar = top_k_similar(
                        query_emb_norm,
                        corpus,
                        k=min(max(top_k * 2, 20), len(corpus)),
                    )

                    expanded_similar = []
                    if len(similar) >= 5:
                        scores = [s for _, s in similar]
                        gap_1_2 = scores[0] - scores[1]
                        gap_2_5 = scores[1] - scores[min(4, len(scores) - 1)]
                        if gap_1_2 < gap_2_5 * 2.0:
                            top_entity = self.kg.get_entity(emb_ids[similar[0][0]])
                            if top_entity:
                                expanded_q = query + " " + top_entity.name
                                exp_emb = self.embedder.embed_single(expanded_q)
                                exp_norm = np.linalg.norm(exp_emb)
                                exp_emb = exp_emb / exp_norm if exp_norm > 0 else exp_emb
                                expanded_similar = top_k_similar(
                                    exp_emb, corpus, k=min(top_k, len(corpus))
                                )

                    entity_scores: dict[int, float] = {
                        idx: float(score) for idx, score in similar
                    }
                    for idx, score in expanded_similar:
                        entity_scores[idx] = max(
                            entity_scores.get(idx, 0.0), float(score) * 0.5
                        )

                    for idx, sim_score in sorted(
                        entity_scores.items(), key=lambda item: item[1], reverse=True
                    )[:max(top_k, 20)]:
                        eid = emb_ids[idx]
                        entity_triples = self.kg.get_triples(
                            subject_id=eid,
                            active_only=(as_of is None),
                            as_of=as_of,
                        )
                        entity_triples += self.kg.get_triples(
                            object_id=eid,
                            active_only=(as_of is None),
                            as_of=as_of,
                        )
                        for triple in entity_triples:
                            pass2_results.append(RetrievalResult(
                                triple=triple,
                                score=sim_score * triple["confidence"],
                                source="vector",
                                entity_name=emb_names[idx],
                            ))

        # --- Pass 3: BM25 + cross-encoder + temporal fusion ---
        all_triples = self.kg.get_triples(
            active_only=(as_of is None),
            as_of=as_of,
        )
        triple_by_id = {triple["id"]: triple for triple in all_triples}
        dense_scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        depths: dict[str, int] = {}
        names: dict[str, str | None] = {}
        for result in pass1_results + pass2_results:
            triple_id = result.triple["id"]
            dense_scores[triple_id] = max(
                dense_scores.get(triple_id, 0.0), result.score
            )
            sources.setdefault(triple_id, set()).add(result.source)
            if result.depth:
                depths[triple_id] = min(depths.get(triple_id, result.depth), result.depth)
            names.setdefault(triple_id, result.entity_name)

        triple_ids = list(triple_by_id)
        triple_docs = [self._triple_text(triple_by_id[tid]) for tid in triple_ids]
        source_docs = self._source_documents([triple_by_id[tid] for tid in triple_ids])
        ranked = self._hybrid_rank(
            query=query,
            document_ids=triple_ids,
            primary_docs=triple_docs,
            dense_scores=dense_scores,
            mode=mode,
            all_docs=source_docs,
        )

        now = as_of or time.time()
        half_life = 30 * 86400  # 30 days in seconds
        final_results = []
        for triple_id, hybrid_score in ranked:
            triple = triple_by_id[triple_id]
            age = max(0.0, now - triple["created_at"])
            recency = np.exp(-0.693 * age / half_life)  # exp(-ln2 * age / half_life)
            access_boost = np.log1p(triple.get("access_count", 0)) * 0.1
            temporal = 0.6 + 0.3 * recency + 0.1 * (1 + access_boost)
            score = hybrid_score * temporal * triple["confidence"]
            source_parts = sources.get(triple_id, set()) | {"bm25", "cross_encoder"}
            final_results.append(RetrievalResult(
                triple=triple,
                score=float(score),
                source="+".join(sorted(source_parts)),
                depth=depths.get(triple_id, 0),
                entity_name=names.get(triple_id),
            ))

        final_results.sort(key=lambda result: result.score, reverse=True)
        top_results = final_results[:max(top_k, 0)]

        # Touch accessed triples (updates access count)
        if top_results:
            self.kg.touch([r.triple["id"] for r in top_results])

        passes = []
        if pass1_results:
            passes.append("graph")
        if pass2_results:
            passes.append("vector")
        if all_triples:
            passes.extend(["bm25", "cross_encoder", "temporal"])

        return RetrievalResponse(
            results=top_results,
            query_entities=query_entities,
            screening_depth=depth,
            local_spectral_gap=gap,
            passes_used=passes,
            mode=mode.value,
        )

    def _triple_text(self, triple: dict) -> str:
        """Render a triple as a compact document for lexical/neural ranking."""
        subject = self.kg.get_entity(triple["subject_id"])
        subject_name = subject.name if subject else triple["subject_id"]
        object_text = triple.get("object_value") or ""
        if not object_text and triple.get("object_id"):
            obj = self.kg.get_entity(triple["object_id"])
            object_text = obj.name if obj else triple["object_id"]
        predicate = triple["predicate"].replace("_", " ")
        return f"{subject_name} {predicate} {object_text}".strip()

    def _source_documents(self, triples: list[dict]) -> list[str]:
        """Resolve provenance text for dual-pass reranking."""
        source_ids = {triple.get("source_ref") for triple in triples}
        source_ids.discard(None)
        documents: dict[str, str] = {}
        for source_id in source_ids:
            row = self.kg.db.execute(
                "SELECT content FROM conversations WHERE id = ?", (source_id,)
            ).fetchone()
            if row:
                documents[source_id] = row[0]
        return [documents.get(triple.get("source_ref"), "") for triple in triples]

    def _hybrid_rank(
        self,
        query: str,
        document_ids: list[str],
        primary_docs: list[str],
        dense_scores: dict[str, float],
        mode: RetrievalMode,
        all_docs: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Fuse dense and BM25 candidates, then cross-encoder rerank.

        Both graph recall and benchmark session retrieval call this method so
        their speed/balanced/quality modes use the same ranking semantics.
        """
        if not document_ids:
            return []
        if len(document_ids) != len(primary_docs):
            raise ValueError("document_ids and primary_docs must have equal length")
        if all_docs is not None and len(all_docs) != len(document_ids):
            raise ValueError("all_docs must align with document_ids")

        from rank_bm25 import BM25Okapi

        tokenized_docs = [_tokenize(document) for document in primary_docs]
        bm25 = BM25Okapi(tokenized_docs)
        query_tokens = _tokenize(query)
        bm25_raw = np.asarray(bm25.get_scores(query_tokens), dtype=float)
        bm25_max = float(bm25_raw.max()) if bm25_raw.size else 0.0
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else np.zeros_like(bm25_raw)

        dense_max = max((max(score, 0.0) for score in dense_scores.values()), default=0.0)
        dense_scale = dense_max if dense_max > 0 else 1.0
        stage1 = {
            document_id: (
                max(dense_scores.get(document_id, 0.0), 0.0) / dense_scale * 0.70
                + float(bm25_norm[index]) * 0.30
            )
            for index, document_id in enumerate(document_ids)
        }

        rerank_k, dual_pass = _RERANK_CONFIG[mode]
        candidates = sorted(stage1.items(), key=lambda item: item[1], reverse=True)[
            :min(rerank_k, len(document_ids))
        ]
        candidate_ids = [document_id for document_id, _ in candidates]
        id_to_index = {document_id: index for index, document_id in enumerate(document_ids)}

        if dual_pass and all_docs is not None:
            primary_pairs = [
                (query, primary_docs[id_to_index[document_id]][:512])
                for document_id in candidate_ids
            ]
            source_pairs = [
                (
                    query,
                    all_docs[id_to_index[document_id]][:512]
                    or primary_docs[id_to_index[document_id]][:512],
                )
                for document_id in candidate_ids
            ]
            cross_scores = np.maximum(
                np.asarray(self.reranker.predict(primary_pairs), dtype=float),
                np.asarray(self.reranker.predict(source_pairs), dtype=float),
            )
        else:
            pairs = []
            for document_id in candidate_ids:
                index = id_to_index[document_id]
                document = primary_docs[index][:384]
                if all_docs is not None and all_docs[index]:
                    document += "\n" + all_docs[index][:128]
                pairs.append((query, document))
            cross_scores = np.asarray(self.reranker.predict(pairs), dtype=float)

        cross_scores = cross_scores.reshape(-1)
        cross_min = float(cross_scores.min()) if cross_scores.size else 0.0
        cross_max = float(cross_scores.max()) if cross_scores.size else 0.0
        cross_range = cross_max - cross_min
        stage1_max = max(stage1.values(), default=0.0)
        stage1_scale = stage1_max if stage1_max > 0 else 1.0

        final: dict[str, float] = {}
        for index, document_id in enumerate(candidate_ids):
            cross_norm = (
                (float(cross_scores[index]) - cross_min) / cross_range
                if cross_range > 0
                else 0.5
            )
            stage1_norm = stage1[document_id] / stage1_scale
            final[document_id] = cross_norm * 0.40 + stage1_norm * 0.60

        minimum_reranked = min(final.values(), default=0.0)
        for document_id, score in stage1.items():
            if document_id not in final:
                final[document_id] = (
                    score / stage1_scale * minimum_reranked * 0.99
                )

        return sorted(final.items(), key=lambda item: item[1], reverse=True)

    def retrieve_sessions(
        self,
        query: str,
        session_docs: list[str],
        session_ids: list[str],
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.BALANCED,
        all_docs: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve ranked session IDs from a corpus of session documents.

        This is the benchmark-facing API: given a query and a set of session
        documents (user-turn text), return ranked (session_id, score) pairs
        using the bi-encoder + BM25 + cross-encoder pipeline.

        Args:
            query: The search query.
            session_docs: List of user-turn text per session.
            session_ids: Corresponding session IDs.
            top_k: Number of results to return.
            mode: Speed/quality tradeoff (SPEED, BALANCED, QUALITY).
            all_docs: Optional full-text (user+assistant) per session for dual-pass.

        Returns:
            List of (session_id, score) tuples, highest first.
        """
        if not session_docs:
            return []
        if len(session_docs) != len(session_ids):
            raise ValueError("session_docs and session_ids must have equal length")
        if self.embedder is None or self._emb_cache is None:
            raise ValueError("retrieve_sessions requires an embedder")

        n = len(session_docs)

        # Stage 1a: bi-encoder with cached embeddings
        corpus_embs = self._emb_cache.get_batch(session_docs)
        query_emb = self.embedder.embed_single(query)

        similar = top_k_similar(query_emb, corpus_embs, k=min(60, n))
        vec_scores = {session_ids[idx]: float(s) for idx, s in similar}

        # Stage 1b: single-probe expansion (conditional)
        scores = [s for _, s in similar]
        if len(similar) >= 5:
            gap_1_2 = scores[0] - scores[1]
            gap_2_5 = scores[1] - scores[min(4, len(scores) - 1)]
            dense = sum(1 for s in scores if s >= scores[0] * 0.9)
            if gap_1_2 < gap_2_5 * 2.0 or dense >= 3:
                expanded = query + " " + session_docs[similar[0][0]][:200]
                exp_emb = self.embedder.embed_single(expanded)
                for idx, s in top_k_similar(exp_emb, corpus_embs, k=min(20, n)):
                    sid = session_ids[idx]
                    vec_scores[sid] = max(vec_scores.get(sid, 0.0), float(s) * 0.5)

        ranked = self._hybrid_rank(
            query=query,
            document_ids=session_ids,
            primary_docs=session_docs,
            dense_scores=vec_scores,
            mode=mode,
            all_docs=all_docs,
        )
        return ranked[:top_k]

    def retrieve_grouped_sessions(
        self,
        query: str,
        chunk_docs: list[str],
        group_ids: list[str],
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.BALANCED,
    ) -> list[tuple[str, float]]:
        """Retrieve unique parent sessions from overlapping text windows.

        Dense and lexical ranks are fused at the parent-session level using
        reciprocal-rank fusion. Explicit multi-entity comparisons also use
        entity-scoped query facets, so a parent can receive credit for distinct
        evidence windows. A cross-encoder then contributes a bounded rank
        signal over the strongest short windows. This keeps exact lexical
        matches available while avoiding the long-document truncation that
        occurs when an entire session is embedded or reranked at once.
        """
        if not chunk_docs:
            return []
        if len(chunk_docs) != len(group_ids):
            raise ValueError("chunk_docs and group_ids must have equal length")
        if self.embedder is None or self._emb_cache is None:
            raise ValueError("retrieve_grouped_sessions requires an embedder")
        if top_k <= 0:
            return []

        from rank_bm25 import BM25Okapi

        document_embeddings = self._emb_cache.get_batch(chunk_docs)
        query_facets = _query_facets(query)
        comparison_entities = _comparison_entities(query)[:_MAX_QUERY_FACETS - 1]
        query_embeddings = self.embedder.embed(query_facets)
        dense_scores = np.asarray(
            query_embeddings @ document_embeddings.T, dtype=float
        )

        bm25 = BM25Okapi([_tokenize(document) for document in chunk_docs])
        bm25_scores = np.stack([
            np.asarray(bm25.get_scores(_tokenize(facet)), dtype=float)
            for facet in query_facets
        ])

        dense_orders = np.argsort(-dense_scores, axis=1, kind="stable")
        bm25_orders = np.argsort(-bm25_scores, axis=1, kind="stable")
        dense_ranks = np.stack([
            _descending_competition_ranks(scores) for scores in dense_scores
        ])
        bm25_ranks = np.stack([
            _descending_competition_ranks(scores) for scores in bm25_scores
        ])

        facet_count = len(query_facets)
        missing_rank = len(chunk_docs) + 1
        group_dense_rank: dict[str, np.ndarray] = {}
        group_bm25_rank: dict[str, np.ndarray] = {}
        group_documents: dict[str, list[str]] = {}
        for index, group_id in enumerate(group_ids):
            dense_group = group_dense_rank.setdefault(
                group_id, np.full(facet_count, missing_rank, dtype=int)
            )
            bm25_group = group_bm25_rank.setdefault(
                group_id, np.full(facet_count, missing_rank, dtype=int)
            )
            dense_group[:] = np.minimum(dense_group, dense_ranks[:, index])
            bm25_group[:] = np.minimum(bm25_group, bm25_ranks[:, index])
            group_documents.setdefault(group_id, []).append(chunk_docs[index])

        group_entity_coverage: dict[str, np.ndarray] = {}
        for group_id, documents in group_documents.items():
            parent_text = "\n".join(documents)
            group_entity_coverage[group_id] = np.asarray([
                bool(re.search(
                    rf"(?<!\w){re.escape(entity)}(?!\w)",
                    parent_text,
                    flags=re.IGNORECASE,
                ))
                for entity in comparison_entities
            ], dtype=bool)

        rerank_count = min(_GROUP_RERANK_CONFIG[mode], len(chunk_docs))
        candidate_limit = min(max(rerank_count * 2, top_k * 6), len(chunk_docs))
        candidate_indices: set[int] = set()
        for facet_index in range(facet_count):
            candidate_indices.update(
                int(index)
                for index in dense_orders[facet_index, :candidate_limit]
            )
            candidate_indices.update(
                int(index)
                for index in bm25_orders[facet_index, :candidate_limit]
            )

        def stage_score(index: int) -> float:
            return float(np.mean(
                1.0 / (_RRF_CONSTANT + dense_ranks[:, index])
                + 1.0 / (_RRF_CONSTANT + bm25_ranks[:, index])
            ))

        def group_stage_score(group_id: str) -> float:
            facet_scores = (
                1.0 / (_RRF_CONSTANT + group_dense_rank[group_id])
                + 1.0 / (_RRF_CONSTANT + group_bm25_rank[group_id])
            )
            if comparison_entities:
                # An entity-conditioned probe cannot count as covered merely
                # because its generic focus words occur.  Coverage is checked
                # across every window belonging to the parent session.
                facet_scores[1:] *= group_entity_coverage[group_id]
            return float(np.mean(facet_scores))

        rerank_indices = sorted(
            candidate_indices,
            key=lambda index: (-stage_score(int(index)), int(index)),
        )[:rerank_count]
        pairs = [(query, chunk_docs[int(index)]) for index in rerank_indices]
        cross_scores = np.asarray(self.reranker.predict(pairs), dtype=float).reshape(-1)
        cross_ranks = _descending_competition_ranks(cross_scores)
        group_cross_rank: dict[str, int] = {}
        for pair_index, cross_rank in enumerate(cross_ranks):
            chunk_index = int(rerank_indices[pair_index])
            group_id = group_ids[chunk_index]
            group_cross_rank[group_id] = min(
                group_cross_rank.get(group_id, rerank_count + 1), int(cross_rank)
            )

        group_scores: dict[str, float] = {}
        for group_id in dict.fromkeys(group_ids):
            score = group_stage_score(group_id)
            cross_rank = group_cross_rank.get(group_id)
            if cross_rank is not None:
                # The reranker is advisory: it can promote a good window but
                # cannot erase a strong full-corpus lexical or dense rank.
                score += 0.5 / (_RRF_CONSTANT + cross_rank)
            group_scores[group_id] = score

        return sorted(
            group_scores.items(), key=lambda item: (-item[1], item[0])
        )[:top_k]

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entity names from a query.

        Uses LLM if available, otherwise falls back to simple heuristics.
        """
        if self.llm_call:
            return self._extract_entities_llm(query)
        return self._extract_entities_heuristic(query)

    def _extract_entities_llm(self, query: str) -> list[str]:
        prompt = (
            "Extract entity names from this query. Return ONLY a JSON list of strings.\n"
            "Include: people, projects, tools, concepts, files, organizations.\n"
            "Exclude: common words, verbs, adjectives.\n\n"
            f"Query: {query}\n\nJSON list:"
        )
        try:
            import json, re
            response = self.llm_call(prompt)
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"^```\w*\n?", "", response)
                response = re.sub(r"\n?```$", "", response)
            return json.loads(response)
        except Exception:
            return self._extract_entities_heuristic(query)

    def _extract_entities_heuristic(self, query: str) -> list[str]:
        """Simple entity extraction: capitalized words and quoted strings."""
        import re
        entities = []

        # Quoted strings
        for match in re.finditer(r'"([^"]+)"', query):
            entities.append(match.group(1))

        # Capitalized words (potential proper nouns), skip sentence starts
        words = query.split()
        for i, word in enumerate(words):
            clean = re.sub(r"[^\w]", "", word)
            if clean and clean[0].isupper() and len(clean) > 1:
                entities.append(clean)

        # Technical terms with special chars (file paths, package names)
        for match in re.finditer(r"[\w./\-]+\.[\w]+", query):
            entities.append(match.group())

        return entities


def format_results(response: RetrievalResponse, kg: KnowledgeGraph) -> str:
    """Format retrieval results as readable text with provenance."""
    lines = []
    lines.append(f"Query entities: {response.query_entities}")
    lines.append(f"Spectral gap: {response.local_spectral_gap:.4f} → search depth: {response.screening_depth}")
    lines.append(f"Passes used: {', '.join(response.passes_used)}")
    lines.append(f"Results: {len(response.results)}")
    lines.append("")

    for i, r in enumerate(response.results):
        t = r.triple
        subj = kg.get_entity(t["subject_id"])
        subj_name = subj.name if subj else t["subject_id"][:8]

        obj_str = t.get("object_value", "")
        if not obj_str and t.get("object_id"):
            obj = kg.get_entity(t["object_id"])
            obj_str = obj.name if obj else t["object_id"][:8]

        lines.append(
            f"  [{i+1}] ({r.source}, score={r.score:.3f}, depth={r.depth}) "
            f"{subj_name} → {t['predicate']} → {obj_str}"
        )
        if t.get("valid_until"):
            lines.append(f"       [SUPERSEDED]")

    return "\n".join(lines)
