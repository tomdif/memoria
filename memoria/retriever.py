"""
Three-pass retrieval engine.

Pass 1 — Graph walk: Extract entities from query, look up in KG,
         walk 1/γ hops along causal/depends edges.
Pass 2 — Scoped vector search: Use entities and cluster IDs from pass 1
         to scope vector search to relevant regions.
Pass 3 — Rerank with temporal decay: Score by relevance × recency × confidence.

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
from .embeddings import Embedder, cosine_similarity, top_k_similar


class RetrievalMode(Enum):
    """Retrieval speed/quality tradeoff.

    SPEED:    top-20 single-pass rerank, ~16 q/s, 94.0% R@5
    BALANCED: top-15 dual-pass rerank,   ~12 q/s, 94.6% R@5
    QUALITY:  top-20 dual-pass rerank,    ~5 q/s, 95.0% R@5
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


@dataclass
class RetrievalResult:
    """A single retrieved memory with provenance."""
    triple: dict
    score: float
    source: str  # "graph", "vector", "fts"
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
        """Execute three-pass retrieval for a query."""

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
                neighbor_triples = self.kg.neighbors(eid, max_depth=depth, active_only=(as_of is None))
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
                emb_ids = []
                emb_names = []
                emb_matrix = []
                for eid, ename, emb_bytes in all_entities:
                    if emb_bytes:
                        emb_ids.append(eid)
                        emb_names.append(ename)
                        emb_matrix.append(np.frombuffer(emb_bytes, dtype=np.float64))

                if emb_matrix:
                    expected_dim = self.embedder.dimension
                    valid = [(i, e) for i, e in enumerate(emb_matrix) if len(e) == expected_dim]
                    if valid:
                        valid_idx, valid_embs = zip(*valid)
                        corpus = np.stack(valid_embs)
                        norms = np.linalg.norm(corpus, axis=1, keepdims=True)
                        norms = np.where(norms > 0, norms, 1.0)
                        corpus = corpus / norms

                        qnorm = np.linalg.norm(query_emb)
                        query_emb_norm = query_emb / qnorm if qnorm > 0 else query_emb

                        similar = top_k_similar(query_emb_norm, corpus, k=min(top_k * 2, len(corpus)))

                        # Adaptive expansion: if top results are clustered,
                        # expand query with top-1 hit to find related entities.
                        # This helps multi-topic queries find all relevant contexts.
                        expanded_similar = []
                        if len(similar) >= 5:
                            scores = [s for _, s in similar]
                            gap_1_2 = scores[0] - scores[1]
                            gap_2_5 = scores[1] - scores[min(4, len(scores) - 1)]
                            if gap_1_2 < gap_2_5 * 2.0:
                                # Top results clustered — expand
                                top_eid = emb_ids[valid_idx[similar[0][0]]]
                                top_entity = self.kg.get_entity(top_eid)
                                if top_entity:
                                    expanded_q = query + " " + top_entity.name
                                    exp_emb = self.embedder.embed_single(expanded_q)
                                    exp_norm = np.linalg.norm(exp_emb)
                                    exp_emb_norm = exp_emb / exp_norm if exp_norm > 0 else exp_emb
                                    expanded_similar = top_k_similar(
                                        exp_emb_norm, corpus, k=min(top_k, len(corpus))
                                    )

                        # Merge initial and expanded results
                        entity_scores = {}
                        for corpus_idx, sim_score in similar:
                            orig_idx = valid_idx[corpus_idx]
                            entity_scores[orig_idx] = float(sim_score)
                        for corpus_idx, sim_score in expanded_similar:
                            orig_idx = valid_idx[corpus_idx]
                            entity_scores[orig_idx] = max(
                                entity_scores.get(orig_idx, 0.0),
                                float(sim_score) * 0.5,
                            )

                        # Fetch triples for similar entities
                        seen_entity_ids = set(entity_ids)
                        ranked_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
                        for orig_idx, sim_score in ranked_entities[:top_k]:
                            eid = emb_ids[orig_idx]
                            if eid in seen_entity_ids:
                                continue
                            seen_entity_ids.add(eid)

                            entity_triples = self.kg.get_triples(subject_id=eid, active_only=True)
                            entity_triples += self.kg.get_triples(object_id=eid, active_only=True)
                            for t in entity_triples:
                                score = sim_score * t["confidence"]
                                pass2_results.append(RetrievalResult(
                                    triple=t, score=score, source="vector",
                                    entity_name=emb_names[orig_idx],
                                ))

        # --- Pass 3: Rerank with temporal decay ---
        all_results = pass1_results + pass2_results

        # Deduplicate by triple ID
        seen = set()
        unique_results = []
        for r in all_results:
            if r.triple["id"] not in seen:
                seen.add(r.triple["id"])
                unique_results.append(r)

        # Temporal reranking
        now = as_of or time.time()
        half_life = 30 * 86400  # 30 days in seconds
        for r in unique_results:
            age = now - r.triple["created_at"]
            recency = np.exp(-0.693 * age / half_life)  # exp(-ln2 * age / half_life)
            access_boost = np.log1p(r.triple.get("access_count", 0)) * 0.1
            r.score = r.score * (0.6 + 0.3 * recency + 0.1 * (1 + access_boost))

        # Sort by score descending
        unique_results.sort(key=lambda r: r.score, reverse=True)
        top_results = unique_results[:top_k]

        # Touch accessed triples (updates access count)
        if top_results:
            self.kg.touch([r.triple["id"] for r in top_results])

        passes = []
        if pass1_results:
            passes.append("graph")
        if pass2_results:
            passes.append("vector")

        return RetrievalResponse(
            results=top_results,
            query_entities=query_entities,
            screening_depth=depth,
            local_spectral_gap=gap,
            passes_used=passes,
            mode=mode.value,
        )

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

        n = len(session_docs)
        rerank_k, dual_pass = _RERANK_CONFIG[mode]

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

        # Stage 1c: BM25
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([_tokenize(d) for d in session_docs])
        bm25_raw = bm25.get_scores(_tokenize(query))
        bm25_max = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
        bm25_norm = bm25_raw / bm25_max

        stage1 = {}
        for i, sid in enumerate(session_ids):
            stage1[sid] = vec_scores.get(sid, 0.0) * 0.85 + float(bm25_norm[i]) * 0.15

        # Stage 2: cross-encoder rerank
        cands = sorted(stage1.items(), key=lambda x: x[1], reverse=True)[:rerank_k]
        csids = [s for s, _ in cands]

        id_to_idx = {sid: i for i, sid in enumerate(session_ids)}

        if dual_pass and all_docs:
            pairs_user = [(query, session_docs[id_to_idx[s]][:512]) for s in csids]
            pairs_all = [(query, all_docs[id_to_idx[s]][:512]) for s in csids]
            ce_scores = np.maximum(
                self.reranker.predict(pairs_user),
                self.reranker.predict(pairs_all),
            )
        else:
            # Single pass: user text + snippet of full text
            pairs = []
            for s in csids:
                idx = id_to_idx[s]
                doc = session_docs[idx][:384]
                if all_docs:
                    doc += "\n" + all_docs[idx][:128]
                pairs.append((query, doc))
            ce_scores = self.reranker.predict(pairs)

        ce_min, ce_max = float(ce_scores.min()), float(ce_scores.max())
        ce_range = ce_max - ce_min if ce_max != ce_min else 1.0
        s1_max = max(stage1.values())

        final = {}
        for i, sid in enumerate(csids):
            ce_norm = (float(ce_scores[i]) - ce_min) / ce_range
            s1_norm = stage1[sid] / s1_max
            final[sid] = ce_norm * 0.4 + s1_norm * 0.6

        # Backfill unreranked candidates
        min_reranked = min(final.values()) if final else 0
        for sid, s1 in stage1.items():
            if sid not in final:
                final[sid] = (s1 / s1_max) * min_reranked * 0.99

        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

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
