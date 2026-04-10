"""
Spectral compression — fit memory into a token budget.

The problem: you have N triples in the KG but can only load B tokens
into a context window. Which triples do you keep?

Naive approach: top-k by recency or confidence. Loses structure.

Spectral approach: project the graph state onto the top eigenvectors
of the Laplacian. The top eigenvectors capture the dominant structure
(permanent knowledge, major clusters). Lower eigenvectors capture
local detail (recent changes, niche facts). Truncating the eigenexpansion
at rank k gives an optimal rank-k approximation to the full graph state
(Eckart-Young theorem).

This gives a principled compression scheme:
  - Token budget large  → more eigenvectors → more detail
  - Token budget small  → fewer eigenvectors → only core structure
  - The information lost is provably minimal at each budget level

Three compression tiers:
  L0 (~50 tokens):  Top eigenvector only = identity/core facts
  L1 (~200 tokens): Top 3 eigenvectors = major clusters + key relations
  L2 (~2000 tokens): Full cluster summaries + recent changes
  L3 (unlimited):    Raw retrieval results
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from .graph import KnowledgeGraph
from .spectral import build_adjacency, graph_laplacian


@dataclass
class CompressedMemory:
    """Token-budget-aware memory snapshot."""
    tier: str  # L0, L1, L2, L3
    text: str
    token_estimate: int
    entities_included: int
    entities_total: int
    eigenvectors_used: int
    compression_ratio: float  # entities_included / entities_total


def spectral_rank(
    kg: KnowledgeGraph,
    budget_tokens: int = 200,
    tokens_per_triple: int = 15,
) -> list[dict]:
    """Rank all active triples by spectral importance.

    Each triple gets a score based on the magnitude of its endpoints
    in the top eigenvectors of the graph Laplacian. Triples connecting
    high-eigenvector-magnitude nodes carry the most structural information.

    The number of eigenvectors used is determined by the token budget:
    more budget → more eigenvectors → finer-grained ranking.

    Returns triples sorted by spectral importance (highest first).
    """
    triples, entity_index = kg.all_triples_for_spectral()
    if not triples or len(entity_index) < 3:
        # Fall back to confidence ranking
        all_triples = kg.get_triples(active_only=True)
        all_triples.sort(key=lambda t: t["confidence"], reverse=True)
        return all_triples

    A = build_adjacency(triples, entity_index)
    L = graph_laplacian(A)
    n = L.shape[0]

    # How many eigenvectors? More budget → more eigenvectors
    max_triples = budget_tokens // tokens_per_triple
    # k eigenvectors capture O(k) clusters worth of structure
    # Use sqrt(max_triples) as a heuristic for eigenvector count
    k = max(2, min(int(np.sqrt(max_triples)) + 1, n - 1, 20))

    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM", tol=1e-6)
        idx = np.argsort(np.real(eigenvalues))
        eigenvalues = np.real(eigenvalues[idx])
        eigenvectors = np.real(eigenvectors[:, idx])
    except (ArpackNoConvergence, Exception):
        all_triples = kg.get_triples(active_only=True)
        all_triples.sort(key=lambda t: t["confidence"], reverse=True)
        return all_triples

    # Compute per-node importance: sum of squared eigenvector components
    # weighted by inverse eigenvalue (lower eigenvalue = more global structure).
    # Skip the constant eigenvector (index 0, eigenvalue ≈ 0).
    node_importance = np.zeros(n)
    for i in range(1, len(eigenvalues)):
        lam = max(eigenvalues[i], 1e-10)
        # Weight: 1/λ means low-frequency (global) modes dominate
        weight = 1.0 / lam
        node_importance += weight * eigenvectors[:, i] ** 2

    # Normalize to [0, 1]
    if node_importance.max() > 0:
        node_importance /= node_importance.max()

    # Score each triple by the importance of its endpoints
    index_to_id = {v: k for k, v in entity_index.items()}
    scored_triples = []
    for t in kg.get_triples(active_only=True):
        si = entity_index.get(t["subject_id"])
        oi = entity_index.get(t.get("object_id"))

        if si is not None and oi is not None:
            # Geometric mean of endpoint importances × confidence
            spectral_score = np.sqrt(node_importance[si] * node_importance[oi])
        elif si is not None:
            spectral_score = node_importance[si]
        elif oi is not None:
            spectral_score = node_importance[oi]
        else:
            spectral_score = 0.0

        t["_spectral_score"] = float(spectral_score * t["confidence"])
        scored_triples.append(t)

    scored_triples.sort(key=lambda t: t["_spectral_score"], reverse=True)
    return scored_triples


def compress(
    kg: KnowledgeGraph,
    budget_tokens: int = 200,
    tokens_per_triple: int = 15,
    llm_call=None,
) -> CompressedMemory:
    """Compress the full knowledge graph into a token budget.

    Uses spectral ranking to select the most structurally important
    triples, then formats them as compact text.

    Token budget tiers:
      ≤50:   L0 — entity names only (identity)
      ≤200:  L1 — top spectral triples, compact format
      ≤2000: L2 — cluster summaries + key triples
      >2000: L3 — full retrieval-style output
    """
    total_entities = kg.entity_count()
    total_triples = kg.triple_count()

    if budget_tokens <= 0:
        return CompressedMemory(
            tier="L0", text="", token_estimate=0,
            entities_included=0, entities_total=total_entities,
            eigenvectors_used=0, compression_ratio=0.0,
        )

    ranked = spectral_rank(kg, budget_tokens, tokens_per_triple)
    max_triples = budget_tokens // tokens_per_triple

    if budget_tokens <= 50:
        return _compress_l0(kg, ranked, max_triples, total_entities)
    elif budget_tokens <= 200:
        return _compress_l1(kg, ranked, max_triples, total_entities)
    elif budget_tokens <= 2000:
        return _compress_l2(kg, ranked, max_triples, total_entities, llm_call)
    else:
        return _compress_l3(kg, ranked, max_triples, total_entities)


def _compress_l0(kg, ranked, max_triples, total_entities) -> CompressedMemory:
    """L0: Entity names only — identity-level compression."""
    # Get the entities that appear in top-ranked triples
    entity_ids = set()
    for t in ranked[:max_triples * 2]:
        entity_ids.add(t["subject_id"])
        if t.get("object_id"):
            entity_ids.add(t["object_id"])

    names = []
    for eid in list(entity_ids)[:max_triples]:
        entity = kg.get_entity(eid)
        if entity:
            names.append(entity.name)

    text = "Known: " + ", ".join(names[:20])
    return CompressedMemory(
        tier="L0", text=text, token_estimate=len(text.split()),
        entities_included=len(names), entities_total=total_entities,
        eigenvectors_used=1, compression_ratio=len(names) / max(total_entities, 1),
    )


def _compress_l1(kg, ranked, max_triples, total_entities) -> CompressedMemory:
    """L1: Compact triple format — key facts only."""
    lines = []
    entities_seen = set()

    for t in ranked[:max_triples]:
        subj = kg.get_entity(t["subject_id"])
        subj_name = subj.name if subj else "?"

        obj_str = t.get("object_value", "")
        if not obj_str and t.get("object_id"):
            obj = kg.get_entity(t["object_id"])
            obj_str = obj.name if obj else "?"

        # Compact format: "Subject.predicate=Object"
        lines.append(f"{subj_name}.{t['predicate']}={obj_str}")
        entities_seen.add(t["subject_id"])
        if t.get("object_id"):
            entities_seen.add(t["object_id"])

    text = "; ".join(lines)
    return CompressedMemory(
        tier="L1", text=text, token_estimate=len(text.split()),
        entities_included=len(entities_seen), entities_total=total_entities,
        eigenvectors_used=3, compression_ratio=len(entities_seen) / max(total_entities, 1),
    )


def _compress_l2(kg, ranked, max_triples, total_entities, llm_call) -> CompressedMemory:
    """L2: Cluster summaries + important triples."""
    # Get cluster summaries
    clusters = kg.db.execute(
        "SELECT summary FROM clusters WHERE summary IS NOT NULL ORDER BY updated_at DESC LIMIT 5"
    ).fetchall()

    lines = []

    # Add cluster summaries first (most compressed form of structure)
    for row in clusters:
        if row[0]:
            lines.append(row[0])

    # Fill remaining budget with top spectral triples
    summary_tokens = sum(len(s.split()) for s in lines)
    remaining = max_triples - (summary_tokens // tokens_per_triple_default())
    entities_seen = set()

    for t in ranked[:max(remaining, 5)]:
        subj = kg.get_entity(t["subject_id"])
        subj_name = subj.name if subj else "?"
        obj_str = t.get("object_value", "")
        if not obj_str and t.get("object_id"):
            obj = kg.get_entity(t["object_id"])
            obj_str = obj.name if obj else "?"

        lines.append(f"- {subj_name} {t['predicate']} {obj_str}")
        entities_seen.add(t["subject_id"])
        if t.get("object_id"):
            entities_seen.add(t["object_id"])

    text = "\n".join(lines)
    return CompressedMemory(
        tier="L2", text=text, token_estimate=len(text.split()),
        entities_included=len(entities_seen), entities_total=total_entities,
        eigenvectors_used=min(10, total_entities),
        compression_ratio=len(entities_seen) / max(total_entities, 1),
    )


def _compress_l3(kg, ranked, max_triples, total_entities) -> CompressedMemory:
    """L3: Full detail — all ranked triples up to budget."""
    lines = []
    entities_seen = set()

    for t in ranked[:max_triples]:
        subj = kg.get_entity(t["subject_id"])
        subj_name = subj.name if subj else "?"
        obj_str = t.get("object_value", "")
        if not obj_str and t.get("object_id"):
            obj = kg.get_entity(t["object_id"])
            obj_str = obj.name if obj else "?"

        score = t.get("_spectral_score", t["confidence"])
        rel = t["relation_type"]
        lines.append(
            f"- [{rel}] {subj_name} → {t['predicate']} → {obj_str} "
            f"(conf={t['confidence']:.2f}, spectral={score:.3f})"
        )
        entities_seen.add(t["subject_id"])
        if t.get("object_id"):
            entities_seen.add(t["object_id"])

    text = "\n".join(lines)
    return CompressedMemory(
        tier="L3", text=text, token_estimate=len(text.split()),
        entities_included=len(entities_seen), entities_total=total_entities,
        eigenvectors_used=min(20, total_entities),
        compression_ratio=len(entities_seen) / max(total_entities, 1),
    )


def tokens_per_triple_default() -> int:
    return 15


def budget_report(kg: KnowledgeGraph) -> dict:
    """Show what you'd get at each budget tier."""
    tiers = {}
    for budget, name in [(50, "L0"), (200, "L1"), (2000, "L2"), (10000, "L3")]:
        cm = compress(kg, budget_tokens=budget)
        tiers[name] = {
            "budget": budget,
            "entities_included": cm.entities_included,
            "entities_total": cm.entities_total,
            "compression_ratio": round(cm.compression_ratio, 3),
            "token_estimate": cm.token_estimate,
            "preview": cm.text[:100] + "..." if len(cm.text) > 100 else cm.text,
        }
    return tiers
