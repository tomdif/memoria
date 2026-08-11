"""Retrieval metrics shared by the benchmark runners.

LongMemEval's official retrieval code reports binary ``recall_any`` and
``recall_all`` plus an NDCG variant named ``ndcg_any``.  These helpers mirror
that implementation so local results can be compared without silently changing
the metric definition.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def rank_flat_bm25(
    query: str,
    documents: Sequence[str],
    document_ids: Sequence[str],
    k: int,
) -> list[str]:
    """Run the whitespace-tokenized flat BM25 baseline used by LongMemEval."""
    from rank_bm25 import BM25Okapi

    if len(documents) != len(document_ids):
        raise ValueError("documents and document_ids must have equal length")
    scores = BM25Okapi([document.split(" ") for document in documents]).get_scores(
        query.split(" ")
    )
    rankings = np.argsort(scores)[::-1][:k]
    return [document_ids[int(index)] for index in rankings]


def rank_dense(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    document_ids: Sequence[str],
    k: int,
) -> list[str]:
    """Run a normalized dot-product dense-retrieval baseline."""
    if len(document_embeddings) != len(document_ids):
        raise ValueError("document_embeddings and document_ids must have equal length")
    scores = np.asarray(document_embeddings @ query_embedding, dtype=float)
    rankings = np.argsort(scores)[::-1][:k]
    return [document_ids[int(index)] for index in rankings]


def recall_any_at_k(retrieved: Sequence[str], correct: Iterable[str], k: int) -> float:
    """Return 1 when at least one relevant item appears in the first *k*."""
    targets = set(correct)
    if not targets:
        return 0.0
    return float(bool(targets.intersection(retrieved[:k])))


def recall_all_at_k(retrieved: Sequence[str], correct: Iterable[str], k: int) -> float:
    """Return 1 when every relevant item appears in the first *k*."""
    targets = set(correct)
    if not targets:
        return 0.0
    return float(targets.issubset(set(retrieved[:k])))


def _official_dcg(relevances: Sequence[float], k: int) -> float:
    """Match LongMemEval's published ``eval_utils.dcg`` implementation."""
    values = list(relevances[:k])
    if not values:
        return 0.0
    return float(values[0] + sum(
        relevance / math.log2(rank)
        for rank, relevance in enumerate(values[1:], start=2)
    ))


def ndcg_any_at_k(retrieved: Sequence[str], correct: Iterable[str], k: int) -> float:
    """Return LongMemEval-compatible binary-relevance NDCG at *k*."""
    targets = set(correct)
    if not targets:
        return 0.0
    actual = [1.0 if item in targets else 0.0 for item in retrieved[:k]]
    ideal = [1.0] * min(len(targets), k)
    ideal_dcg = _official_dcg(ideal, k)
    return _official_dcg(actual, k) / ideal_dcg if ideal_dcg else 0.0


def metric_bundle(
    retrieved: Sequence[str],
    correct: Iterable[str],
    k: int,
) -> dict[str, float]:
    """Calculate the three official-style retrieval metrics at *k*."""
    return {
        f"recall_any@{k}": recall_any_at_k(retrieved, correct, k),
        f"recall_all@{k}": recall_all_at_k(retrieved, correct, k),
        f"ndcg_any@{k}": ndcg_any_at_k(retrieved, correct, k),
    }
