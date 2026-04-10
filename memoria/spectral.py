"""
CAG-inspired spectral analysis for the knowledge graph.

The key insight: the memory graph is a constrained surface model.
The Laplacian spectrum controls retrieval depth, cluster boundaries,
and consolidation dynamics — replacing arbitrary hyperparameters
with principled, data-driven quantities.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, ArpackNoConvergence


def build_adjacency(triples: list[dict], entity_index: dict[str, int]) -> sparse.csr_matrix:
    """Build sparse adjacency matrix from KG triples.

    Weights encode relation strength:
      fact=1.0, causal=1.5, depends=1.2, supersedes=0.5, contradicts=0.3
    Weighted by triple confidence. Symmetrized for spectral analysis.
    """
    n = len(entity_index)
    if n == 0:
        return sparse.csr_matrix((0, 0))

    relation_weights = {
        "fact": 1.0,
        "causal": 1.5,
        "depends": 1.2,
        "supersedes": 0.5,
        "contradicts": 0.3,
    }

    rows, cols, vals = [], [], []
    for t in triples:
        si = entity_index.get(t["subject_id"])
        oi = entity_index.get(t.get("object_id"))
        if si is None or oi is None:
            continue
        w = relation_weights.get(t["relation_type"], 1.0) * t.get("confidence", 1.0)
        rows.extend([si, oi])
        cols.extend([oi, si])
        vals.extend([w, w])

    if not rows:
        return sparse.csr_matrix((n, n))
    return sparse.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n))


def graph_laplacian(A: sparse.csr_matrix) -> sparse.csr_matrix:
    """Compute the normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    n = A.shape[0]
    if n == 0:
        return A

    degrees = np.array(A.sum(axis=1)).flatten()
    # Avoid division by zero for isolated nodes
    degrees_safe = np.where(degrees > 0, degrees, 1.0)
    degrees_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees_safe), 0.0)
    D_inv_sqrt = sparse.diags(degrees_inv_sqrt)
    L = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    return L.tocsr()


def spectral_gap(A: sparse.csr_matrix, k: int = 6) -> tuple[float, np.ndarray]:
    """Compute the spectral gap γ = λ_2 - λ_1 of the normalized Laplacian.

    Returns (gap, eigenvalues). The gap determines:
      - Retrieval depth: ~1/γ hops
      - Cluster separation quality
      - Mixing time of random walks on the graph

    For disconnected graphs, γ=0 and each component should be analyzed separately.
    """
    L = graph_laplacian(A)
    n = L.shape[0]
    if n < 3:
        return (1.0, np.array([0.0, 1.0][:n]))

    k_actual = min(k, n - 1)
    try:
        eigenvalues, _ = eigsh(L, k=k_actual, which="SM", tol=1e-6)
        eigenvalues = np.sort(np.real(eigenvalues))
        # λ_1 ≈ 0 (constant eigenvector), gap = λ_2
        gap = float(eigenvalues[1]) if len(eigenvalues) > 1 else 1.0
        return (max(gap, 1e-10), eigenvalues)
    except (ArpackNoConvergence, Exception):
        return (0.1, np.array([0.0, 0.1]))


def screening_radius(gap: float) -> int:
    """Compute the screening radius from the spectral gap.

    Beyond this distance in the graph, nodes are effectively independent
    (correlations decay as exp(-γ * d)). This gives the optimal walk depth
    for retrieval: searching beyond this radius adds noise, not signal.

    From the CAG screening theorem: Θ ~ 1/γ
    We use ceil(2/γ) to be conservative (capture 2 correlation lengths).
    """
    if gap <= 0:
        return 10  # fallback for disconnected graphs
    radius = int(np.ceil(2.0 / gap))
    return max(1, min(radius, 15))  # clamp to [1, 15]


def consolidation_operator(
    A: sparse.csr_matrix,
    confidences: np.ndarray,
    access_counts: np.ndarray,
    ages: np.ndarray,
    alpha: float = 0.6,
    beta: float = 0.2,
    gamma_decay: float = 0.2,
    half_life_days: float = 30.0,
) -> np.ndarray:
    """Compute new confidence scores via the spectral consolidation operator.

    The transfer matrix T combines:
      - Graph structure (neighbors reinforce each other): weight α
      - Access frequency (frequently recalled facts persist): weight β
      - Temporal decay (old unreferenced facts fade): weight γ

    T = α · D^{-1}A + β · diag(access_score) + γ · diag(recency_score)

    New confidence = T @ old_confidence, then normalized to [0, 1].

    The eigenstructure of T determines steady-state:
      - Top eigenvector = permanent knowledge
      - λ_2/λ_1 = retention rate per consolidation cycle
    """
    n = A.shape[0]
    if n == 0:
        return np.array([])

    # Normalize adjacency: D^{-1}A (random walk matrix)
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees_safe = np.where(degrees > 0, degrees, 1.0)
    rw = sparse.diags(1.0 / degrees_safe) @ A

    # Access score: log(1 + count) normalized to [0, 1]
    access_score = np.log1p(access_counts)
    if access_score.max() > 0:
        access_score /= access_score.max()

    # Recency score: exponential decay
    recency_score = np.exp(-np.log(2) * ages / half_life_days)

    # Transfer operator
    graph_component = alpha * (rw @ confidences)
    access_component = beta * access_score * confidences
    recency_component = gamma_decay * recency_score * confidences

    new_conf = graph_component + access_component + recency_component
    # Normalize to [0, 1]
    if new_conf.max() > 0:
        new_conf = np.clip(new_conf / new_conf.max(), 0, 1)
    return new_conf


def find_clusters(A: sparse.csr_matrix, n_clusters: int | None = None) -> list[list[int]]:
    """Spectral clustering using the Fiedler vector and recursive bisection.

    Uses the eigenvectors of the graph Laplacian to find natural clusters.
    When n_clusters is None, automatically determines the number of clusters
    from the eigenvalue gaps (largest gap after λ_1 indicates optimal k).
    """
    n = A.shape[0]
    if n <= 1:
        return [list(range(n))]

    L = graph_laplacian(A)
    # Compute enough eigenvalues to detect cluster count
    k = min(max(10, n // 5), n - 1) if n_clusters is None else min(n_clusters + 1, n - 1)
    k = max(k, 2)

    try:
        eigenvalues, eigenvectors = eigsh(L, k=k, which="SM", tol=1e-6)
        idx = np.argsort(np.real(eigenvalues))
        eigenvalues = np.real(eigenvalues[idx])
        eigenvectors = np.real(eigenvectors[:, idx])
    except (ArpackNoConvergence, Exception):
        return [list(range(n))]

    if n_clusters is None:
        # Find optimal k from largest eigenvalue gap
        if len(eigenvalues) < 3:
            n_clusters = 1
        else:
            gaps = np.diff(eigenvalues[1:])  # skip λ_1 ≈ 0
            n_clusters = int(np.argmax(gaps) + 2) if len(gaps) > 0 else 1
            n_clusters = max(1, min(n_clusters, n // 2))

    if n_clusters <= 1:
        return [list(range(n))]

    # Use first n_clusters eigenvectors (skip the constant one)
    features = eigenvectors[:, 1 : n_clusters + 1]

    # Simple k-means on spectral embedding
    clusters = _spectral_kmeans(features, n_clusters)
    return clusters


def _spectral_kmeans(X: np.ndarray, k: int, max_iter: int = 50) -> list[list[int]]:
    """Minimal k-means on spectral embedding. No sklearn dependency."""
    n, d = X.shape
    if d == 0 or k <= 1:
        return [list(range(n))]

    # Initialize centroids via k-means++
    rng = np.random.default_rng(42)
    centroids = np.empty((k, d))
    centroids[0] = X[rng.integers(n)]
    for i in range(1, k):
        dists = np.min([np.sum((X - centroids[j]) ** 2, axis=1) for j in range(i)], axis=0)
        probs = dists / (dists.sum() + 1e-30)
        centroids[i] = X[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centroids])
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids[i] = X[mask].mean(axis=0)

    clusters = [[] for _ in range(k)]
    for i, lbl in enumerate(labels):
        clusters[lbl].append(i)
    return [c for c in clusters if c]  # remove empty


def local_gap(
    A: sparse.csr_matrix, center_nodes: list[int], max_radius: int = 5
) -> tuple[float, int]:
    """Compute the spectral gap of the local subgraph around center_nodes.

    This gives an adaptive retrieval depth for a specific query region,
    rather than using the global gap. Returns (local_gap, recommended_depth).
    """
    if A.shape[0] == 0 or not center_nodes:
        return (1.0, 1)

    # BFS to find the local subgraph
    visited = set(center_nodes)
    frontier = set(center_nodes)
    for _ in range(max_radius):
        next_frontier = set()
        for node in frontier:
            neighbors = A[node].nonzero()[1]
            next_frontier.update(int(n) for n in neighbors if n not in visited)
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier

    nodes = sorted(visited)
    if len(nodes) < 3:
        return (1.0, 1)

    # Extract subgraph
    node_idx = {n: i for i, n in enumerate(nodes)}
    sub_A = A[np.ix_(nodes, nodes)]

    gap, _ = spectral_gap(sub_A)
    return (gap, screening_radius(gap))
