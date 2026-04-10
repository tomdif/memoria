"""
Embedding layer — local sentence-transformers, no API calls.

Embeddings are used for:
  - Vector similarity search (Layer 2 fallback, Layer 3 cluster centroids)
  - Scoped retrieval in pass 2
  - Cluster summary matching
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache


class Embedder:
    """Lazy-loaded local embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Embed one or more texts. Returns (n, dim) array."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (dim,) array."""
        return self.embed(text)[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between a (n, d) and b (m, d). Returns (n, m) matrix.

    Assumes inputs are already L2-normalized (which sentence-transformers does).
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return a @ b.T


def top_k_similar(query_emb: np.ndarray, corpus_embs: np.ndarray,
                  k: int = 10) -> list[tuple[int, float]]:
    """Return top-k most similar indices and scores."""
    if corpus_embs.shape[0] == 0:
        return []
    sims = cosine_similarity(query_emb, corpus_embs).flatten()
    if k >= len(sims):
        top_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, k)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [(int(i), float(sims[i])) for i in top_idx]
