"""
Embedding layer — local sentence-transformers, no API calls.

Embeddings are used for:
  - Vector similarity search (Layer 2 fallback, Layer 3 cluster centroids)
  - Scoped retrieval in pass 2
  - Cluster summary matching
"""

from __future__ import annotations

import numpy as np


EMBEDDING_DTYPE = np.float32


def serialize_embedding(embedding: np.ndarray | None) -> bytes | None:
    """Serialize an embedding using the canonical on-disk dtype."""
    if embedding is None:
        return None
    return np.asarray(embedding, dtype=EMBEDDING_DTYPE).tobytes()


def deserialize_embedding(
    data: bytes | None,
    expected_dimension: int | None = None,
) -> np.ndarray | None:
    """Deserialize canonical float32 data, with legacy float64 compatibility."""
    if not data:
        return None

    value = np.frombuffer(data, dtype=EMBEDDING_DTYPE)
    if expected_dimension is None or value.size == expected_dimension:
        return value

    # Early versions wrote the source dtype without metadata. Accept old
    # float64 blobs when their decoded dimension identifies them unambiguously.
    if len(data) % np.dtype(np.float64).itemsize == 0:
        legacy = np.frombuffer(data, dtype=np.float64)
        if legacy.size == expected_dimension:
            return legacy.astype(EMBEDDING_DTYPE)
    return None


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
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Embed one or more texts. Returns (n, dim) array."""
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(encoded, dtype=EMBEDDING_DTYPE)

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
