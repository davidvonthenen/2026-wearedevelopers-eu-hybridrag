"""Embedding utilities using sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Settings, load_settings


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    """Lazy-load and cache the sentence-transformers model."""

    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def load_embedder() -> "EmbeddingModel":
    """Return a cached embedding wrapper."""

    return EmbeddingModel()


@dataclass
class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model.

    The vector index dimension is derived from the model unless
    ``EMBEDDING_DIMENSION`` is provided. That keeps the ingestion code boring,
    which is the highest compliment infrastructure code can receive.
    """

    settings: Settings
    _cached_model: Optional[SentenceTransformer] = None

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self._cached_model = None

    @property
    def model_name(self) -> str:
        return self.settings.embedding_model

    @property
    def model(self) -> SentenceTransformer:
        # Loading the model is deferred until the first embedding call so CLI
        # argument validation can fail quickly without downloading model weights.
        if self._cached_model is None:
            self._cached_model = _load_model(self.model_name)
        return self._cached_model

    @property
    def dimension(self) -> int:
        configured = self.settings.embedding_dimension
        if configured is not None:
            return int(configured)

        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            vec = self.encode(["_dimension_probe_"])[0]
            return int(vec.shape[-1])

    def encode(self, texts: Iterable[str]) -> List[np.ndarray]:
        """Encode text to L2-normalized numpy arrays for cosine kNN."""

        items = [str(t or "") for t in texts]
        if not items:
            return []

        # Normalization aligns the vectors with the cosine-similarity OpenSearch
        # index mapping created during ingestion.
        arr = self.model.encode(
            items,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return [arr[i] for i in range(arr.shape[0])]
        if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(items) == 1:
            return [arr]
        return [np.asarray(v, dtype=float) for v in arr]


def embed_question(question: str) -> List[float]:
    """Embed a single query into a plain Python list."""

    embedder = load_embedder()
    return to_list(embedder.encode([question])[0])


def to_list(vec: Union[np.ndarray, Sequence[float], List[float]]) -> List[float]:
    """Convert a vector-like object to a JSON-serializable list of floats."""

    if isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()
    return [float(x) for x in vec]


__all__ = ["EmbeddingModel", "load_embedder", "embed_question", "to_list"]
