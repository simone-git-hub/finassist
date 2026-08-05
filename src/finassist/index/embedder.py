from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...


@lru_cache(maxsize=2)
def _get_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = _get_sentence_transformer(model_name)

    def embed_texts(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 64,
            normalize_embeddings=True,
        )
        return np.asarray(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        vector = self._model.encode(query, normalize_embeddings=True)
        return np.asarray(vector)


class TfidfEmbedder:
    """Lightweight embedder for offline tests and quick local experiments."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        if texts:
            self._vectorizer.fit(texts)
            self._fitted = True

    def embed_texts(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        del batch_size
        if not texts:
            return np.empty((0, 0))
        if not self._fitted:
            self.fit(texts)
        matrix = self._vectorizer.transform(texts).toarray()
        return _normalize_rows(matrix)

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder must be fit before querying.")
        vector = self._vectorizer.transform([query]).toarray()[0]
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self._vectorizer, handle)
        self._fitted = True

    @classmethod
    def load(cls, path: Path) -> TfidfEmbedder:
        embedder = cls()
        with path.open("rb") as handle:
            embedder._vectorizer = pickle.load(handle)
        embedder._fitted = True
        return embedder


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def create_embedder(model_name: str, *, backend: str = "sentence_transformers") -> EmbeddingBackend:
    if backend == "tfidf":
        return TfidfEmbedder()
    return SentenceTransformerEmbedder(model_name)


def tfidf_state_path(persist_dir: Path) -> Path:
    return persist_dir / "tfidf_embedder.pkl"
