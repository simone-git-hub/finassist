from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from finassist.models import TextChunk


class ChromaStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        *,
        embedding_model_name: str,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedding_model_name,
            },
        )

    @property
    def collection(self) -> Collection:
        return self._collection

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create_collection()

    def count(self) -> int:
        return self._collection.count()

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [_chunk_metadata(chunk) for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )


def _chunk_metadata(chunk: TextChunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "source_url": chunk.source_url,
        "title": chunk.title,
        "category": chunk.category,
        "chunk_index": chunk.chunk_index,
    }
