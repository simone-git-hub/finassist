from __future__ import annotations

from pathlib import Path

from finassist.config import (
    AppConfig,
    get_app_config,
    load_yaml_config,
    project_root,
    resolve_chroma_dir,
)
from finassist.index.builder import load_embedder_from_index
from finassist.index.embedder import EmbeddingBackend
from finassist.index.store import ChromaStore
from finassist.models import ScoredChunk


def _resolve_config(config: AppConfig | None, root: Path) -> AppConfig:
    if config is not None:
        return config
    config_path = root / "configs" / "default.yaml"
    if config_path.exists():
        return AppConfig.model_validate(load_yaml_config(config_path))
    return get_app_config()


def _distance_to_similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance))


class Retriever:
    def __init__(
        self,
        store: ChromaStore,
        embedder: EmbeddingBackend,
        *,
        default_top_k: int = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.default_top_k = default_top_k

    def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or self.default_top_k
        query_vector = self.embedder.embed_query(query).tolist()

        results = self.store.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        scored: list[ScoredChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=True):
            metadata = metadata or {}
            scored.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    doc_id=str(metadata.get("doc_id", "")),
                    source_url=str(metadata.get("source_url", "")),
                    title=str(metadata.get("title", "")),
                    category=str(metadata.get("category", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    text=text or "",
                    score=_distance_to_similarity(float(distance)),
                )
            )

        return scored


def create_retriever(
    config: AppConfig | None = None,
    *,
    root: Path | None = None,
    embedder: EmbeddingBackend | None = None,
) -> Retriever:
    base = root or project_root()
    cfg = _resolve_config(config, base)
    persist_dir = resolve_chroma_dir(cfg, base)

    store = ChromaStore(
        persist_dir=persist_dir,
        collection_name=cfg.index.collection_name,
        embedding_model_name=cfg.retrieval.embedding_model,
    )
    if store.count() == 0:
        raise RuntimeError(
            "Vector index is empty. Run `finassist-build-index` after ingestion."
        )

    active_embedder = embedder or load_embedder_from_index(cfg, root=base)
    return Retriever(store, active_embedder, default_top_k)
