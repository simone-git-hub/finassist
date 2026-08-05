from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from finassist.config import (
    AppConfig,
    get_app_config,
    load_yaml_config,
    project_root,
    resolve_chroma_dir,
    resolve_embedding_model,
)
from finassist.index.embedder import (
    EmbeddingBackend,
    TfidfEmbedder,
    create_embedder,
    tfidf_state_path,
)
from finassist.index.store import ChromaStore
from finassist.models import TextChunk


def load_chunks_jsonl(chunks_path: Path) -> list[TextChunk]:
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}. Run `finassist-ingest` first."
        )

    chunks: list[TextChunk] = []
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(TextChunk.model_validate_json(line))
    return chunks


def _resolve_config(config: AppConfig | None, root: Path) -> AppConfig:
    if config is not None:
        return config
    config_path = root / "configs" / "default.yaml"
    if config_path.exists():
        return AppConfig.model_validate(load_yaml_config(config_path))
    return get_app_config()


@dataclass
class IndexBuildResult:
    num_chunks: int
    collection_name: str
    persist_dir: Path
    embedding_model: str
    embedding_backend: str


def _prepare_embedder(
    cfg: AppConfig,
    chunks: list[TextChunk],
    *,
    persist_dir: Path,
    embedder: EmbeddingBackend | None = None,
) -> EmbeddingBackend:
    if embedder is not None:
        return embedder

    backend = cfg.index.embedding_backend
    model_name = resolve_embedding_model(cfg)
    embedder = create_embedder(model_name, backend=backend)

    if isinstance(embedder, TfidfEmbedder):
        texts = [chunk.text for chunk in chunks]
        embedder.fit(texts)
        embedder.save(tfidf_state_path(persist_dir))

    return embedder


def build_index_from_chunks(
    chunks: list[TextChunk],
    config: AppConfig | None = None,
    *,
    root: Path | None = None,
    rebuild: bool = True,
    embedder: EmbeddingBackend | None = None,
) -> IndexBuildResult:
    base = root or project_root()
    cfg = _resolve_config(config, base)
    embedding_model = resolve_embedding_model(cfg)
    persist_dir = resolve_chroma_dir(cfg, base)

    store = ChromaStore(
        persist_dir=persist_dir,
        collection_name=cfg.index.collection_name,
        embedding_model_name=embedding_model,
    )
    prepared_embedder = _prepare_embedder(cfg, chunks, persist_dir=persist_dir, embedder=embedder)

    if rebuild:
        store.reset()

    batch_size = cfg.index.batch_size
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = prepared_embedder.embed_texts([chunk.text for chunk in batch], batch_size=batch_size)
        store.upsert_chunks(batch, vectors.tolist())

    summary = IndexBuildResult(
        num_chunks=len(chunks),
        collection_name=cfg.index.collection_name,
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        embedding_backend=cfg.index.embedding_backend,
    )

    summary_dir = base / "data" / "index"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "index_summary.json").write_text(
        json.dumps(
            {
                "num_chunks": summary.num_chunks,
                "collection_name": summary.collection_name,
                "persist_dir": str(summary.persist_dir),
                "embedding_model": summary.embedding_model,
                "embedding_backend": summary.embedding_backend,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def build_index_from_config(
    config: AppConfig | None = None,
    *,
    root: Path | None = None,
    rebuild: bool = True,
    chunks_path: Path | None = None,
    embedder: EmbeddingBackend | None = None,
) -> IndexBuildResult:
    base = root or project_root()
    cfg = _resolve_config(config, base)
    path = chunks_path or (base / cfg.index.chunks_path)
    chunks = load_chunks_jsonl(path)
    return build_index_from_chunks(
        chunks,
        cfg,
        root=base,
        rebuild=rebuild,
        embedder=embedder,
    )


def load_embedder_from_index(config: AppConfig, *, root: Path | None = None) -> EmbeddingBackend:
    base = root or project_root()
    persist_dir = resolve_chroma_dir(config, base)
    summary_path = base / "data" / "index" / "index_summary.json"

    backend = config.index.embedding_backend
    if summary_path.exists():
        backend = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "embedding_backend",
            backend,
        )

    if backend == "tfidf":
        return TfidfEmbedder.load(tfidf_state_path(persist_dir))

    return create_embedder(resolve_embedding_model(config), backend=backend)
