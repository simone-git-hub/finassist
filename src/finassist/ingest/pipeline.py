from __future__ import annotations

import json
from pathlib import Path

from finassist.config import AppConfig, get_app_config, load_yaml_config, project_root
from finassist.ingest.chunk import chunk_documents
from finassist.ingest.fetch import (
    content_hash,
    fetch_url,
    load_manifest,
    resolve_ingest_paths,
)
from finassist.ingest.parse import extract_main_text
from finassist.models import RawDocument, TextChunk


def _resolve_config(config: AppConfig | None, root: Path) -> AppConfig:
    if config is not None:
        return config
    config_path = root / "configs" / "default.yaml"
    if config_path.exists():
        return AppConfig.model_validate(load_yaml_config(config_path))
    return get_app_config()


def ingest_manifest(
    config: AppConfig | None = None,
    *,
    root: Path | None = None,
    force_refresh: bool = False,
    limit: int | None = None,
) -> list[RawDocument]:
    base = root or project_root()
    cfg = _resolve_config(config, base)
    manifest_path, cache_dir, _ = resolve_ingest_paths(cfg, base)
    entries = load_manifest(manifest_path)

    if limit is not None:
        entries = entries[:limit]

    documents: list[RawDocument] = []
    for entry in entries:
        url = str(entry.url)
        local_path = Path(entry.local_path) if entry.local_path else None
        if local_path is not None and not local_path.is_absolute():
            local_path = base / local_path

        html, fetched_at, from_cache = fetch_url(
            url,
            cache_dir=cache_dir,
            timeout_s=cfg.ingest.request_timeout_s,
            user_agent=cfg.ingest.user_agent,
            force_refresh=force_refresh,
            local_path=local_path,
        )
        text, extracted_title = extract_main_text(html, fallback_title=entry.title)
        if len(text) < cfg.chunking.min_chunk_chars:
            raise ValueError(
                f"Document '{entry.doc_id}' produced too little text ({len(text)} chars)."
            )

        documents.append(
            RawDocument(
                doc_id=entry.doc_id,
                url=url,
                source=entry.source,
                title=extracted_title or entry.title,
                category=entry.category,
                text=text,
                fetched_at=fetched_at,
                content_hash=content_hash(text),
                metadata={
                    "license_note": entry.license_note,
                    "from_cache": from_cache,
                },
            )
        )

    return documents


def save_processed_documents(
    documents: list[RawDocument],
    *,
    processed_dir: Path,
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "documents.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(document.model_dump_json())
            handle.write("\n")

    return output_path


def save_chunks(chunks: list[TextChunk], *, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "chunks.jsonl"

    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json())
            handle.write("\n")

    return output_path


def run_ingest_pipeline(
    config: AppConfig | None = None,
    *,
    root: Path | None = None,
    force_refresh: bool = False,
    limit: int | None = None,
) -> tuple[list[RawDocument], list[TextChunk], Path, Path]:
    base = root or project_root()
    cfg = _resolve_config(config, base)
    _, _, processed_dir = resolve_ingest_paths(cfg, base)

    documents = ingest_manifest(
        cfg,
        root=base,
        force_refresh=force_refresh,
        limit=limit,
    )
    chunks = chunk_documents(documents, cfg.chunking)

    documents_path = save_processed_documents(documents, processed_dir=processed_dir)
    chunks_path = save_chunks(chunks, processed_dir=processed_dir)

    summary = {
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "documents_path": str(documents_path),
        "chunks_path": str(chunks_path),
    }
    summary_path = processed_dir / "ingest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return documents, chunks, documents_path, chunks_path
