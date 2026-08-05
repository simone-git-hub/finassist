from __future__ import annotations

from finassist.config import ChunkingConfig
from finassist.models import RawDocument, TextChunk


def _split_with_overlap(text: str, size: int, overlap: int) -> list[tuple[int, int, str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    spans: list[tuple[int, int, str]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + size, text_len)
        chunk_text = text[start:end].strip()
        if chunk_text:
            spans.append((start, end, chunk_text))
        if end >= text_len:
            break
        start = end - overlap

    return spans


def chunk_document(document: RawDocument, config: ChunkingConfig) -> list[TextChunk]:
    spans = _split_with_overlap(
        document.text,
        size=config.size,
        overlap=config.overlap,
    )

    chunks: list[TextChunk] = []
    for index, (char_start, char_end, chunk_text) in enumerate(spans):
        if len(chunk_text) < config.min_chunk_chars:
            continue

        chunks.append(
            TextChunk(
                chunk_id=f"{document.doc_id}__{index:04d}",
                doc_id=document.doc_id,
                source_url=document.url,
                title=document.title,
                category=document.category,
                chunk_index=index,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
            )
        )

    return chunks


def chunk_documents(
    documents: list[RawDocument],
    config: ChunkingConfig,
) -> list[TextChunk]:
    all_chunks: list[TextChunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, config))
    return all_chunks
