from datetime import UTC, datetime

from finassist.config import ChunkingConfig
from finassist.ingest.chunk import chunk_document
from finassist.models import RawDocument


def _sample_document(text: str) -> RawDocument:
    return RawDocument(
        doc_id="sample-freeze-card",
        url="https://example.com/help/freeze-card",
        source="ExampleBank",
        title="How to freeze your card",
        category="cards",
        text=text,
        fetched_at=datetime.now(tz=UTC),
        content_hash="abc123",
    )


def test_chunk_document_respects_overlap_and_min_size() -> None:
    text = ("Freeze your card from the app. " * 40).strip()
    config = ChunkingConfig(size=120, overlap=20, min_chunk_chars=50)
    chunks = chunk_document(_sample_document(text), config)

    assert len(chunks) >= 2
    assert all(len(chunk.text) >= config.min_chunk_chars for chunk in chunks)
    assert chunks[0].doc_id == "sample-freeze-card"
    assert chunks[0].chunk_index == 0


def test_chunk_document_skips_tiny_tail() -> None:
    text = "A" * 100 + " tiny"
    config = ChunkingConfig(size=100, overlap=0, min_chunk_chars=20)
    chunks = chunk_document(_sample_document(text), config)

    assert len(chunks) == 1
    assert chunks[0].text == "A" * 100
