import numpy as np

from finassist.index.store import ChromaStore
from finassist.models import TextChunk


def test_chroma_store_upsert_and_query(tmp_path) -> None:
    store = ChromaStore(
        persist_dir=tmp_path / "chroma",
        collection_name="test_chunks",
        embedding_model_name="manual",
    )
    store.reset()

    chunks = [
        TextChunk(
            chunk_id="doc-a__0000",
            doc_id="doc-a",
            source_url="https://example.com/a",
            title="Freeze card",
            category="cards",
            chunk_index=0,
            text="Freeze your card in the mobile app under Cards settings.",
            char_start=0,
            char_end=50,
        ),
        TextChunk(
            chunk_id="doc-b__0000",
            doc_id="doc-b",
            source_url="https://example.com/b",
            title="Dispute payment",
            category="cards",
            chunk_index=0,
            text="Report a problem on a transaction to start a dispute.",
            char_start=0,
            char_end=50,
        ),
    ]

    query_vector = np.array([1.0, 0.0], dtype=float)
    other_vector = np.array([0.0, 1.0], dtype=float)

    store.upsert_chunks(chunks, [query_vector.tolist(), other_vector.tolist()])
    assert store.count() == 2

    results = store.collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=1,
        include=["metadatas", "distances"],
    )

    assert results["ids"][0][0] == "doc-a__0000"
