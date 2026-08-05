from finassist.ingest.fetch import load_manifest


def test_manifest_loads_seed_entries() -> None:
    entries = load_manifest()
    assert len(entries) >= 5
    assert any(entry.doc_id == "wise-freeze-card" for entry in entries)
