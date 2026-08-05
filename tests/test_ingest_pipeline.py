from finassist.ingest.pipeline import run_ingest_pipeline


def test_offline_ingest_pipeline_writes_outputs(tmp_path) -> None:
    root = tmp_path
    (root / "data" / "raw").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "tests" / "fixtures").mkdir(parents=True)

    fixture_html = """<!DOCTYPE html><html><body><article><h1>Freeze card</h1>
    <p>Freeze your card from the app under Cards settings immediately if lost.</p>
    </article></body></html>"""
    (root / "tests" / "fixtures" / "sample_faq.html").write_text(fixture_html)

    manifest = (
        "doc_id,url,source,title,category,license_note,local_path\n"
        "example-freeze-card,https://example.com/help/freeze-card,ExampleBank,"
        "How to freeze your card,cards,fixture,tests/fixtures/sample_faq.html\n"
    )
    (root / "data" / "raw" / "manifest.csv").write_text(manifest)

    config_yaml = """
chunking:
  size: 80
  overlap: 10
  min_chunk_chars: 20
ingest:
  manifest_path: data/raw/manifest.csv
  cache_dir: data/raw/cache
  processed_dir: data/raw/processed
"""
    (root / "configs" / "default.yaml").write_text(config_yaml)

    documents, chunks, documents_path, chunks_path = run_ingest_pipeline(root=root)

    assert len(documents) == 1
    assert len(chunks) >= 1
    assert documents_path.exists()
    assert chunks_path.exists()
