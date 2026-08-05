from pathlib import Path

import pytest

from finassist.config import AppConfig, load_yaml_config
from finassist.index.builder import build_index_from_chunks, load_chunks_jsonl
from finassist.index.retriever import create_retriever
from finassist.ingest.pipeline import run_ingest_pipeline


@pytest.fixture()
def project_tree(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "configs").mkdir()
    (root / "tests" / "fixtures").mkdir(parents=True)

    freeze_html = Path("tests/fixtures/sample_faq.html").read_text(encoding="utf-8")
    dispute_html = Path("tests/fixtures/sample_dispute.html").read_text(encoding="utf-8")
    (root / "tests" / "fixtures" / "sample_faq.html").write_text(freeze_html, encoding="utf-8")
    (root / "tests" / "fixtures" / "sample_dispute.html").write_text(dispute_html, encoding="utf-8")

    manifest = Path("data/raw/manifest.offline.csv").read_text(encoding="utf-8")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "manifest.offline.csv").write_text(manifest, encoding="utf-8")

    config = AppConfig.model_validate(load_yaml_config(Path("configs/default.yaml")))
    config.ingest.manifest_path = "data/raw/manifest.offline.csv"
    config.index.embedding_backend = "tfidf"

    import yaml

    (root / "configs" / "default.yaml").write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False),
        encoding="utf-8",
    )
    return root


def test_build_index_and_retrieve_freeze_card_query(project_tree: Path) -> None:
    config = AppConfig.model_validate(load_yaml_config(project_tree / "configs" / "default.yaml"))
    config.ingest.manifest_path = "data/raw/manifest.offline.csv"

    run_ingest_pipeline(config, root=project_tree)
    chunks = load_chunks_jsonl(project_tree / config.index.chunks_path)
    result = build_index_from_chunks(chunks, config, root=project_tree, rebuild=True)

    assert result.num_chunks >= 2

    retriever = create_retriever(config, root=project_tree)
    results = retriever.search("How do I freeze my card?", top_k=3)

    assert results
    assert results[0].doc_id == "example-freeze-card"
    assert results[0].score > 0.4
    assert "freeze" in results[0].text.lower()


def test_retriever_ranks_dispute_doc_for_dispute_query(project_tree: Path) -> None:
    config = AppConfig.model_validate(load_yaml_config(project_tree / "configs" / "default.yaml"))
    config.ingest.manifest_path = "data/raw/manifest.offline.csv"

    run_ingest_pipeline(config, root=project_tree)
    chunks = load_chunks_jsonl(project_tree / config.index.chunks_path)
    build_index_from_chunks(chunks, config, root=project_tree, rebuild=True)

    retriever = create_retriever(config, root=project_tree)
    results = retriever.search("I want to dispute an unrecognized payment", top_k=2)

    assert results[0].doc_id == "example-dispute"
