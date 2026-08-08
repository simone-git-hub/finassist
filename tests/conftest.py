from pathlib import Path

import pytest

from finassist.config import AppConfig, load_yaml_config


@pytest.fixture()
def rag_project_tree(tmp_path: Path) -> tuple[Path, AppConfig]:
    root = tmp_path
    (root / "configs").mkdir()
    (root / "tests" / "fixtures").mkdir(parents=True)
    (root / "data" / "eval").mkdir(parents=True)

    freeze_html = Path("tests/fixtures/sample_faq.html").read_text(encoding="utf-8")
    dispute_html = Path("tests/fixtures/sample_dispute.html").read_text(encoding="utf-8")
    (root / "tests" / "fixtures" / "sample_faq.html").write_text(freeze_html, encoding="utf-8")
    (root / "tests" / "fixtures" / "sample_dispute.html").write_text(dispute_html, encoding="utf-8")

    manifest = Path("data/raw/manifest.offline.csv").read_text(encoding="utf-8")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "manifest.offline.csv").write_text(manifest, encoding="utf-8")

    gold = Path("data/eval/qa_gold_offline.jsonl").read_text(encoding="utf-8")
    (root / "data" / "eval" / "qa_gold_offline.jsonl").write_text(gold, encoding="utf-8")

    config = AppConfig.model_validate(load_yaml_config(Path("configs/default.yaml")))
    config.ingest.manifest_path = "data/raw/manifest.offline.csv"
    config.index.embedding_backend = "tfidf"
    config.llm.provider = "mock"
    config.retrieval.min_score = 0.45
    config.eval.gold_path = "data/eval/qa_gold_offline.jsonl"

    import yaml

    (root / "configs" / "default.yaml").write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False),
        encoding="utf-8",
    )
    return root, config
