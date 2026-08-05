from finassist.config import get_app_config, project_root


def test_project_root_contains_configs() -> None:
    root = project_root()
    assert (root / "configs" / "default.yaml").exists()


def test_app_config_loads_defaults() -> None:
    config = get_app_config()
    assert config.chunking.size == 512
    assert config.retrieval.top_k == 5
    assert config.ingest.manifest_path == "data/raw/manifest.csv"
