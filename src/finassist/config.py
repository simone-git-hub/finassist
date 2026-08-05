from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ChunkingConfig(BaseModel):
    size: int = 512
    overlap: int = 64
    min_chunk_chars: int = 100


class RetrievalConfig(BaseModel):
    top_k: int = 5
    min_score: float = 0.55
    embedding_model: str = "BAAI/bge-small-en-v1.5"


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    max_tokens: int = 256
    temperature: float = 0.1


class GuardrailsConfig(BaseModel):
    refuse_out_of_domain: bool = True
    max_answer_words: int = 120


class EvalConfig(BaseModel):
    gold_path: str = "data/eval/qa_gold.jsonl"
    k_values: list[int] = Field(default_factory=lambda: [3, 5])


class IngestConfig(BaseModel):
    manifest_path: str = "data/raw/manifest.csv"
    cache_dir: str = "data/raw/cache"
    processed_dir: str = "data/raw/processed"
    request_timeout_s: float = 20.0
    user_agent: str = "FinAssistBot/0.1 (+portfolio project; educational use)"


class AppConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str | None = None
    chroma_persist_dir: str = "./data/index/chroma"
    min_retrieval_score: float | None = None


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or project_root() / "configs" / "default.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache
def get_app_config(config_path: str | None = None) -> AppConfig:
    path = Path(config_path) if config_path else project_root() / "configs" / "default.yaml"
    data = load_yaml_config(path)
    return AppConfig.model_validate(data)


@lru_cache
def get_settings() -> Settings:
    return Settings()
