from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class ManifestEntry(BaseModel):
    doc_id: str
    url: HttpUrl
    source: str
    title: str
    category: str = "general"
    license_note: str = "Public help-center content; see source URL terms."
    local_path: str | None = None


class RawDocument(BaseModel):
    doc_id: str
    url: str
    source: str
    title: str
    category: str
    text: str
    fetched_at: datetime
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_url: str
    title: str
    category: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


class ScoredChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_url: str
    title: str
    category: str
    chunk_index: int
    text: str
    score: float
