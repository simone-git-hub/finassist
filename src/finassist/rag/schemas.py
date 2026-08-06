from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class RefusalReason(StrEnum):
    LOW_RETRIEVAL_CONFIDENCE = "LOW_RETRIEVAL_CONFIDENCE"
    EMPTY_RETRIEVAL = "EMPTY_RETRIEVAL"
    ADVICE_BOUNDARY = "ADVICE_BOUNDARY"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"


class SourceCitation(BaseModel):
    index: int
    doc_id: str
    title: str
    source_url: str
    excerpt: str
    score: float


class RAGResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    refused: bool
    refusal_reason: RefusalReason | None = None
    latency_ms: int
