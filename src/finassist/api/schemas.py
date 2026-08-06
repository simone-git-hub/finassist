from pydantic import BaseModel, Field

from finassist.rag.schemas import RAGResponse, RefusalReason, SourceCitation


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class HealthResponse(BaseModel):
    status: str = "ok"


class AskResponse(RAGResponse):
    pass


__all__ = [
    "AskRequest",
    "AskResponse",
    "HealthResponse",
    "RAGResponse",
    "RefusalReason",
    "SourceCitation",
]
