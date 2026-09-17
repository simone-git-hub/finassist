from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from finassist.api.schemas import AskRequest, AskResponse, HealthResponse
from finassist.config import get_app_config
from finassist.rag.pipeline import RAGPipeline, build_pipeline

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        _pipeline = build_pipeline(get_app_config())
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline unavailable. Run ingest and finassist-build-index first.",
        ) from exc
    return _pipeline


def set_pipeline(pipeline: RAGPipeline | None) -> None:
    global _pipeline
    _pipeline = pipeline


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    set_pipeline(None)


app = FastAPI(
    title="FinAssist RAG Copilot",
    description="Finance support copilot with retrieval-augmented generation.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> AskResponse:
    result = pipeline.ask(request.question, top_k=request.top_k)
    return AskResponse.model_validate(result.model