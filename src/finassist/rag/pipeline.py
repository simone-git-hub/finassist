from __future__ import annotations

import time

from finassist.config import AppConfig, get_app_config
from finassist.index.retriever import Retriever, create_retriever
from finassist.models import ScoredChunk
from finassist.rag.guardrails import (
    check_question_guardrails,
    check_retrieval_guardrails,
    filter_chunks_for_context,
    refusal_message,
)
from finassist.rag.llm import LLMBackend, create_llm
from finassist.rag.schemas import RAGResponse, RefusalReason, SourceCitation


class RAGPipeline:
    def __init__(
        self,
        config: AppConfig,
        retriever: Retriever,
        llm: LLMBackend,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str, *, top_k: int | None = None) -> RAGResponse:
        started = time.perf_counter()

        question_check = check_question_guardrails(question, self.config)
        if question_check.refuse and question_check.reason is not None:
            return self._refused(question_check.reason, started)

        chunks = self.retriever.search(question, top_k=top_k)
        retrieval_check = check_retrieval_guardrails(chunks, self.config)
        if retrieval_check.refuse and retrieval_check.reason is not None:
            return self._refused(retrieval_check.reason, started)

        context_chunks = filter_chunks_for_context(chunks, self.config)
        answer = self.llm.generate(
            question=question,
            context_chunks=[chunk.text for chunk in context_chunks],
            max_words=self.config.guardrails.max_answer_words,
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        return RAGResponse(
            answer=answer.strip(),
            sources=self._build_sources(context_chunks),
            refused=False,
            refusal_reason=None,
            latency_ms=latency_ms,
        )

    def _refused(self, reason: RefusalReason, started: float) -> RAGResponse:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return RAGResponse(
            answer=refusal_message(reason),
            sources=[],
            refused=True,
            refusal_reason=reason,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _build_sources(chunks: list[ScoredChunk]) -> list[SourceCitation]:
        sources: list[SourceCitation] = []
        for index, chunk in enumerate(chunks, start=1):
            excerpt = chunk.text[:240] + ("..." if len(chunk.text) > 240 else "")
            sources.append(
                SourceCitation(
                    index=index,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    source_url=chunk.source_url,
                    excerpt=excerpt,
                    score=chunk.score,
                )
            )
        return sources


def build_pipeline(
    config: AppConfig | None = None,
    *,
    retriever: Retriever | None = None,
    llm: LLMBackend | None = None,
) -> RAGPipeline:
    cfg = config or get_app_config()
    active_retriever = retriever or create_retriever(cfg)
    active_llm = llm or create_llm(cfg)
    return RAGPipeline(cfg, active_retriever, active_llm)
