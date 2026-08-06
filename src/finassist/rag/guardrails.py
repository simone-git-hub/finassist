from __future__ import annotations

import re
from dataclasses import dataclass

from finassist.config import AppConfig, get_settings
from finassist.models import ScoredChunk
from finassist.rag.schemas import RefusalReason

ADVICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(should i|is it worth|recommend).*\b(invest|crypto|stock|etf)\b", re.I),
    re.compile(r"\b(give me|need).*\b(tax|legal|investment)\s+advice\b", re.I),
    re.compile(r"\bwhat stock\b", re.I),
)

OUT_OF_DOMAIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bweather\b", re.I),
    re.compile(r"\b(recipe|cook)\b", re.I),
)

PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
)

REFUSAL_MESSAGES: dict[RefusalReason, str] = {
    RefusalReason.LOW_RETRIEVAL_CONFIDENCE: (
        "I don't have enough information in the available help articles to answer that confidently."
    ),
    RefusalReason.EMPTY_RETRIEVAL: (
        "I don't have enough information in the available help articles."
    ),
    RefusalReason.ADVICE_BOUNDARY: (
        "I can't provide legal, tax, or investment advice. Please consult a qualified professional."
    ),
    RefusalReason.OUT_OF_DOMAIN: (
        "That question is outside the scope of our finance support help articles."
    ),
}


@dataclass(frozen=True)
class GuardrailDecision:
    refuse: bool
    reason: RefusalReason | None = None


def refusal_message(reason: RefusalReason) -> str:
    return REFUSAL_MESSAGES[reason]


def sanitize_for_log(text: str) -> str:
    redacted = text
    for pattern in PII_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def check_question_guardrails(question: str, config: AppConfig) -> GuardrailDecision:
    if not config.guardrails.refuse_out_of_domain:
        return GuardrailDecision(refuse=False)

    for pattern in ADVICE_PATTERNS:
        if pattern.search(question):
            return GuardrailDecision(refuse=True, reason=RefusalReason.ADVICE_BOUNDARY)

    for pattern in OUT_OF_DOMAIN_PATTERNS:
        if pattern.search(question):
            return GuardrailDecision(refuse=True, reason=RefusalReason.OUT_OF_DOMAIN)

    return GuardrailDecision(refuse=False)


def check_retrieval_guardrails(
    chunks: list[ScoredChunk],
    config: AppConfig,
) -> GuardrailDecision:
    settings = get_settings()
    min_score = settings.min_retrieval_score or config.retrieval.min_score

    if not chunks:
        return GuardrailDecision(refuse=True, reason=RefusalReason.EMPTY_RETRIEVAL)

    if chunks[0].score < min_score:
        return GuardrailDecision(refuse=True, reason=RefusalReason.LOW_RETRIEVAL_CONFIDENCE)

    return GuardrailDecision(refuse=False)


def filter_chunks_for_context(
    chunks: list[ScoredChunk],
    config: AppConfig,
) -> list[ScoredChunk]:
    settings = get_settings()
    min_score = settings.min_retrieval_score or config.retrieval.min_score
    return [chunk for chunk in chunks if chunk.score >= min_score]
