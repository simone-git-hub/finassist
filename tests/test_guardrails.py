from finassist.models import ScoredChunk
from finassist.rag.guardrails import (
    check_question_guardrails,
    check_retrieval_guardrails,
    refusal_message,
    sanitize_for_log,
)
from finassist.rag.schemas import RefusalReason


def _chunk(doc_id: str, text: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"{doc_id}__0000",
        doc_id=doc_id,
        source_url=f"https://example.com/{doc_id}",
        title=doc_id,
        category="cards",
        chunk_index=0,
        text=text,
        score=score,
    )


def test_refuses_investment_advice_question() -> None:
    from finassist.config import get_app_config

    config = get_app_config()
    decision = check_question_guardrails("Should I invest my savings in crypto?", config)
    assert decision.refuse is True
    assert decision.reason == RefusalReason.ADVICE_BOUNDARY


def test_refuses_out_of_domain_question() -> None:
    from finassist.config import get_app_config

    config = get_app_config()
    decision = check_question_guardrails("What is the weather in London?", config)
    assert decision.refuse is True
    assert decision.reason == RefusalReason.OUT_OF_DOMAIN


def test_refuses_low_retrieval_confidence() -> None:
    from finassist.config import get_app_config

    config = get_app_config()
    chunks = [_chunk("doc-a", "some text", score=0.2)]
    decision = check_retrieval_guardrails(chunks, config)
    assert decision.refuse is True
    assert decision.reason == RefusalReason.LOW_RETRIEVAL_CONFIDENCE


def test_refusal_messages_are_non_empty() -> None:
    for reason in RefusalReason:
        assert refusal_message(reason)


def test_sanitize_for_log_redacts_card_number() -> None:
    sanitized = sanitize_for_log("my card is 4111 1111 1111 1111")
    assert "[REDACTED]" in sanitized
    assert "4111" not in sanitized
