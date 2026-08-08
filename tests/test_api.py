from finassist.config import AppConfig
from finassist.models import ScoredChunk
from finassist.rag.llm import MockLLM
from finassist.rag.pipeline import RAGPipeline
from finassist.rag.schemas import RefusalReason


class FakeRetriever:
    def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        del query, top_k
        return [
            ScoredChunk(
                chunk_id="example-freeze-card__0000",
                doc_id="example-freeze-card",
                source_url="https://example.com/help/freeze-card",
                title="How to freeze your card",
                category="cards",
                chunk_index=0,
                text="Freeze your card from the app under Cards settings.",
                score=0.9,
            )
        ]


def test_ask_endpoint_returns_answer() -> None:
    from fastapi.testclient import TestClient

    from finassist.api.app import app, set_pipeline

    config = AppConfig()
    config.llm.provider = "mock"
    pipeline = RAGPipeline(config, FakeRetriever(), MockLLM())
    set_pipeline(pipeline)

    client = TestClient(app)
    response = client.post("/ask", json={"question": "How do I freeze my card?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["refused"] is False
    assert payload["sources"]
    assert payload["latency_ms"] >= 0

    set_pipeline(None)


def test_health_endpoint() -> None:
    from fastapi.testclient import TestClient

    from finassist.api.app import app, set_pipeline

    set_pipeline(None)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_refuses_advice_question() -> None:
    from fastapi.testclient import TestClient

    from finassist.api.app import app, set_pipeline

    config = AppConfig()
    pipeline = RAGPipeline(config, FakeRetriever(), MockLLM())
    set_pipeline(pipeline)

    client = TestClient(app)
    response = client.post("/ask", json={"question": "Should I invest in crypto?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["refused"] is True
    assert payload["refusal_reason"] == RefusalReason.ADVICE_BOUNDARY.value

    set_pipeline(None)
