from finassist.index.builder import build_index_from_chunks, load_chunks_jsonl
from finassist.index.retriever import create_retriever
from finassist.ingest.pipeline import run_ingest_pipeline
from finassist.rag.llm import MockLLM
from finassist.rag.pipeline import RAGPipeline
from finassist.rag.schemas import RefusalReason


def test_rag_pipeline_answers_freeze_card_question(rag_project_tree) -> None:
    root, config = rag_project_tree
    config.ingest.manifest_path = "data/raw/manifest.offline.csv"

    run_ingest_pipeline(config, root=root)
    chunks = load_chunks_jsonl(root / config.index.chunks_path)
    build_index_from_chunks(chunks, config, root=root, rebuild=True)

    retriever = create_retriever(config, root=root)
    pipeline = RAGPipeline(config, retriever, MockLLM())
    result = pipeline.ask("How do I freeze my card?")

    assert result.refused is False
    assert result.sources
    assert result.sources[0].doc_id == "example-freeze-card"
    assert "context" in result.answer.lower() or "[1]" in result.answer


def test_rag_pipeline_refuses_investment_advice(rag_project_tree) -> None:
    root, config = rag_project_tree
    config.ingest.manifest_path = "data/raw/manifest.offline.csv"

    run_ingest_pipeline(config, root=root)
    chunks = load_chunks_jsonl(root / config.index.chunks_path)
    build_index_from_chunks(chunks, config, root=root, rebuild=True)

    retriever = create_retriever(config, root=root)
    pipeline = RAGPipeline(config, retriever, MockLLM())
    result = pipeline.ask("Should I invest in crypto for higher returns?")

    assert result.refused is True
    assert result.refusal_reason == RefusalReason.ADVICE_BOUNDARY
