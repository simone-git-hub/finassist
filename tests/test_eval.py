from finassist.eval.dataset import load_gold_dataset
from finassist.eval.runner import run_evaluation
from finassist.index.builder import build_index_from_chunks, load_chunks_jsonl
from finassist.index.retriever import create_retriever
from finassist.ingest.pipeline import run_ingest_pipeline
from finassist.rag.llm import MockLLM
from finassist.rag.pipeline import RAGPipeline


def test_eval_runner_on_offline_project(rag_project_tree) -> None:
    root, config = rag_project_tree
    config.ingest.manifest_path = "data/raw/manifest.offline.csv"
    config.eval.gold_path = "data/eval/qa_gold_offline.jsonl"

    run_ingest_pipeline(config, root=root)
    chunks = load_chunks_jsonl(root / config.index.chunks_path)
    build_index_from_chunks(chunks, config, root=root, rebuild=True)

    report = run_evaluation(
        config,
        gold_path=root / "data" / "eval" / "qa_gold_offline.jsonl",
        output_dir=root / "results",
        root=root,
    )

    assert report.retrieval.recall_at_k[5] >= 0.85
    assert report.rag.refusal_accuracy >= 0.65
    assert report.rag.num_examples == 11


def test_load_gold_dataset_has_examples() -> None:
    examples = load_gold_dataset()
    assert len(examples) >= 10
    assert any(example.should_refuse for example in examples)


def test_no_rag_baseline_answers_without_sources(rag_project_tree) -> None:
    root, config = rag_project_tree
    config.ingest.manifest_path = "data/raw/manifest.offline.csv"

    run_ingest_pipeline(config, root=root)
    chunks = load_chunks_jsonl(root / config.index.chunks_path)
    build_index_from_chunks(chunks, config, root=root, rebuild=True)

    retriever = create_retriever(config, root=root)
    pipeline = RAGPipeline(config, retriever, MockLLM())
    response = pipeline.ask_without_rag("How do I freeze my card?")

    assert response.refused is False
    assert response.sources == []
