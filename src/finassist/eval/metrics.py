from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from finassist.eval.dataset import GoldExample
from finassist.index.retriever import Retriever
from finassist.rag.pipeline import RAGPipeline


@dataclass
class ExampleResult:
    question: str
    category: str
    should_refuse: bool
    refused: bool
    refusal_reason: str | None
    latency_ms: int
    retrieved_doc_ids: list[str]
    expected_doc_ids: list[str]
    recall_at_k: dict[int, float] = field(default_factory=dict)
    answer: str = ""


@dataclass
class EvalMetrics:
    num_examples: int
    recall_at_k: dict[int, float]
    mrr: float
    refusal_accuracy: float
    in_domain_answer_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    mode: str


def _recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    if not expected_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    return 1.0 if any(doc_id in top_k for doc_id in expected_doc_ids) else 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return float(ordered[index])


def evaluate_retrieval(
    retriever: Retriever,
    examples: list[GoldExample],
    *,
    k_values: list[int],
) -> tuple[list[ExampleResult], EvalMetrics]:
    results: list[ExampleResult] = []

    for example in examples:
        if example.should_refuse:
            continue

        chunks = retriever.search(example.question)
        retrieved_doc_ids = [chunk.doc_id for chunk in chunks]
        recall_scores = {
            k: _recall_at_k(retrieved_doc_ids, example.expected_doc_ids, k) for k in k_values
        }

        results.append(
            ExampleResult(
                question=example.question,
                category=example.category,
                should_refuse=example.should_refuse,
                refused=False,
                refusal_reason=None,
                latency_ms=0,
                retrieved_doc_ids=retrieved_doc_ids,
                expected_doc_ids=example.expected_doc_ids,
                recall_at_k=recall_scores,
            )
        )

    recall_at_k = {k: _mean([r.recall_at_k[k] for r in results]) for k in k_values}
    mrr_values: list[float] = []
    for result in results:
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(result.retrieved_doc_ids, start=1):
            if doc_id in result.expected_doc_ids:
                reciprocal_rank = 1.0 / rank
                break
        mrr_values.append(reciprocal_rank)

    metrics = EvalMetrics(
        num_examples=len(results),
        recall_at_k=recall_at_k,
        mrr=_mean(mrr_values),
        refusal_accuracy=0.0,
        in_domain_answer_rate=0.0,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        mode="retrieval",
    )
    return results, metrics


def evaluate_pipeline(
    pipeline: RAGPipeline,
    examples: list[GoldExample],
    *,
    k_values: list[int],
    mode: str = "rag",
) -> tuple[list[ExampleResult], EvalMetrics]:
    results: list[ExampleResult] = []

    for example in examples:
        if mode == "no_rag":
            response = pipeline.ask_without_rag(example.question)
            chunks = []
        else:
            response = pipeline.ask(example.question)
            chunks = pipeline.retriever.search(example.question)

        retrieved_doc_ids = [chunk.doc_id for chunk in chunks]
        recall_scores = {
            k: _recall_at_k(retrieved_doc_ids, example.expected_doc_ids, k) for k in k_values
        }

        results.append(
            ExampleResult(
                question=example.question,
                category=example.category,
                should_refuse=example.should_refuse,
                refused=response.refused,
                refusal_reason=(
                    response.refusal_reason.value if response.refusal_reason else None
                ),
                latency_ms=response.latency_ms,
                retrieved_doc_ids=retrieved_doc_ids,
                expected_doc_ids=example.expected_doc_ids,
                recall_at_k=recall_scores,
                answer=response.answer,
            )
        )

    refusal_predictions = [result.refused for result in results]
    refusal_gold = [example.should_refuse for example in examples]
    refusal_accuracy = _mean(
        [
            float(pred == gold)
            for pred, gold in zip(refusal_predictions, refusal_gold, strict=True)
        ]
    )

    in_domain = [
        result
        for result, example in zip(results, examples, strict=True)
        if not example.should_refuse
    ]
    in_domain_answer_rate = _mean([0.0 if result.refused else 1.0 for result in in_domain])

    in_domain_recall = [r for r in results if not r.should_refuse]
    recall_at_k = {
        k: _mean([r.recall_at_k.get(k, 0.0) for r in in_domain_recall]) for k in k_values
    }
    latencies = [result.latency_ms for result in results]

    metrics = EvalMetrics(
        num_examples=len(results),
        recall_at_k=recall_at_k,
        mrr=0.0,
        refusal_accuracy=refusal_accuracy,
        in_domain_answer_rate=in_domain_answer_rate,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        mode=mode,
    )
    return results, metrics
