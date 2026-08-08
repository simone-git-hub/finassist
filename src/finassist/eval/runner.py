from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from finassist.config import AppConfig, get_app_config, project_root
from finassist.eval.dataset import load_gold_dataset
from finassist.eval.metrics import EvalMetrics, evaluate_pipeline, evaluate_retrieval
from finassist.index.retriever import create_retriever
from finassist.rag.pipeline import build_pipeline


@dataclass
class EvalReport:
    generated_at: str
    config_path: str
    retrieval: EvalMetrics
    rag: EvalMetrics
    no_rag: EvalMetrics
    ablation: dict[str, float]


def run_evaluation(
    config: AppConfig | None = None,
    *,
    gold_path: Path | None = None,
    output_dir: Path | None = None,
    root: Path | None = None,
) -> EvalReport:
    cfg = config or get_app_config()
    examples = load_gold_dataset(gold_path)
    k_values = cfg.eval.k_values

    retriever = create_retriever(cfg, root=root)
    _, retrieval_metrics = evaluate_retrieval(retriever, examples, k_values=k_values)

    pipeline = build_pipeline(cfg, root=root)
    _, rag_metrics = evaluate_pipeline(
        pipeline, examples, k_values=k_values, mode="rag"
    )
    _, no_rag_metrics = evaluate_pipeline(
        pipeline, examples, k_values=k_values, mode="no_rag"
    )

    ablation = {
        "in_domain_answer_rate_delta_rag_minus_no_rag": (
            rag_metrics.in_domain_answer_rate - no_rag_metrics.in_domain_answer_rate
        ),
        "refusal_accuracy_delta_rag_minus_no_rag": (
            rag_metrics.refusal_accuracy - no_rag_metrics.refusal_accuracy
        ),
    }

    report = EvalReport(
        generated_at=datetime.now(tz=UTC).isoformat(),
        config_path=str(gold_path or project_root() / cfg.eval.gold_path),
        retrieval=retrieval_metrics,
        rag=rag_metrics,
        no_rag=no_rag_metrics,
        ablation=ablation,
    )

    out_dir = output_dir or (root or project_root()) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"eval_{stamp}.json"
    output_path.write_text(json.dumps(_report_to_dict(report), indent=2), encoding="utf-8")

    latest_path = out_dir / "eval_latest.json"
    latest_path.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")

    return report


def _report_to_dict(report: EvalReport) -> dict:
    return {
        "generated_at": report.generated_at,
        "config_path": report.config_path,
        "retrieval": asdict(report.retrieval),
        "rag": asdict(report.rag),
        "no_rag": asdict(report.no_rag),
        "ablation": report.ablation,
    }


def format_report_table(report: EvalReport) -> str:
    lines = [
        "| Mode | Recall@5 | Refusal acc. | In-domain answer rate | p50 latency (ms) | p95 latency (ms) |",
        "|------|----------|--------------|------------------------|------------------|------------------|",
    ]

    def row(label: str, metrics: EvalMetrics) -> str:
        recall5 = metrics.recall_at_k.get(5, 0.0)
        return (
            f"| {label} | {recall5:.2f} | {metrics.refusal_accuracy:.2f} | "
            f"{metrics.in_domain_answer_rate:.2f} | {metrics.latency_p50_ms:.0f} | "
            f"{metrics.latency_p95_ms:.0f} |"
        )

    lines.append(row("Retrieval-only", report.retrieval))
    lines.append(row("RAG", report.rag))
    lines.append(row("No-RAG baseline", report.no_rag))
    lines.append("")
    lines.append("### Ablation (RAG − No-RAG)")
    lines.append(
        f"- In-domain answer rate delta: **{report.ablation['in_domain_answer_rate_delta_rag_minus_no_rag']:+.2f}**"
    )
    lines.append(
        f"- Refusal accuracy delta: **{report.ablation['refusal_accuracy_delta_rag_minus_no_rag']:+.2f}**"
    )
    if report.retrieval.num_examples:
        lines.append(f"- Retrieval MRR: **{report.retrieval.mrr:.2f}**")
    return "\n".join(lines)
