from __future__ import annotations

import argparse
import json

from finassist.config import get_app_config, get_settings
from finassist.index.retriever import create_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a semantic search query against the vector index.")
    parser.add_argument("query", type=str, help="Natural language question.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = get_app_config(args.config) if args.config else get_app_config()
    retriever = create_retriever(config)

    top_k = args.top_k or config.retrieval.top_k
    results = retriever.search(args.query, top_k=top_k)
    min_score = get_settings().min_retrieval_score or config.retrieval.min_score

    if args.json:
        payload = [result.model_dump() for result in results]
        print(json.dumps(payload, indent=2))
        return

    print(f"Query: {args.query}\n")
    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        flag = "OK" if result.score >= min_score else "LOW"
        print(f"[{rank}] score={result.score:.3f} ({flag}) doc={result.doc_id}")
        print(f"    title: {result.title}")
        print(f"    text:  {result.text[:180]}{'...' if len(result.text) > 180 else ''}")
        print()


if __name__ == "__main__":
    main()
