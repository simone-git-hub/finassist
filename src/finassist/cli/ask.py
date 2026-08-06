from __future__ import annotations

import argparse
import json

from finassist.config import get_app_config
from finassist.rag.pipeline import build_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question via the RAG copilot pipeline.")
    parser.add_argument("question", type=str, help="Natural language question.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = get_app_config(args.config) if args.config else get_app_config()
    pipeline = build_pipeline(config)
    result = pipeline.ask(args.question, top_k=args.top_k)

    if args.json:
        print(json.dumps(result.model_dump(), indent=2))
        return

    if result.refused:
        print(f"REFUSED ({result.refusal_reason}): {result.answer}")
        return

    print(result.answer)
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"  [{source.index}] {source.title} (score={source.score:.3f})")


if __name__ == "__main__":
    main()
