from __future__ import annotations

import argparse
from pathlib import Path

from finassist.config import get_app_config
from finassist.index.builder import build_index_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed processed chunks and persist a Chroma vector index."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--chunks-path",
        type=str,
        default=None,
        help="Override path to chunks.jsonl (default: configs/index.chunks_path).",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Upsert into existing collection without deleting it first.",
    )
    parser.add_argument(
        "--backend",
        choices=["sentence_transformers", "tfidf"],
        default=None,
        help="Override embedding backend from config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = get_app_config(args.config) if args.config else get_app_config()
    if args.backend:
        config.index.embedding_backend = args.backend

    chunks_path = Path(args.chunks_path) if args.chunks_path else None
    result = build_index_from_config(
        config,
        rebuild=not args.no_rebuild,
        chunks_path=chunks_path,
    )

    print(f"Indexed {result.num_chunks} chunks into '{result.collection_name}'")
    print(f"Embedding model: {result.embedding_model}")
    print(f"Chroma persist dir: {result.persist_dir}")


if __name__ == "__main__":
    main()
